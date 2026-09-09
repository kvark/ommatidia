//! Train and evaluate the lobe-separated recurrent reconstructor on full sequences.
use ommatidia::{
    dataset, metrics,
    transport::{Config, Frame, Target, cpu, graph, native},
};
use serde::Serialize;
use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};
type EvaluationHistory = ([Vec<f32>; 2], Vec<f32>, Vec<ommatidia::temporal::Surface>);
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[path = "transport/oracle_report.rs"]
mod oracle_report;

#[derive(Clone)]
struct Corpus {
    frames: Vec<(Frame, Target)>,
    length: usize,
    provenance: serde_json::Value,
}
impl Corpus {
    fn load(path: &Path, config: Config) -> Result<Self> {
        let provenance: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.with_extension("transport.json"))?)?;
        if provenance["matching_path_depth"] != true
            || provenance["input_estimator"] != "independent-paths"
        {
            return Err(
                "transport training requires verified, matched independent-path captures".into(),
            );
        }
        let mut reader = dataset::Reader::open(path)?;
        let layout = *reader.layout();
        if provenance["records"].as_u64() != Some(reader.len() as u64) {
            return Err("provenance record count differs from dataset".into());
        }
        if reader.sequence_length() < 2 || reader.is_empty() {
            return Err("a nonempty sequence dataset is required".into());
        }
        let mut frames = Vec::new();
        for index in 0..reader.len() {
            let sample = reader.sample(index)?;
            frames.push((
                Frame::from_sample(&sample, layout, config)?,
                Target::from_sample(&sample, layout, config)?,
            ));
        }
        Ok(Self {
            frames,
            length: reader.sequence_length(),
            provenance,
        })
    }
    fn combine(corpora: &[Self]) -> Result<Self> {
        let first = corpora.first().ok_or("empty capture list")?;
        if corpora
            .iter()
            .any(|c| c.length != first.length || c.frames[0].0.low != first.frames[0].0.low)
        {
            return Err("capture sequence/extent mismatch".into());
        }
        Ok(Self {
            frames: corpora.iter().flat_map(|c| c.frames.clone()).collect(),
            length: first.length,
            provenance: serde_json::json!({"captures":corpora.iter().map(|c|&c.provenance).collect::<Vec<_>>()}),
        })
    }
    fn disjoint(&self, other: &Self) -> Result<()> {
        let ids = |p: &serde_json::Value| -> Result<Vec<u64>> {
            Ok(p["scene_seeds"]
                .as_array()
                .ok_or("capture lacks scene-seed provenance; regenerate it")?
                .iter()
                .map(|v| v.as_u64().ok_or("invalid scene seed"))
                .collect::<std::result::Result<_, _>>()?)
        };
        let a = ids(&self.provenance)?;
        let b = ids(&other.provenance)?;
        if a.iter().any(|v| b.contains(v)) {
            return Err("training/evaluation scene seeds overlap".into());
        }
        let families =
            |p: &serde_json::Value| p["family_ids"].as_array().cloned().unwrap_or_default();
        let fa = families(&self.provenance);
        let fb = families(&other.provenance);
        if fa.iter().any(|f| fb.contains(f)) {
            return Err("training/evaluation catalog families overlap".into());
        }
        Ok(())
    }
}
fn paired_noise(corpora: &[Corpus]) -> Result<()> {
    let first = corpora.first().ok_or("missing paired captures")?;
    let mut ranges = Vec::new();
    for corpus in corpora {
        if corpus.length != first.length || corpus.frames.len() != first.frames.len() {
            return Err("paired capture layout mismatch".into());
        }
        let p = &corpus.provenance;
        if !p["reference_from"].is_null() {
            return Err("construction requires fresh references".into());
        }
        for key in [
            "scene_seeds",
            "family_ids",
            "capture_seed",
            "input_max_bounces",
            "reference_max_bounces",
            "canonical_frames",
            "reference_sample_offset",
            "input_frames",
        ] {
            if p[key] != first.provenance[key] {
                return Err(format!("paired provenance differs: {key}").into());
            }
        }
        let start = p["input_sample_offset"]
            .as_u64()
            .ok_or("missing input offset")?;
        let count = p["input_frames"]
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or("missing input count")?;
        let end = start
            .checked_add(
                count
                    .checked_mul(corpus.frames.len() as u64)
                    .ok_or("sample range overflow")?,
            )
            .ok_or("sample range overflow")?;
        if end
            > p["reference_sample_offset"]
                .as_u64()
                .ok_or("missing reference offset")?
        {
            return Err("reference overlaps input sample range".into());
        }
        if ranges.iter().any(|&(a, b)| start < b && a < end) {
            return Err("overlapping path streams".into());
        }
        ranges.push((start, end));
        for ((f, t), (a, b)) in corpus.frames.iter().zip(&first.frames) {
            if f.low != a.low
                || f.jitter != a.jitter
                || t.rgb != b.rgb
                || t.lobes != b.lobes
                || f.surfaces != a.surfaces
                || f.rays.len() != a.rays.len()
                || f.rays.iter().zip(&a.rays).any(|(r, s)| {
                    r.normal_depth != s.normal_depth || r.albedo_roughness != s.albedo_roughness
                })
            {
                return Err("paired capture changed reference or observed geometry".into());
            }
        }
    }
    Ok(())
}
fn save_png(path: &Path, rgb: &[f32], extent: [u32; 2]) -> Result<()> {
    let bytes: Vec<_> = rgb
        .iter()
        .map(|&v| {
            let v = ommatidia::transform::compress(v);
            let s = if v <= 0.0031308 {
                12.92 * v
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            };
            (s.clamp(0.0, 1.0) * 255.0).round() as u8
        })
        .collect();
    let mut encoder = png::Encoder::new(std::fs::File::create(path)?, extent[0], extent[1]);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&bytes)?;
    Ok(())
}
#[derive(Default, Serialize)]
struct Score {
    frames: usize,
    psnr: f64,
    ssim: f64,
    low_frequency_psnr: f64,
    relative_mse: f64,
    linear_mse: f64,
    energy_ratio: f64,
    detail_ratio: f64,
    temporal_mse: f64,
    temporal_frames: usize,
    reset_psnr: f64,
    resets: usize,
}
impl Score {
    fn add(&mut self, image: &[f32], target: &[f32], extent: [u32; 2], reset: bool) {
        self.frames += 1;
        let psnr = -10.0 * (metrics::error(image, target) as f64).max(1e-20).log10();
        self.psnr += psnr;
        self.ssim += metrics::ssim(image, target, extent[0] as usize, extent[1] as usize) as f64;
        self.low_frequency_psnr += -10.0
            * metrics::low_frequency_error(
                image,
                target,
                extent[0] as usize,
                extent[1] as usize,
                8,
            )
            .max(1e-20)
            .log10();
        self.relative_mse += metrics::relative_error(image, target);
        self.linear_mse += image
            .iter()
            .zip(target)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / image.len() as f64;
        self.energy_ratio += image.iter().map(|v| *v as f64).sum::<f64>()
            / target.iter().map(|v| *v as f64).sum::<f64>().max(1e-12);
        self.detail_ratio += metrics::detail(image, extent[0] as usize, extent[1] as usize)
            / metrics::detail(target, extent[0] as usize, extent[1] as usize).max(1e-12);
        if reset {
            self.reset_psnr += psnr;
            self.resets += 1;
        }
    }
    fn finish(&mut self) {
        let n = self.frames.max(1) as f64;
        self.psnr /= n;
        self.ssim /= n;
        self.low_frequency_psnr /= n;
        self.relative_mse /= n;
        self.linear_mse /= n;
        self.energy_ratio /= n;
        self.detail_ratio /= n;
        self.temporal_mse /= self.temporal_frames.max(1) as f64;
        self.reset_psnr /= self.resets.max(1) as f64;
    }
}
fn surfaces(frame: &Frame) -> Vec<ommatidia::temporal::Surface> {
    frame
        .surfaces
        .iter()
        .map(|s| ommatidia::temporal::Surface {
            depth: s.normal_depth[3],
            normal: s.normal_depth[..3].try_into().unwrap(),
            albedo: s.albedo_roughness[..3].try_into().unwrap(),
        })
        .collect()
}
fn evaluate(
    corpus: &Corpus,
    config: Config,
    learned: &mut native::Native,
    baseline: &mut native::Native,
    out: &Path,
    candidate_oracle: bool,
) -> Result<serde_json::Value> {
    let mut oracle_frames = Vec::new();
    let mut scores = [Score::default(), Score::default()];
    let mut previous: Option<EvaluationHistory> = None;
    let mut rows = std::fs::File::create(out.join("frames.csv"))?;
    writeln!(rows, "sequence,frame,baseline_psnr,learned_psnr")?;
    for (index, (frame, target)) in corpus.frames.iter().enumerate() {
        let reset = index % corpus.length == 0;
        if reset {
            learned.reset();
            baseline.reset();
            previous = None;
        }
        let images = [baseline.process(frame)?, learned.process(frame)?];
        let extent = frame.low.map(|v| v * config.scale);
        let current = surfaces(frame);
        let motion: Vec<_> = frame
            .surfaces
            .iter()
            .flat_map(|s| s.motion[..2].iter().copied())
            .collect();
        for (score, image) in scores.iter_mut().zip(&images) {
            if image.iter().any(|v| !v.is_finite()) {
                return Err("non-finite reconstruction".into());
            }
            score.add(image, &target.rgb, extent, reset);
        }
        if let Some((old, reference, old_surfaces)) = &previous {
            let warp = ommatidia::temporal::Reprojection {
                motion: &motion,
                current: &current,
                previous: old_surfaces,
                rejection: Default::default(),
            };
            for k in 0..2 {
                if let Some(e) = metrics::temporal_error(
                    [&images[k], &old[k]],
                    [&target.rgb, reference],
                    warp,
                    None,
                    frame.low.map(|v| v as usize),
                    config.scale as usize,
                ) {
                    scores[k].temporal_mse += e.mean();
                    scores[k].temporal_frames += 1;
                }
            }
        }
        writeln!(
            rows,
            "{},{},{:.6},{:.6}",
            index / corpus.length,
            index % corpus.length,
            -10.0 * metrics::error(&images[0], &target.rgb).max(1e-20).log10(),
            -10.0 * metrics::error(&images[1], &target.rgb).max(1e-20).log10()
        )?;
        // All frames are retained: evaluation is not a cherry-picked screenshot.
        let prefix = format!("{:03}-{:03}", index / corpus.length, index % corpus.length);
        if candidate_oracle {
            let mut report = oracle_report::frame(learned, frame, target, config, out, &prefix)?;
            report["sequence"] = serde_json::json!(index / corpus.length);
            report["frame"] = serde_json::json!(index % corpus.length);
            oracle_frames.push(report);
        }
        for (name, image) in [
            ("base", &images[0]),
            ("learned", &images[1]),
            ("reference", &target.rgb),
        ] {
            save_png(&out.join(format!("{prefix}-{name}.png")), image, extent)?;
        }
        previous = Some((images, target.rgb.clone(), current));
    }
    if candidate_oracle {
        std::fs::write(
            out.join("candidate-oracle.json"),
            serde_json::to_vec_pretty(&oracle_frames)?,
        )?;
    }
    scores.iter_mut().for_each(Score::finish);
    Ok(
        serde_json::json!({"baseline":scores[0],"learned":scores[1],"metric_space":"PSNR/SSIM: x/(1+x); energy: scene-linear; PNG: same compression then sRGB", "speed_claim":false}),
    )
}
fn main() -> Result<()> {
    env_logger::init();
    let mut data = Vec::new();
    let mut eval = Vec::new();
    let mut construction_noise = false;
    let mut rgb_loss = false;
    let mut out = PathBuf::from("runs/transport");
    let mut steps = 128usize;
    let mut unroll = 2usize;
    let mut channels = 8;
    let mut mixture = None::<ommatidia::transport::mixture::Mode>;
    let mut seed = 7u64;
    let mut rate = 0.001f32;
    let mut eval_only = false;
    let mut candidate_oracle = false;
    let mut projected_weight = None::<f32>;
    let mut fixed_exposure_loss = false;
    let mut weights = graph::LossWeights::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!(
                "transport --data TRAIN.omd --eval-data HOLDOUT.omd [--out DIR] [--steps 128] [--unroll 2] [--channels 8] [--mixture softplus|masked-softmax] [--seed 7] [--lr 0.001] [--eval-only] [--candidate-oracle] [--fixed-exposure-loss] [--projected-weight F] [--rgb-loss] [--construction-noise]\n  --compressed-weight F [1] --physical-weight F [0.1] --low-frequency-weight F [0.05]\n  --confidence-weight F [0.01] --temporal-weight F [0.01]\nRepeat --data/--eval-data for multiple captures. Matched transport, split radiance and HR surfaces required. Scene seeds must be disjoint unless --construction-noise verifies equal truth and nonoverlapping path streams."
            );
            return Ok(());
        }
        if arg == "--construction-noise" {
            construction_noise = true;
            continue;
        }
        if arg == "--rgb-loss" {
            rgb_loss = true;
            continue;
        }
        if arg == "--candidate-oracle" {
            candidate_oracle = true;
            continue;
        }
        if arg == "--fixed-exposure-loss" {
            fixed_exposure_loss = true;
            continue;
        }
        if arg == "--eval-only" {
            eval_only = true;
            continue;
        }
        let v = args.next().ok_or(format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--data" => data.push(PathBuf::from(v)),
            "--eval-data" => eval.push(PathBuf::from(v)),
            "--out" => out = v.into(),
            "--steps" => steps = v.parse()?,
            "--unroll" => unroll = v.parse()?,
            "--channels" => channels = v.parse()?,
            "--mixture" => mixture = Some(v.parse()?),
            "--seed" => seed = v.parse()?,
            "--lr" => rate = v.parse()?,
            "--projected-weight" => projected_weight = Some(v.parse()?),
            "--compressed-weight" => weights.compressed = v.parse()?,
            "--physical-weight" => weights.physical = v.parse()?,
            "--low-frequency-weight" => weights.low_frequency = v.parse()?,
            "--confidence-weight" => weights.confidence = v.parse()?,
            "--temporal-weight" => weights.temporal = v.parse()?,
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    weights.validate()?;
    if eval_only && mixture.is_some() {
        return Err("evaluation reads mixture from the checkpoint sidecar".into());
    }
    if rgb_loss && (!fixed_exposure_loss || eval_only) {
        return Err("--rgb-loss requires --fixed-exposure-loss and training".into());
    }
    if !eval_only && out.join("model.safetensors").exists() {
        return Err("refusing to overwrite an existing checkpoint".into());
    }
    if projected_weight.is_some_and(|w| !w.is_finite() || w < 0.0) {
        return Err("projected weight must be finite and nonnegative".into());
    }
    if projected_weight.is_some() && eval_only {
        return Err("projected targets are training-only".into());
    }
    if !(1..=8).contains(&unroll) || !rate.is_finite() || rate <= 0.0 {
        return Err("invalid unroll or learning rate".into());
    }
    std::fs::create_dir_all(&out)?;
    let config = if eval_only {
        ron::from_str(&std::fs::read_to_string(out.join("model.transport.ron"))?)?
    } else {
        Config {
            version: mixture.unwrap_or_default().version(),
            mixture: mixture.unwrap_or_default(),
            channels,
            ..Config::default()
        }
    };
    if eval.is_empty() {
        return Err("--eval-data required".into());
    }
    let held_corpora = eval
        .iter()
        .map(|p| Corpus::load(p, config))
        .collect::<Result<Vec<_>>>()?;
    let holdout = Corpus::combine(&held_corpora)?;
    let low = holdout.frames[0].0.low;
    let context = ommatidia::gpu::create_context(None, false);
    let mut learned = native::Native::new(Arc::clone(&context), config, low)?;
    let mut baseline = native::Native::new(
        Arc::clone(&context),
        Config {
            version: 1,
            mixture: Default::default(),
            ..config
        },
        low,
    )?;
    let checkpoint = out.join("model.safetensors");
    if eval_only {
        learned.session.load_checkpoint(&checkpoint)?;
    } else {
        if data.is_empty() {
            return Err("--data required".into());
        }
        let training = data
            .iter()
            .map(|p| Corpus::load(p, config))
            .collect::<Result<Vec<_>>>()?;
        if construction_noise {
            paired_noise(
                &training
                    .iter()
                    .chain(&held_corpora)
                    .cloned()
                    .collect::<Vec<_>>(),
            )?;
        } else {
            for train in &training {
                for held in &held_corpora {
                    train.disjoint(held)?;
                }
            }
        }
        let train = Corpus::combine(&training)?;
        if unroll > train.length || train.frames.iter().any(|(f, _)| f.low != low) {
            return Err("unroll exceeds sequence or extents differ".into());
        }
        let network =
            graph::build_objective(config, low, unroll, projected_weight.is_some(), rgb_loss)?;
        let mut session = ommatidia::gpu::training_session(&network.graph, Arc::clone(&context));
        network.initialize(&mut session, seed);
        let mut rng = ommatidia::rng::Rng::new(seed);
        let mut losses = std::fs::File::create(out.join("loss.csv"))?;
        writeln!(losses, "update,loss")?;
        for update in 0..steps {
            session.wait();
            learned.sync_parameters(&session);
            learned.reset();
            let sequence = rng.below((train.frames.len() / train.length) as u32) as usize;
            let start = rng.below((train.length - unroll + 1) as u32) as usize;
            let offset = sequence * train.length;
            for (frame, _) in &train.frames[offset..offset + start] {
                learned.process(frame)?;
            }
            for slot in 0..unroll {
                let (frame, target) = &train.frames[offset + start + slot];
                let old = if start + slot == 0 {
                    Vec::new()
                } else {
                    learned.read_state()
                };
                let prepared = cpu::prepare(frame, &old, config);
                graph::feed(&mut session, &format!("f{slot}"), &prepared, target, slot);
                if rgb_loss {
                    graph::feed_rgb(&mut session, &format!("f{slot}"), frame, target, config);
                }
                if let Some(weight) = projected_weight {
                    let candidates = ommatidia::transport::oracle::Candidates {
                        spatial: prepared.candidates.clone(),
                        history: prepared.history.clone(),
                        prior: prepared.prior.clone(),
                        selected: Vec::new(),
                    };
                    let projection =
                        ommatidia::transport::oracle::reconstruct(&candidates, &target.lobes)?;
                    session.set_input(&format!("f{slot}.projected"), &projection.lobes);
                    session.set_input("loss.projected_weight", &[weight]);
                }
                if fixed_exposure_loss {
                    // Override only loss inputs: same graph, queries and initialization.
                    session.set_input(
                        &format!("f{slot}.loss_scale"),
                        &vec![config.exposure; target.lobes.len()],
                    );
                }
                learned.process(frame)?;
            }
            weights.feed(&mut session);
            let fraction = update as f32 / steps.max(1) as f32;
            session.set_adam(
                rate * (0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * fraction).cos())),
                0.9,
                0.999,
                1e-8,
            );
            session.step();
            session.wait();
            let loss = session.read_loss();
            if !loss.is_finite() {
                return Err("training became non-finite".into());
            }
            writeln!(losses, "{update},{loss}")?;
            if update % 16 == 0 {
                println!("update {update}/{steps}: loss {loss:.7}");
            }
        }
        session.save_checkpoint(&checkpoint)?;
        std::fs::write(
            out.join("model.transport.ron"),
            ron::ser::to_string_pretty(&config, ron::ser::PrettyConfig::default())?,
        )?;
        // Evaluate the serialized asset, not the still-live training graph.
        learned.session.load_checkpoint(&checkpoint)?;
        std::fs::write(
            out.join("training.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"steps":steps,"unroll":unroll,"seed":seed,"learning_rate":rate,"fixed_exposure_loss":fixed_exposure_loss,"loss_weights":weights,"projected_weight":projected_weight,"rgb_loss":rgb_loss,"construction_noise":construction_noise,"training":train.provenance,"evaluation":holdout.provenance}),
            )?,
        )?;
    }
    let mut report = evaluate(
        &holdout,
        config,
        &mut learned,
        &mut baseline,
        &out,
        candidate_oracle,
    )?;
    report["role"] = serde_json::json!(if construction_noise {
        "same-scenes held-noise construction"
    } else {
        "scene-disjoint evaluation"
    });
    if construction_noise && !eval_only {
        let fit = Corpus::combine(
            &data
                .iter()
                .map(|p| Corpus::load(p, config))
                .collect::<Result<Vec<_>>>()?,
        )?;
        let dir = out.join("fitting");
        std::fs::create_dir_all(&dir)?;
        let result = evaluate(&fit, config, &mut learned, &mut baseline, &dir, false)?;
        std::fs::write(
            dir.join("quality.json"),
            serde_json::to_vec_pretty(&result)?,
        )?;
    }
    std::fs::write(
        out.join("quality.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
