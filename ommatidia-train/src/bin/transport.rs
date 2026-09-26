//! Train and evaluate the lobe-separated recurrent reconstructor on full sequences.
use ommatidia::{
    dataset, metrics,
    transport::{Config, Frame, Target, graph, native},
};
use serde::Serialize;
use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};
type EvaluationHistory = ([Vec<f32>; 2], Vec<f32>, Vec<ommatidia::temporal::Surface>);
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn validate_capture(provenance: &serde_json::Value) -> Result<()> {
    if provenance["matching_path_depth"] != true
        || provenance["input_estimator"] != "independent-paths"
    {
        return Err(
            "transport training requires verified, matched independent-path captures".into(),
        );
    }
    if let Some(value) = provenance
        .get("minimum_catalog_visible_fraction")
        .filter(|v| !v.is_null())
    {
        let coverage = value
            .as_f64()
            .ok_or("invalid catalog coverage provenance")?;
        if !(0.01..=1.0).contains(&coverage) {
            return Err(
                "catalog coverage below 1%; fix the asset, camera or driver before training".into(),
            );
        }
    }
    Ok(())
}

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
        validate_capture(&provenance)?;
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
            let frame = Frame::from_sample(&sample, layout, config)?;
            let target = Target::from_sample(&sample, layout, config)?;
            if target
                .rgb
                .iter()
                .chain(&target.lobes)
                .any(|v| !v.is_finite() || *v < 0.0)
                || target.rgb.iter().all(|v| *v <= 1e-6)
            {
                return Err(format!("{} record {index}: invalid or entirely black reference; inspect the capture camera", path.display()).into());
            }
            frames.push((frame, target));
        }
        Ok(Self {
            frames,
            length: reader.sequence_length(),
            provenance,
        })
    }
    fn combine(corpora: &mut [Self]) -> Result<Self> {
        let first = corpora.first().ok_or("empty capture list")?;
        if corpora
            .iter()
            .any(|c| c.length != first.length || c.frames[0].0.low != first.frames[0].0.low)
        {
            return Err("capture sequence/extent mismatch".into());
        }
        let length = first.length;
        let provenance = serde_json::json!({"captures":corpora.iter().map(|c|&c.provenance).collect::<Vec<_>>()});
        Ok(Self {
            frames: corpora
                .iter_mut()
                .flat_map(|c| std::mem::take(&mut c.frames))
                .collect(),
            length,
            provenance,
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
    gradient_mse: f64,
    temporal_mse: f64,
    temporal_frames: usize,
    reset_psnr: f64,
    resets: usize,
    rejected_history_mse: Option<f64>,
    rejected_history_pixels: usize,
    history_pixels: usize,
    rejected_history_fraction: Option<f64>,
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
        self.gradient_mse +=
            metrics::gradient_error(image, target, extent[0] as usize, extent[1] as usize);
        if reset {
            self.reset_psnr += psnr;
            self.resets += 1;
        }
    }
    fn add_rejected_history(&mut self, image: &[f32], target: &[f32], mask: &[bool]) {
        self.history_pixels += mask.len();
        if let Some(mse) = metrics::masked_error(image, target, mask) {
            let pixels = mask.iter().filter(|&&v| v).count();
            *self.rejected_history_mse.get_or_insert(0.0) += mse * pixels as f64;
            self.rejected_history_pixels += pixels;
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
        self.gradient_mse /= n;
        self.temporal_mse /= self.temporal_frames.max(1) as f64;
        self.reset_psnr /= self.resets.max(1) as f64;
        if let Some(total) = &mut self.rejected_history_mse {
            *total /= self.rejected_history_pixels as f64;
        }
        self.rejected_history_fraction = (self.history_pixels != 0)
            .then(|| self.rejected_history_pixels as f64 / self.history_pixels as f64);
    }
}

fn rejected_history_mask(validity: &[f32], low: [u32; 2], config: Config) -> Vec<bool> {
    let width = (low[0] * config.scale) as usize;
    let n = (low[0] * low[1] * config.scale.pow(2)) as usize;
    assert_eq!(validity.len(), 2 * n);
    assert!(
        validity.iter().all(|&v| v == 0.0 || v == 1.0),
        "invalid reprojection mask"
    );
    (0..n)
        .map(|i| (0..2).any(|lobe| validity[config.index(low, lobe, i % width, i / width)] == 0.0))
        .collect()
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

fn reference_lobes(frame: &Frame, target: &Target, config: Config) -> [Vec<f32>; 2] {
    let width = (frame.low[0] * config.scale) as usize;
    std::array::from_fn(|lobe| {
        (0..frame.surfaces.len())
            .flat_map(|i| {
                (0..3).map(move |c| {
                    target.lobes[config.index(frame.low, lobe * 3 + c, i % width, i / width)]
                })
            })
            .collect()
    })
}

fn compose_lobes(frame: &Frame, lobes: &[Vec<f32>; 2]) -> Vec<f32> {
    frame
        .surfaces
        .iter()
        .enumerate()
        .flat_map(|(i, s)| {
            (0..3).map(move |c| {
                lobes[0][i * 3 + c] * s.albedo_roughness[c] + lobes[1][i * 3 + c] + s.emission[c]
            })
        })
        .collect()
}

#[derive(Clone, Copy, Default)]
struct EvaluationOptions {
    reset_history: bool,
    save_lobes: bool,
}

fn evaluate(
    corpus: &Corpus,
    config: Config,
    learned: &mut native::Native,
    baseline: &mut native::Native,
    out: &Path,
    options: EvaluationOptions,
) -> Result<serde_json::Value> {
    let mut scores = [Score::default(), Score::default()];
    let mut previous: Option<EvaluationHistory> = None;
    let mut diagnostics = Vec::new();
    let mut rows = std::fs::File::create(out.join("frames.csv"))?;
    writeln!(rows, "sequence,frame,baseline_psnr,learned_psnr")?;
    for (index, (frame, target)) in corpus.frames.iter().enumerate() {
        let sequence_start = index % corpus.length == 0;
        let reset = sequence_start || options.reset_history;
        if sequence_start {
            previous = None;
        }
        if reset {
            learned.reset();
            baseline.reset();
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
        if !reset {
            for (k, model) in [&*baseline, &*learned].into_iter().enumerate() {
                let mask = rejected_history_mask(&model.read_history_validity(), frame.low, config);
                scores[k].add_rejected_history(&images[k], &target.rgb, &mask);
            }
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
        let truth = reference_lobes(frame, target, config);
        let composition = compose_lobes(frame, &truth);
        let mut diagnostic = serde_json::json!({
            "sequence": index / corpus.length,
            "frame": index % corpus.length,
            "reference_composition_mse": metrics::error(&composition, &target.rgb),
        });
        for (name, model) in [("baseline", &*baseline), ("learned", &*learned)] {
            let state = model.read_state();
            let lobes: [Vec<f32>; 2] = [
                state
                    .iter()
                    .flat_map(|s| s.diffuse[..3].iter().copied())
                    .collect(),
                state
                    .iter()
                    .flat_map(|s| s.specular[..3].iter().copied())
                    .collect(),
            ];
            let shaded = |values: &[f32]| -> Vec<f32> {
                values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v * frame.surfaces[i / 3].albedo_roughness[i % 3])
                    .collect()
            };
            diagnostic[name] = serde_json::json!({
                "diffuse_illumination_mse": metrics::error(&lobes[0], &truth[0]),
                "diffuse_radiance_mse": metrics::error(&shaded(&lobes[0]), &shaded(&truth[0])),
                "specular_radiance_mse": metrics::error(&lobes[1], &truth[1]),
                "mean_diffuse_age": state.iter().map(|s| f64::from(s.diffuse[3])).sum::<f64>() / state.len() as f64,
                "mean_specular_age": state.iter().map(|s| f64::from(s.specular[3])).sum::<f64>() / state.len() as f64,
            });
            if options.save_lobes {
                for (lobe, values) in ["diffuse", "specular"].into_iter().zip(&lobes) {
                    save_png(
                        &out.join(format!("{prefix}-{name}-{lobe}.png")),
                        values,
                        extent,
                    )?;
                }
            }
        }
        if options.save_lobes {
            for (lobe, values) in ["diffuse", "specular"].into_iter().zip(&truth) {
                save_png(
                    &out.join(format!("{prefix}-reference-{lobe}.png")),
                    values,
                    extent,
                )?;
            }
        }
        diagnostics.push(diagnostic);
        for (name, image) in [
            ("base", &images[0]),
            ("learned", &images[1]),
            ("reference", &target.rgb),
        ] {
            save_png(&out.join(format!("{prefix}-{name}.png")), image, extent)?;
        }
        previous = Some((images, target.rgb.clone(), current));
    }
    std::fs::write(
        out.join("diagnostics.json"),
        serde_json::to_vec_pretty(&diagnostics)?,
    )?;
    scores.iter_mut().for_each(Score::finish);
    Ok(
        serde_json::json!({"baseline":scores[0],"learned":scores[1],"history_mode":if options.reset_history { "reset-every-frame diagnostic" } else { "causal" }, "metric_space":"PSNR/SSIM/gradient/lobe MSE: x/(1+x); energy: scene-linear; PNG: same compression then sRGB", "rejected_history_space":"pixel-weighted compressed RGB MSE on non-reset pixels with unavailable reprojection in either lobe; includes disocclusions and out-of-frame motion, excludes reactive/learned gate suppression; empty regions are null", "speed_claim":false}),
    )
}
fn main() -> Result<()> {
    env_logger::init();
    let mut data = Vec::new();
    let mut eval = Vec::new();
    let mut out = PathBuf::from("runs/transport");
    let mut steps = 4000usize;
    let mut unroll = 2usize;
    let mut channels = 16;
    let mut seed = 7u64;
    let mut rate = 0.0003f32;
    let mut eval_only = false;
    let mut checkpoint_input = None::<PathBuf>;
    let mut eval_every = 500usize;
    let mut device_id = None;
    let mut weights = graph::LossWeights::default();
    let mut evaluation = EvaluationOptions::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!(
                "transport --data TRAIN.omd --eval-data DEV.omd --out DIR
  --steps N [4000] --unroll N [2] --channels N [16] --seed N [7]
  --lr F [0.0003] --eval-every N [500] --device-id ID
  --checkpoint FILE (weights-only warm start, or evaluation input)
  --eval-only (loads checkpoint sidecar; evaluation resolution may differ)
  --reset-history (eval-only diagnostic: reset the model before every frame)
  --save-lobes (save diffuse/specular images alongside per-frame diagnostics)
  --compressed-weight F [1] --physical-weight F [0.005]
  --low-frequency-weight F [0.01] --temporal-weight F [0.02]
  --lobe-weight F [0.5] (absolute diffuse/specular supervision)
Repeat data arguments for multiple captures. Training and development scene
seeds/catalog families must be disjoint. Use a separate final audit split."
            );
            return Ok(());
        }
        if arg == "--eval-only" {
            eval_only = true;
            continue;
        }
        if arg == "--reset-history" {
            evaluation.reset_history = true;
            continue;
        }
        if arg == "--save-lobes" {
            evaluation.save_lobes = true;
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
            "--seed" => seed = v.parse()?,
            "--lr" => rate = v.parse()?,
            "--eval-every" => eval_every = v.parse()?,
            "--checkpoint" => checkpoint_input = Some(v.into()),
            "--device-id" => device_id = Some(ommatidia::gpu::parse_device_id(&v)?),
            "--compressed-weight" => weights.compressed = v.parse()?,
            "--physical-weight" => weights.physical = v.parse()?,
            "--low-frequency-weight" => weights.low_frequency = v.parse()?,
            "--temporal-weight" => weights.temporal = v.parse()?,
            "--lobe-weight" => weights.lobes = v.parse()?,
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    weights.validate()?;
    if evaluation.reset_history && !eval_only {
        return Err("--reset-history is an evaluation-only diagnostic".into());
    }
    if !eval_only && out.join("model.safetensors").exists() {
        return Err("refusing to overwrite an existing checkpoint".into());
    }
    if !(1..=8).contains(&unroll) || !rate.is_finite() || rate <= 0.0 || steps == 0 {
        return Err("invalid unroll, step count or learning rate".into());
    }
    std::fs::create_dir_all(&out)?;
    let checkpoint = out.join("model.safetensors");
    if eval_only && checkpoint_input.is_none() {
        checkpoint_input = Some(checkpoint.clone());
    }
    let config = if let Some(path) = &checkpoint_input {
        ron::from_str(&std::fs::read_to_string(
            path.parent().unwrap().join("model.transport.ron"),
        )?)?
    } else {
        Config {
            channels,
            ..Config::default()
        }
    };
    if eval.is_empty() {
        return Err("--eval-data required".into());
    }
    let mut held_corpora = eval
        .iter()
        .map(|p| Corpus::load(p, config))
        .collect::<Result<Vec<_>>>()?;
    let holdout = Corpus::combine(&mut held_corpora)?;
    let low = holdout.frames[0].0.low;
    let context = ommatidia::gpu::create_context(device_id, false);
    let mut learned = native::Native::new(Arc::clone(&context), config, low)?;
    let mut baseline = native::Native::new(Arc::clone(&context), config, low)?;
    if eval_only {
        learned
            .session
            .load_checkpoint(checkpoint_input.as_ref().unwrap())?;
    } else {
        if data.is_empty() {
            return Err("--data required".into());
        }
        let mut training = data
            .iter()
            .map(|p| Corpus::load(p, config))
            .collect::<Result<Vec<_>>>()?;
        for train in &training {
            for held in &held_corpora {
                train.disjoint(held)?;
            }
        }
        let train = Corpus::combine(&mut training)?;
        if unroll > train.length || train.frames.iter().any(|(f, _)| f.low != low) {
            return Err("unroll exceeds sequence or extents differ".into());
        }
        let network = graph::build(config, low, unroll)?;
        println!(
            "{} parameters; {} training / {} development frames",
            network.params.iter().map(|p| p.len).sum::<usize>(),
            train.frames.len(),
            holdout.frames.len()
        );
        let mut session = ommatidia::gpu::training_session(&network.graph, Arc::clone(&context));
        network.initialize(&mut session, seed);
        if let Some(path) = &checkpoint_input {
            // A training-session load also restores Adam moments and its step.
            // Use inference to validate the asset, then copy parameters only.
            learned.session.load_checkpoint(path)?;
            let names: Vec<_> = network.params.iter().map(|p| p.name.as_str()).collect();
            for (name, values) in names.iter().zip(learned.session.read_params(&names)) {
                session.set_parameter(name, &values);
            }
        }
        std::fs::write(
            out.join("model.transport.ron"),
            ron::ser::to_string_pretty(&config, ron::ser::PrettyConfig::default())?,
        )?;
        std::fs::write(
            out.join("training.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "steps":steps, "unroll":unroll, "seed":seed, "learning_rate":rate,
                "loss_weights":weights, "warm_start":checkpoint_input, "optimizer_resumed":false,
                "preparation":"native GPU features/guide/history; CPU differentiable gather maps",
                "training":train.provenance, "development":holdout.provenance,
                "parameters":network.params.iter().map(|p|p.len).sum::<usize>()
            }))?,
        )?;
        let mut rng = ommatidia::rng::Rng::new(seed);
        let mut losses = std::fs::File::create(out.join("loss.csv"))?;
        writeln!(losses, "update,loss")?;
        let started = std::time::Instant::now();
        for update in 0..steps {
            session.wait();
            learned.sync_parameters(&session);
            learned.reset();
            let sequence = rng.below((train.frames.len() / train.length) as u32) as usize;
            let start = rng.below((train.length - unroll + 1) as u32) as usize;
            let offset = sequence * train.length;
            for (frame, _) in &train.frames[offset..offset + start] {
                learned.advance(frame)?;
            }
            for slot in 0..unroll {
                let (frame, target) = &train.frames[offset + start + slot];
                let old = if start + slot == 0 {
                    Vec::new()
                } else {
                    learned.read_state()
                };
                learned.advance(frame)?;
                let prepared = learned.read_prepared(frame, &old);
                graph::feed(&mut session, &format!("f{slot}"), &prepared, target, slot);
                graph::feed_rgb(&mut session, &format!("f{slot}"), frame, target, config);
            }
            weights.feed(&mut session);
            let fraction = update as f32 / steps as f32;
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
            writeln!(losses, "{},{loss}", update + 1)?;
            if update % 16 == 0 {
                println!(
                    "update {}/{steps}: loss {loss:.7}, {:.2} updates/s",
                    update + 1,
                    (update + 1) as f32 / started.elapsed().as_secs_f32()
                );
            }
            if eval_every > 0 && (update + 1) % eval_every == 0 && update + 1 < steps {
                let path = out.join(format!("step-{}.safetensors", update + 1));
                session.save_checkpoint(&path)?;
                learned.session.load_checkpoint(&path)?;
                let dir = out.join(format!("dev-{}", update + 1));
                std::fs::create_dir_all(&dir)?;
                let report = evaluate(
                    &holdout,
                    config,
                    &mut learned,
                    &mut baseline,
                    &dir,
                    evaluation,
                )?;
                std::fs::write(
                    dir.join("quality.json"),
                    serde_json::to_vec_pretty(&report)?,
                )?;
                println!(
                    "development {}: {}",
                    update + 1,
                    serde_json::to_string(&report)?
                );
            }
        }
        session.save_checkpoint(&checkpoint)?;
        // Score serialized weights, not the still-live training session.
        learned.session.load_checkpoint(&checkpoint)?;
    }
    let mut report = evaluate(
        &holdout,
        config,
        &mut learned,
        &mut baseline,
        &out,
        evaluation,
    )?;
    report["role"] = serde_json::json!(if eval_only {
        "evaluation"
    } else {
        "scene-disjoint development"
    });
    report["capture"] = holdout.provenance;
    report["checkpoint"] =
        serde_json::json!(checkpoint_input.filter(|_| eval_only).unwrap_or(checkpoint));
    std::fs::write(
        out.join("quality.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lobe_diagnostics_preserve_subpixel_packing_and_observed_materials() {
        let config = Config::default();
        let low = [4, 8];
        let n = (low[0] * low[1] * config.scale.pow(2)) as usize;
        let frame = Frame {
            low,
            jitter: [0.0; 2],
            rays: Vec::new(),
            surfaces: vec![
                ommatidia::transport::Surface {
                    albedo_roughness: [0.2, 0.4, 0.8, 0.5],
                    emission: [0.01, 0.02, 0.03, 0.0],
                    ..Default::default()
                };
                n
            ],
        };
        let mut target = Target {
            lobes: vec![0.0; 6 * n],
            rgb: Vec::new(),
        };
        for i in 0..n {
            for c in 0..6 {
                target.lobes[config.index(low, c, i % 8, i / 8)] = c as f32 + i as f32 * 0.01;
            }
        }
        let lobes = reference_lobes(&frame, &target, config);
        let composition = compose_lobes(&frame, &lobes);
        for i in 0..n {
            for c in 0..3 {
                assert_eq!(lobes[0][i * 3 + c], c as f32 + i as f32 * 0.01);
                assert_eq!(lobes[1][i * 3 + c], (c + 3) as f32 + i as f32 * 0.01);
                let s = frame.surfaces[i];
                assert_eq!(
                    composition[i * 3 + c],
                    lobes[0][i * 3 + c] * s.albedo_roughness[c]
                        + lobes[1][i * 3 + c]
                        + s.emission[c]
                );
            }
        }
    }

    #[test]
    fn capture_quality_rejects_measured_invisible_assets() {
        let mut p =
            serde_json::json!({"matching_path_depth":true,"input_estimator":"independent-paths"});
        assert!(validate_capture(&p).is_ok());
        for coverage in [
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!(1.1),
            serde_json::json!("unknown"),
        ] {
            p["minimum_catalog_visible_fraction"] = coverage;
            assert!(validate_capture(&p).is_err());
        }
        p["minimum_catalog_visible_fraction"] = serde_json::json!(0.05);
        assert!(validate_capture(&p).is_ok());
    }

    #[test]
    fn training_and_development_must_not_share_scenes_or_families() {
        let corpus = |seed, family| Corpus {
            frames: Vec::new(),
            length: 8,
            provenance: serde_json::json!({"scene_seeds":[seed],"family_ids":[family]}),
        };
        let training = corpus(7, "chair-a");
        assert!(training.disjoint(&corpus(7, "chair-b")).is_err());
        assert!(training.disjoint(&corpus(8, "chair-a")).is_err());
        assert!(training.disjoint(&corpus(8, "chair-b")).is_ok());
    }

    #[test]
    fn rejected_history_preserves_lobe_and_subpixel_layout() {
        let config = Config::default();
        let low = [4, 8];
        let mut validity = vec![1.0; 2 * 8 * 16];
        assert!(!rejected_history_mask(&validity, low, config).contains(&true));
        validity[config.index(low, 0, 3, 0)] = 0.0;
        validity[config.index(low, 1, 0, 15)] = 0.0;
        let mask = rejected_history_mask(&validity, low, config);
        assert_eq!(mask.iter().filter(|&&v| v).count(), 2);
        assert!(mask[3] && mask[120]);
    }

    #[test]
    fn rejected_history_is_pixel_weighted_and_empty_is_unscored() {
        let reference = [1.0; 9];
        let mut score = Score::default();
        score.add_rejected_history(&[0.0; 9], &reference, &[false; 3]);
        assert_eq!(score.rejected_history_mse, None);
        score.add_rejected_history(&[0.0; 9], &reference, &[true, false, false]);
        score.add_rejected_history(&reference, &reference, &[true; 3]);
        score.finish();
        assert_eq!(score.rejected_history_pixels, 4);
        assert_eq!(score.history_pixels, 9);
        assert_eq!(score.rejected_history_mse, Some(0.25 / 4.0));
        assert_eq!(score.rejected_history_fraction, Some(4.0 / 9.0));

        let mut empty = Score::default();
        empty.finish();
        assert_eq!(empty.rejected_history_mse, None);
        assert_eq!(empty.rejected_history_fraction, None);
    }
}
