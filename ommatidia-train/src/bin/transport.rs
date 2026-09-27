//! Train and evaluate the lobe-separated recurrent reconstructor on full sequences.
use ommatidia::{
    dataset::{Layout, Sample},
    metrics,
    transport::{Config, Frame, Target, graph, native},
};
use ommatidia_train::{
    evaluation::{self, ControlRun},
    open_capture,
    profile::{Stage, TrainingProfile, Update},
    save_linear, save_png,
};
use serde::Serialize;
use std::{
    io::Write,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
type EvaluationHistory = (Vec<Vec<f32>>, Vec<f32>, Vec<ommatidia::temporal::Surface>);
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const DEFAULT_RESET_INTERVAL: usize = 2;

fn warmup_length(update: usize, start: usize, reset_every: NonZeroUsize) -> usize {
    if update.is_multiple_of(reset_every.get()) {
        0
    } else {
        start
    }
}

struct Record {
    layout: Layout,
    sample: Sample,
}
impl Record {
    fn low(&self) -> [u32; 2] {
        [self.layout.lr_width, self.layout.lr_height]
    }
    fn frame(&self, config: Config) -> Result<Frame> {
        Ok(Frame::from_sample(&self.sample, self.layout, config)?)
    }
    fn decode(&self, config: Config) -> Result<(Frame, Target)> {
        Ok((
            self.frame(config)?,
            Target::from_sample(&self.sample, self.layout, config)?,
        ))
    }
}

struct Corpus {
    frames: Vec<Record>,
    length: usize,
    provenance: serde_json::Value,
}
impl Corpus {
    fn load(path: &Path, config: Config) -> Result<Self> {
        let (mut reader, provenance) = open_capture(path)?;
        let layout = *reader.layout();
        let mut frames = Vec::new();
        for index in 0..reader.len() {
            let sample = reader.sample(index)?;
            let record = Record { layout, sample };
            let (_, target) = record.decode(config)?;
            if target
                .rgb
                .iter()
                .chain(&target.lobes)
                .any(|v| !v.is_finite() || *v < 0.0)
                || target.rgb.iter().all(|v| *v <= 1e-6)
            {
                return Err(format!("{} record {index}: invalid or entirely black reference; inspect the capture camera", path.display()).into());
            }
            // Retain original f16 records; expand only the frames being consumed.
            frames.push(record);
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
            .any(|c| c.length != first.length || c.frames[0].low() != first.frames[0].low())
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
#[derive(Clone, Default, Serialize)]
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
    fn accumulate(&mut self, other: &Self) {
        self.frames += other.frames;
        self.psnr += other.psnr;
        self.ssim += other.ssim;
        self.low_frequency_psnr += other.low_frequency_psnr;
        self.relative_mse += other.relative_mse;
        self.linear_mse += other.linear_mse;
        self.energy_ratio += other.energy_ratio;
        self.detail_ratio += other.detail_ratio;
        self.gradient_mse += other.gradient_mse;
        self.temporal_mse += other.temporal_mse;
        self.temporal_frames += other.temporal_frames;
        self.reset_psnr += other.reset_psnr;
        self.resets += other.resets;
        if let Some(value) = other.rejected_history_mse {
            *self.rejected_history_mse.get_or_insert(0.0) += value;
        }
        self.rejected_history_pixels += other.rejected_history_pixels;
        self.history_pixels += other.history_pixels;
    }

    fn report(&self) -> serde_json::Value {
        if self.frames == 0 {
            return serde_json::Value::Null;
        }
        let mut score = self.clone();
        score.finish();
        let mut report = serde_json::to_value(score).unwrap();
        if self.temporal_frames == 0 {
            report["temporal_mse"] = serde_json::Value::Null;
        }
        if self.resets == 0 {
            report["reset_psnr"] = serde_json::Value::Null;
        }
        report
    }

    fn cells(&self) -> [Option<f64>; 9] {
        [
            Some(self.psnr),
            Some(self.ssim),
            Some(self.low_frequency_psnr),
            Some(self.relative_mse),
            Some(self.linear_mse),
            Some(self.energy_ratio),
            Some(self.detail_ratio),
            Some(self.gradient_mse),
            (self.temporal_frames > 0).then_some(self.temporal_mse),
        ]
    }

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
    assert_eq!(validity.len(), n);
    assert!(
        validity.iter().all(|&v| v == 0.0 || v == 1.0),
        "invalid reprojection mask"
    );
    (0..n)
        .map(|i| validity[config.index(low, 0, i % width, i / width)] == 0.0)
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

#[derive(Clone, Default)]
struct EvaluationOptions {
    reset_every: Option<NonZeroUsize>,
    save_lobes: bool,
    save_linear: bool,
    no_images: bool,
    control_run: Option<PathBuf>,
}
impl EvaluationOptions {
    fn training() -> Self {
        Self {
            no_images: true,
            ..Self::default()
        }
    }
}

fn evaluate(
    corpus: &Corpus,
    config: Config,
    learned: &mut native::Native,
    out: &Path,
    options: &EvaluationOptions,
) -> Result<serde_json::Value> {
    let mut report = serde_json::json!({
        "schema":2, "capture":corpus.provenance,
        "extent":corpus.frames[0].low().map(|v| v * config.scale),
        "sequence_length":corpus.length,"frames":corpus.frames.len(),
        "reset_every":options.reset_every,"save_linear":options.save_linear,
        "control_run":options.control_run,
        "history_mode":if options.reset_every.is_some() { "periodic-cuts" } else { "causal" },
        "metric_space":"PSNR/SSIM/gradient/lobe MSE: x/(1+x); energy: scene-linear; PNG: same compression then sRGB",
        "temporal_scope":"motion-compensated change residual; pairs crossing resets are excluded; undefined metrics are null",
        "rejected_history_space":"pixel-weighted compressed RGB MSE on non-reset pixels with no in-frame geometric warp tap; excludes learned alpha suppression; empty regions are null",
        "bucket_definitions":{"cold":"0","early":"1-7","settling":"8-15","warm":">=16"},
        "speed_claim":false,
    });
    let control = options
        .control_run
        .as_ref()
        .map(|dir| ControlRun::open(dir, &report))
        .transpose()?;
    let roles: &[&str] = if control.is_some() {
        &["learned", "control"]
    } else {
        &["learned"]
    };
    let mut scores = vec![Score::default(); roles.len()];
    let mut buckets = std::collections::BTreeMap::from(
        ["cold", "early", "settling", "warm"].map(|key| (key, vec![Score::default(); roles.len()])),
    );
    let mut previous: Option<EvaluationHistory> = None;
    let mut diagnostics = Vec::new();
    let mut rows = std::fs::File::create(out.join("frames.csv"))?;
    const METRICS: [&str; 9] = [
        "psnr",
        "ssim",
        "low_frequency_psnr",
        "relative_mse",
        "linear_mse",
        "energy_ratio",
        "detail_ratio",
        "gradient_mse",
        "temporal_mse",
    ];
    write!(rows, "sequence,frame,frames_since_reset")?;
    for role in roles
        .iter()
        .copied()
        .chain(control.as_ref().map(|_| "delta"))
    {
        for metric in METRICS {
            write!(rows, ",{role}_{metric}")?;
        }
    }
    writeln!(rows)?;
    for (index, record) in corpus.frames.iter().enumerate() {
        let (frame, target) = record.decode(config)?;
        let (frame, target) = (&frame, &target);
        let since_reset = evaluation::age(index % corpus.length, options.reset_every);
        let reset = since_reset == 0;
        if reset {
            previous = None;
        }
        if reset {
            learned.reset();
        }
        let prefix = format!("{:03}-{:03}", index / corpus.length, index % corpus.length);
        let mut images = vec![learned.process(frame)?];
        if let Some(control) = &control {
            images.push(control.load(&prefix, &target.rgb)?);
        }
        let mut frame_scores = vec![Score::default(); roles.len()];
        let extent = frame.low.map(|v| v * config.scale);
        let current = surfaces(frame);
        let motion: Vec<_> = frame
            .surfaces
            .iter()
            .flat_map(|s| s.motion[..2].iter().copied())
            .collect();
        for (score, image) in frame_scores.iter_mut().zip(&images) {
            if image.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err("non-finite or negative reconstruction".into());
            }
            score.add(image, &target.rgb, extent, reset);
        }
        if !reset {
            let mask = rejected_history_mask(&learned.read_history_validity(), frame.low, config);
            frame_scores[0].add_rejected_history(&images[0], &target.rgb, &mask);
        }
        if let Some((old, reference, old_surfaces)) = &previous {
            let warp = ommatidia::temporal::Reprojection {
                motion: &motion,
                current: &current,
                previous: old_surfaces,
                rejection: Default::default(),
            };
            for k in 0..images.len() {
                if let Some(e) = metrics::temporal_error(
                    [&images[k], &old[k]],
                    [&target.rgb, reference],
                    warp,
                    None,
                    frame.low.map(|v| v as usize),
                    config.scale as usize,
                ) {
                    frame_scores[k].temporal_mse = e.mean();
                    frame_scores[k].temporal_frames = 1;
                }
            }
        }
        write!(
            rows,
            "{},{},{since_reset}",
            index / corpus.length,
            index % corpus.length
        )?;
        let mut cells: Vec<_> = frame_scores.iter().map(Score::cells).collect();
        if control.is_some() {
            cells.push(std::array::from_fn(|i| {
                cells[0][i].zip(cells[1][i]).map(|(new, old)| new - old)
            }));
        }
        for value in cells.into_iter().flatten() {
            write!(
                rows,
                ",{}",
                value.map(|v| v.to_string()).unwrap_or_default()
            )?;
        }
        writeln!(rows)?;
        for (k, score) in frame_scores.iter().enumerate() {
            scores[k].accumulate(score);
            buckets.get_mut(evaluation::bucket(since_reset)).unwrap()[k].accumulate(score);
        }
        // All frames are retained: evaluation is not a cherry-picked screenshot.
        let truth = reference_lobes(frame, target, config);
        let composition = compose_lobes(frame, &truth);
        let mut diagnostic = serde_json::json!({
            "sequence": index / corpus.length,
            "frame": index % corpus.length,
            "reference_composition_mse": metrics::error(&composition, &target.rgb),
        });
        for (name, model) in [("learned", &*learned)] {
            let state = model.read_state();
            let width = extent[0] as usize;
            let lobes: [Vec<f32>; 2] = std::array::from_fn(|lobe| {
                (0..frame.surfaces.len())
                    .flat_map(|i| {
                        let values = &state.values;
                        (0..3).map(move |c| {
                            values[config.index(frame.low, 3 * lobe + c, i % width, i / width)]
                        })
                    })
                    .collect()
            });
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
        for (name, image) in [("learned", &images[0]), ("reference", &target.rgb)] {
            if !options.no_images {
                save_png(&out.join(format!("{prefix}-{name}.png")), image, extent)?;
            }
            if options.save_linear {
                save_linear(&out.join(format!("{prefix}-{name}.rgbf32")), image)?;
            }
        }
        previous = Some((images, target.rgb.clone(), current));
    }
    std::fs::write(
        out.join("diagnostics.json"),
        serde_json::to_vec_pretty(&diagnostics)?,
    )?;
    for (role, score) in roles.iter().zip(&scores) {
        report[*role] = score.report();
    }
    report["buckets"] = serde_json::Value::Object(
        buckets
            .into_iter()
            .map(|(bucket, scores)| {
                (
                    bucket.to_owned(),
                    serde_json::Value::Object(
                        roles
                            .iter()
                            .zip(scores)
                            .map(|(role, score)| ((*role).to_owned(), score.report()))
                            .collect(),
                    ),
                )
            })
            .collect(),
    );
    Ok(report)
}
fn main() -> Result<()> {
    env_logger::init();
    let mut data = Vec::new();
    let mut eval = Vec::new();
    let mut out = PathBuf::from("runs/transport");
    let mut steps = 4000usize;
    let mut unroll = 4usize;
    let mut channels = 16;
    let mut seed = 7u64;
    let mut rate = 0.0003f32;
    let mut eval_only = false;
    let mut profile_only = false;
    let mut checkpoint_input = None::<PathBuf>;
    let mut eval_every = 10_000usize;
    let mut reset_every = NonZeroUsize::new(DEFAULT_RESET_INTERVAL).unwrap();
    let mut device_id = None;
    let mut weights = graph::LossWeights::default();
    let mut evaluation = EvaluationOptions::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!(
                "transport --data TRAIN.omd --eval-data DEV.omd --out DIR
  --steps N [4000] --unroll N [4] --channels N [16] --seed N [7]
  --lr F [0.0003] --eval-every N [10000] (causal metrics, no images) --device-id ID
  --train-reset-every N [2] (training: empty history each Nth window)
  --reset-every N [off] (evaluation: simulated cut each N frames, within each sequence)
  --checkpoint FILE (evaluation input; optimizer-resume belongs to Phase 3)
  --eval-only (loads checkpoint sidecar; evaluation resolution may differ)
  --profile-only (training-loop measurement; save checkpoint/timings, skip evaluation)
  --reset-history (eval-only diagnostic: reset the model before every frame)
  --save-lobes (save diffuse/specular images alongside per-frame diagnostics)
  --save-linear (save row-major, little-endian scene-linear RGB f32 for crop scoring)
  --no-images (skip PNG writing; compatible with --save-linear)
  --control-run DIR (compare saved control outputs; identical ordered data and reset protocol)
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
        if arg == "--profile-only" {
            profile_only = true;
            continue;
        }
        if arg == "--reset-history" {
            if evaluation
                .reset_every
                .replace(NonZeroUsize::new(1).unwrap())
                .is_some()
            {
                return Err("specify only one evaluation reset option".into());
            }
            continue;
        }
        if arg == "--no-images" {
            evaluation.no_images = true;
            continue;
        }
        if arg == "--save-lobes" {
            evaluation.save_lobes = true;
            continue;
        }
        if arg == "--save-linear" {
            evaluation.save_linear = true;
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
            "--train-reset-every" => reset_every = v.parse()?,
            "--reset-every" => {
                if evaluation.reset_every.replace(v.parse()?).is_some() {
                    return Err("specify only one evaluation reset option".into());
                }
            }
            "--control-run" => evaluation.control_run = Some(PathBuf::from(v).canonicalize()?),
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
    if checkpoint_input.is_some() && !eval_only {
        return Err(
            "training starts from scratch; optimizer/cursor resume is not available until Phase 3"
                .into(),
        );
    }
    if profile_only && eval_only {
        return Err("--profile-only and --eval-only are mutually exclusive".into());
    }
    if profile_only {
        eval_every = 0;
    }
    if !eval_only && (evaluation.reset_every.is_some() || evaluation.control_run.is_some()) {
        return Err("--reset-every/--reset-history/--control-run require --eval-only; training checks are causal".into());
    }
    if evaluation.no_images && evaluation.save_lobes {
        return Err("--no-images and --save-lobes are mutually exclusive".into());
    }
    if !eval_only && (evaluation.save_linear || evaluation.save_lobes) {
        return Err(
            "--save-linear/--save-lobes require --eval-only; training checks save metrics only"
                .into(),
        );
    }
    if out.join("frames.csv").exists() || out.join("quality.json").exists() {
        return Err("refusing to overwrite existing evaluation outputs".into());
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
        Config::parse(&std::fs::read_to_string(
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
    let low = holdout.frames[0].low();
    let inference_macs = graph::build(config, low, 0)?.macs();
    println!(
        "{inference_macs} convolution MACs/frame; {:.6} GFLOP/frame (forward convolutions only)",
        inference_macs as f64 * 2.0e-9
    );
    let context = ommatidia::gpu::create_context(device_id, false);
    let mut learned = native::Native::new(Arc::clone(&context), config, low)?;
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
        if unroll > train.length || train.frames.iter().any(|r| r.low() != low) {
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
        // Training targets only, never development/audit images, set the radiance prior.
        let mut means = [0.0_f64; 6];
        let mut mean_pixels = 0usize;
        for record in &train.frames {
            let (frame, target) = record.decode(config)?;
            let n = frame.surfaces.len();
            mean_pixels += n;
            for (c, sum) in means.iter_mut().enumerate() {
                *sum += target.lobes[c * n..(c + 1) * n]
                    .iter()
                    .map(|v| f64::from(*v) * f64::from(frame.exposure))
                    .sum::<f64>();
            }
        }
        for v in &mut means {
            *v /= mean_pixels as f64;
        }
        let bias: Vec<_> = means
            .iter()
            .flat_map(|v| {
                std::iter::repeat_n(
                    v.ln().clamp(-16.0, 11.0) as f32,
                    config.scale.pow(2) as usize,
                )
            })
            .collect();
        session.set_parameter("head.radiance.bias", &bias);
        std::fs::write(
            out.join("model.transport.ron"),
            ron::ser::to_string_pretty(&config, ron::ser::PrettyConfig::default())?,
        )?;
        std::fs::write(
            out.join("training.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "steps":steps, "unroll":unroll, "seed":seed, "learning_rate":rate,
                "inference_macs":inference_macs, "inference_flops":2 * inference_macs,
                "macs_scope":"dense forward convolutions per frame, padded taps included; excludes activations, warp, preparation and backward",
                "profile_only":profile_only,
                "evaluation":{"every":eval_every,"history_mode":"causal","pngs":false},
                "loss_weights":weights, "optimizer_resumed":false, "corpus_mean_normalized_lobes":means,
                "reload_check":"bitwise equality of every trained parameter before final evaluation",
                "preparation":"native GPU observation packing and geometric warp maps; graph radiance/latent recurrence",
                "cpu_corpus":"original f16 capture records; per-frame f32 expansion with no requantization",
                "window_sampling":{
                    "sequence":"uniform", "start":"uniform among complete unrolls",
                    "reset_every_updates":reset_every.get(), "reset_phase_zero_based":0,
                    "otherwise":"warm from sequence start using current weights"
                },
                "training":train.provenance, "development":holdout.provenance,
                "parameters":network.params.iter().map(|p|p.len).sum::<usize>()
            }))?,
        )?;
        let mut rng = ommatidia::rng::Rng::new(seed);
        let mut losses = std::fs::File::create(out.join("loss.csv"))?;
        writeln!(losses, "update,loss,sequence,start,warmup_frames")?;
        let mut profile_rows = std::fs::File::create(out.join("profile.csv"))?;
        Update::write_header(&mut profile_rows)?;
        let mut profile = TrainingProfile::default();
        let mut steady_profile = TrainingProfile::default();
        let started = std::time::Instant::now();
        for update in 0..steps {
            let update_start = Instant::now();
            let mut timing = Update::default();
            let stage = Instant::now();
            session.wait();
            learned.sync_parameters(&session);
            learned.reset();
            timing.record(Stage::ParameterSync, stage);
            let sequence = rng.below((train.frames.len() / train.length) as u32) as usize;
            let start = rng.below((train.length - unroll + 1) as u32) as usize;
            let offset = sequence * train.length;
            // Simulate cuts at varied times, retaining full causal warmup otherwise.
            let warmup = warmup_length(update, start, reset_every);
            for record in &train.frames[offset..offset + warmup] {
                let stage = Instant::now();
                let frame = record.frame(config)?;
                timing.record(Stage::FrameDecode, stage);
                let stage = Instant::now();
                learned.advance(&frame)?;
                timing.record(Stage::WarmupAdvance, stage);
            }
            for slot in 0..unroll {
                let stage = Instant::now();
                let (frame, target) = train.frames[offset + start + slot].decode(config)?;
                timing.record(Stage::FrameDecode, stage);
                let (frame, target) = (&frame, &target);
                let stage = Instant::now();
                learned.advance(frame)?;
                timing.record(Stage::SlotAdvance, stage);
                let stage = Instant::now();
                let prepared = learned.read_prepared(frame);
                timing.record(Stage::ReadPrepared, stage);
                let stage = Instant::now();
                graph::feed(&mut session, &format!("f{slot}"), &prepared, target, slot);
                graph::feed_rgb(&mut session, &format!("f{slot}"), frame, target, config);
                timing.record(Stage::Feed, stage);
            }
            let stage = Instant::now();
            weights.feed(&mut session);
            let fraction = update as f32 / steps as f32;
            session.set_adam(
                rate * (0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * fraction).cos())),
                0.9,
                0.999,
                1e-8,
            );
            timing.record(Stage::Feed, stage);
            let stage = Instant::now();
            session.step();
            session.wait();
            timing.record(Stage::StepWait, stage);
            let stage = Instant::now();
            let loss = session.read_loss();
            timing.record(Stage::LossReadback, stage);
            if !loss.is_finite() {
                return Err("training became non-finite".into());
            }
            writeln!(losses, "{},{loss},{sequence},{start},{warmup}", update + 1)?;
            if update % 16 == 0 {
                println!(
                    "update {}/{steps}: loss {loss:.7}, {:.2} updates/s",
                    update + 1,
                    (update + 1) as f32 / started.elapsed().as_secs_f32()
                );
            }
            timing.finish(update_start);
            timing.write(&mut profile_rows, update + 1)?;
            profile.add(&timing);
            if update >= 10 {
                steady_profile.add(&timing);
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
                    &dir,
                    &EvaluationOptions::training(),
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
        let profile_report = serde_json::json!({
            "scope":"non-overlapping host wall time; excludes corpus loading, session setup, checkpoints, evaluation and profile CSV writes; advance and step include their GPU waits",
            "read_prepared_scope":"prepared-input and GPU-generated warp-map readback",
            "device":context.device_information().device_name,
            "low_extent":low, "unroll":unroll,
            "training_frames":train.frames.len(), "training_sequences":train.frames.len() / train.length,
            "all_updates":profile.report(), "after_first_10_updates":steady_profile.report(),
        });
        std::fs::write(
            out.join("profile.json"),
            serde_json::to_vec_pretty(&profile_report)?,
        )?;
        println!(
            "training profile: {}",
            serde_json::to_string(&profile_report)?
        );
        session.save_checkpoint(&checkpoint)?;
        // Verify the actual final training state, not two loads of the same file.
        learned.session.load_checkpoint(&checkpoint)?;
        let names: Vec<_> = network.params.iter().map(|p| p.name.as_str()).collect();
        for ((name, live), loaded) in names
            .iter()
            .zip(session.read_params(&names))
            .zip(learned.session.read_params(&names))
        {
            if live.len() != loaded.len()
                || live
                    .iter()
                    .zip(&loaded)
                    .any(|(a, b)| !a.is_finite() || a.to_bits() != b.to_bits())
            {
                return Err(format!("final checkpoint did not reload {name} bit-exactly").into());
            }
        }
        println!("final checkpoint: all parameters finite and bit-exact after reload");
        if profile_only {
            return Ok(());
        }
        // Score serialized weights, not the still-live training session.
    }
    let mut report = evaluate(
        &holdout,
        config,
        &mut learned,
        &out,
        &if eval_only {
            evaluation
        } else {
            EvaluationOptions::training()
        },
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
    fn training_evaluation_is_causal_and_does_not_save_images() {
        let options = EvaluationOptions::training();
        assert!(options.no_images);
        assert!(options.reset_every.is_none() && options.control_run.is_none());
        assert!(!options.save_lobes && !options.save_linear);
    }

    #[test]
    fn frame_scores_aggregate_once_and_undefined_metrics_stay_null() {
        assert_eq!(Score::default().report(), serde_json::Value::Null);
        let mut cold = Score::default();
        cold.add(&[0.5; 3 * 16 * 16], &[1.0; 3 * 16 * 16], [16, 16], true);
        assert!(cold.cells()[8].is_none());
        assert!(cold.report()["temporal_mse"].is_null());
        let mut early = Score::default();
        early.add(&[1.0; 3 * 16 * 16], &[1.0; 3 * 16 * 16], [16, 16], false);
        early.temporal_mse = 0.25;
        early.temporal_frames = 1;
        assert!(early.report()["reset_psnr"].is_null());
        let mut total = Score::default();
        total.accumulate(&cold);
        total.accumulate(&early);
        let report = total.report();
        assert_eq!(report["frames"], 2);
        assert_eq!(report["psnr"], (cold.psnr + early.psnr) / 2.0);
        assert_eq!(report["temporal_mse"], 0.25);
        assert_eq!(report["energy_ratio"], 0.75);
        assert_eq!(report["reset_psnr"], cold.psnr);
    }

    #[test]
    fn training_cuts_preserve_random_frame_coverage_and_full_warmup() {
        let mut rng = ommatidia::rng::Rng::new(31);
        let mut cut_scenes = [0; 40];
        let mut cut_starts = [false; 63];
        let mut forced_cuts = 0;
        let reset_every = NonZeroUsize::new(DEFAULT_RESET_INTERVAL).unwrap();
        for update in 0..4000 {
            let sequence = rng.below(cut_scenes.len() as u32) as usize;
            let start = rng.below(63) as usize;
            let warmup = warmup_length(update, start, reset_every);
            if update % 2 == 0 {
                assert_eq!(warmup, 0);
                cut_scenes[sequence] += 1;
                cut_starts[start] = true;
                forced_cuts += 1;
            } else {
                assert_eq!(warmup, start);
            }
            assert_eq!(warmup_length(update, 0, reset_every), 0);
        }
        assert_eq!(forced_cuts, 2000);
        assert!(cut_scenes.iter().all(|&count| count > 0));
        assert!(cut_starts.iter().all(|&seen| seen));
    }

    #[test]
    fn explicit_reset_interval_controls_warmup_without_changing_start() {
        assert!("0".parse::<NonZeroUsize>().is_err());
        for interval in [1, 2, 4] {
            let reset_every = NonZeroUsize::new(interval).unwrap();
            for update in 0..16 {
                for start in 0..64 {
                    assert_eq!(
                        warmup_length(update, start, reset_every),
                        if update % interval == 0 { 0 } else { start },
                    );
                }
            }
        }
    }

    #[test]
    fn lobe_diagnostics_preserve_subpixel_packing_and_observed_materials() {
        let config = Config::default();
        let low = [4, 8];
        let n = (low[0] * low[1] * config.scale.pow(2)) as usize;
        let frame = Frame {
            low,
            jitter: [0.0; 2],
            exposure: 1.0,
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
    fn geometric_validity_preserves_subpixel_layout() {
        let config = Config::default();
        let low = [4, 8];
        let mut validity = vec![1.0; 8 * 16];
        assert!(!rejected_history_mask(&validity, low, config).contains(&true));
        validity[config.index(low, 0, 3, 0)] = 0.0;
        validity[config.index(low, 0, 0, 15)] = 0.0;
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
