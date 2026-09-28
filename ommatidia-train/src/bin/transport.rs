//! Train and evaluate the lobe-separated recurrent reconstructor on full sequences.
use ommatidia::{
    metrics,
    transport::{Config, Frame, Target, graph, native},
};
use ommatidia_train::{
    checkpoint::{Checkpoint, Settings},
    corpus::Corpus,
    evaluation::{self, ControlRun},
    sampler::{Batch, Prefetch, Sampler, learning_rate},
    save_linear, save_png,
    training::Trainer,
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
    alpha_frames: Vec<[usize; 2]>,
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
    if options
        .alpha_frames
        .iter()
        .any(|&[sequence, frame]| sequence >= corpus.sequences.len() || frame >= corpus.length)
    {
        return Err("alpha diagnostic frame outside evaluation corpus".into());
    }
    let mut report = serde_json::json!({
        "schema":2, "capture":corpus.provenance,
        "extent":corpus.low.map(|v| v * config.scale),
        "sequence_length":corpus.length,"frames":corpus.len(),
        "reset_every":options.reset_every,"save_linear":options.save_linear,
        "control_run":options.control_run,
        "alpha_frames":options.alpha_frames,
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
    for index in 0..corpus.len() {
        let (frame, target) = corpus.decode(index, config)?;
        if target.rgb.iter().all(|v| *v <= 1e-6) {
            return Err("entirely black evaluation reference".into());
        }
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
        if options
            .alpha_frames
            .contains(&[index / corpus.length, index % corpus.length])
        {
            let packed = learned
                .read_alpha()
                .ok_or("alpha diagnostic output is unavailable")?;
            let width = extent[0] as usize;
            for (lobe, name) in ["diffuse", "specular"].into_iter().enumerate() {
                let alpha: Vec<_> = (0..frame.surfaces.len())
                    .map(|i| packed[config.index(frame.low, lobe, i % width, i / width)])
                    .collect();
                let summary = alpha_summary(&alpha)?;
                let raw = format!("{prefix}-alpha-{name}.f32");
                save_linear(&out.join(&raw), &alpha)?;
                if !options.no_images {
                    // Alpha is a probability, not radiance: encode direct linear
                    // grayscale, without the radiance display transform.
                    let bytes: Vec<_> = alpha.iter().map(|v| (v * 255.0).round() as u8).collect();
                    let mut encoder = png::Encoder::new(
                        std::fs::File::create(out.join(format!("{prefix}-alpha-{name}.png")))?,
                        extent[0],
                        extent[1],
                    );
                    encoder.set_color(png::ColorType::Grayscale);
                    encoder.set_depth(png::BitDepth::Eight);
                    encoder.write_header()?.write_image_data(&bytes)?;
                }
                diagnostic[format!("alpha_{name}")] = serde_json::json!({
                    "summary":summary,"raw":raw,"layout":"row-major little-endian f32 probability",
                    "histogram_bins":"20 equal-width bins on [0,1]; right edge 1 belongs to the final bin",
                });
            }
        }
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
fn alpha_summary(values: &[f32]) -> Result<serde_json::Value> {
    if values.is_empty()
        || values
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    {
        return Err("alpha must contain finite probabilities in [0,1]".into());
    }
    let mut histogram = [0_usize; 20];
    for &value in values {
        histogram[((value * 20.0) as usize).min(19)] += 1;
    }
    let mut summary = finite_summary(values.iter().copied());
    summary["histogram"] = serde_json::json!(histogram);
    Ok(summary)
}

fn finite_summary(values: impl Iterator<Item = f32>) -> serde_json::Value {
    let mut count = 0;
    let mut nonfinite = 0;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;
    for value in values {
        if value.is_finite() {
            count += 1;
            min = min.min(value);
            max = max.max(value);
            sum += value as f64;
        } else {
            nonfinite += 1;
        }
    }
    serde_json::json!({
        "finite":count,"nonfinite":nonfinite,
        "min":(count>0).then_some(min),"max":(count>0).then_some(max),
        "mean":(count>0).then_some(sum / count.max(1) as f64),
    })
}

fn loss_diagnostics(trainer: &mut Trainer, batch: &Batch, config: Config) -> serde_json::Value {
    let states = trainer.read_states();
    let losses = trainer.last_microbatch_losses();
    let n = batch.frames[0][0].0.surfaces.len();
    let state_len = n * config.state_channels();
    let cursors: Vec<_> = batch.windows.iter().enumerate().map(|(index, window)| {
        let state = &states[index * state_len..(index + 1) * state_len];
        let frames = &batch.frames[index];
        serde_json::json!({
            "cursor":index,"window":window,"loss":losses[index],
            "input_lobes":finite_summary(frames.iter().flat_map(|(f, _)| f.rays.iter())
                .flat_map(|r| r.diffuse[..3].iter().chain(&r.specular[..3])).copied()),
            "target_lobes":finite_summary(frames.iter().flat_map(|(_, t)| t.lobes.iter()).copied()),
            "target_rgb":finite_summary(frames.iter().flat_map(|(_, t)| t.rgb.iter()).copied()),
            "last_frame_predicted_lobes":finite_summary(state[..6*n].iter().copied()),
            "last_frame_latent":finite_summary(state[6*n..(6+config.latent_channels as usize)*n].iter().copied()),
        })
    }).collect();
    serde_json::json!({"scope":"read-only, post-update; predictions are the carried final-frame forward outputs before the optimizer update",
        "cursors":cursors})
}

fn main() -> Result<()> {
    env_logger::init();
    let mut data = Vec::new();
    let mut eval = Vec::new();
    let mut out = PathBuf::from("runs/transport");
    let mut steps = 4000usize;
    let mut unroll = 4usize;
    let mut channels = 16;
    let mut levels = 3;
    let mut seed = 7u64;
    let mut rate = 0.0003f32;
    let mut eval_only = false;
    let mut profile_only = false;
    let mut checkpoint_input = None::<PathBuf>;
    let mut eval_every = 10_000usize;
    let mut batch_size = 8usize;
    let mut crop = 64u32;
    let mut stop_after = None::<usize>;
    let mut device_id = None;
    let mut weights = graph::LossWeights::default();
    let mut evaluation = EvaluationOptions::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!(
                "transport --data TRAIN.omd --eval-data DEV.omd --out DIR
  --steps N [4000] --unroll N [4] --channels N [16] --levels N [3] --seed N [7]
  --lr F [0.0003] --eval-every N [10000] (causal metrics, no images) --device-id ID
  --batch N [8] --crop N [64] (LR pixels; persistent GPU cursors)\n  --stop-after N (checkpoint an interruption; --steps retains the planned schedule)
  --reset-every N [off] (evaluation: simulated cut each N frames, within each sequence)
  --checkpoint FILE (evaluation, or true resume with matching training bundle)
  --eval-only (loads checkpoint sidecar; evaluation resolution may differ)
  --profile-only (training-loop measurement; save checkpoint/timings, skip evaluation)
  --reset-history (eval-only diagnostic: reset the model before every frame)
  --save-lobes (save diffuse/specular images alongside per-frame diagnostics)
  --save-linear (save row-major, little-endian scene-linear RGB f32 for crop scoring)
  --alpha-frame SEQUENCE:FRAME (repeatable, eval-only: raw gates, maps and histograms)
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
            "--levels" => levels = v.parse()?,
            "--seed" => seed = v.parse()?,
            "--lr" => rate = v.parse()?,
            "--eval-every" => eval_every = v.parse()?,
            "--batch" => batch_size = v.parse()?,
            "--crop" => crop = v.parse()?,
            "--stop-after" => stop_after = Some(v.parse()?),
            "--reset-every" => {
                if evaluation.reset_every.replace(v.parse()?).is_some() {
                    return Err("specify only one evaluation reset option".into());
                }
            }
            "--control-run" => evaluation.control_run = Some(PathBuf::from(v).canonicalize()?),
            "--checkpoint" => checkpoint_input = Some(v.into()),
            "--device-id" => device_id = Some(ommatidia::gpu::parse_device_id(&v)?),
            "--alpha-frame" => {
                let (sequence, frame) = v
                    .split_once(':')
                    .ok_or("alpha frame must be SEQUENCE:FRAME")?;
                let pair = [sequence.parse()?, frame.parse()?];
                if evaluation.alpha_frames.contains(&pair) {
                    return Err("duplicate alpha diagnostic frame".into());
                }
                evaluation.alpha_frames.push(pair);
            }
            "--compressed-weight" => weights.compressed = v.parse()?,
            "--physical-weight" => weights.physical = v.parse()?,
            "--low-frequency-weight" => weights.low_frequency = v.parse()?,
            "--temporal-weight" => weights.temporal = v.parse()?,
            "--lobe-weight" => weights.lobes = v.parse()?,
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    weights.validate()?;
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
    if !eval_only
        && (evaluation.save_linear || evaluation.save_lobes || !evaluation.alpha_frames.is_empty())
    {
        return Err(
            "--save-linear/--save-lobes/--alpha-frame require --eval-only; training checks save metrics only"
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
            levels,
            ..Config::default()
        }
    };
    if eval.is_empty() {
        return Err("--eval-data required".into());
    }
    let holdout = Corpus::open(&eval, config)?;
    let low = holdout.low;
    let inference_macs = graph::build(config, low, 0)?.macs();
    println!(
        "{inference_macs} convolution MACs/frame; {:.6} GFLOP/frame (forward convolutions only)",
        inference_macs as f64 * 2.0e-9
    );
    let context = ommatidia::gpu::create_context(device_id, false);
    let mut learned = if evaluation.alpha_frames.is_empty() {
        native::Native::new(Arc::clone(&context), config, low)?
    } else {
        native::Native::with_alpha_output(Arc::clone(&context), config, low)?
    };
    if eval_only {
        learned
            .session
            .load_checkpoint(checkpoint_input.as_ref().unwrap())?;
    } else {
        if data.is_empty() {
            return Err("--data required".into());
        }
        let train = Arc::new(Corpus::open(&data, config)?);
        train.disjoint(&holdout)?;
        if train.low != low || unroll > train.length {
            return Err("training/evaluation extent or unroll mismatch".into());
        }
        let settings = Settings {
            steps,
            batch: batch_size,
            unroll,
            crop: [crop; 2],
            margin: 4,
            peak_rate: rate,
            seed,
            weights,
            model: ommatidia_train::checkpoint::model_text(config)?,
        };
        let mut sampler = Sampler::new(
            seed,
            batch_size,
            [crop; 2],
            unroll,
            low,
            train.sequences.len(),
            train.length,
        )?;
        let last_step = stop_after.unwrap_or(steps);
        if last_step == 0 || last_step > steps || steps > u32::MAX as usize {
            return Err("invalid stop/schedule boundary".into());
        }
        let mut trainer = Trainer::new(
            Arc::clone(&context),
            config,
            [crop; 2],
            unroll,
            batch_size,
            4,
        )?;
        let (first_step, means) = if let Some(path) = &checkpoint_input {
            let restored = Checkpoint::restore(path, &mut trainer, &settings, &train.identities())?;
            sampler = restored.sampler;
            (restored.step, restored.means)
        } else {
            let means = train.means()?;
            trainer.initialize(seed, means);
            (0, means)
        };
        if first_step >= last_step {
            return Err("checkpoint already reaches the requested stop boundary".into());
        }
        trainer.share_parameters(&mut learned.session)?;
        let parameters = trainer.network.params.iter().map(|p| p.len).sum::<usize>();
        println!(
            "{parameters} parameters; {} scenes; {batch_size} cursors, {unroll} frames, {crop}x{crop} LR crops",
            train.sequences.len()
        );
        std::fs::write(
            out.join("training.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "settings":settings, "start_step":first_step, "stop_step":last_step,
                "optimizer_resumed":checkpoint_input.is_some(), "corpus_mean_normalized_lobes":means,
                "inference_macs":inference_macs, "inference_flops":2*inference_macs, "parameters":parameters,
                "evaluation":{"every":eval_every,"history_mode":"causal","pngs":false,"shared_parameters":true},
                "training":train.provenance, "development":holdout.provenance, "capture_identities":train.identities(),
                "mapped_bytes":train.identities().iter().map(|i| i.bytes).sum::<usize>(),
                "preparation":"GPU observation packing and geometric maps; GPU detached state carry",
                "io":"immutable memory-mapped f16 crops, bounded one-worker prefetch; no requantization",
                "sampling":"persistent 8-16-window crop lives, uniform fitting start, 0.1 extra reset probability; 2^U(-2,2) radiance gain per life",
                "loss_margin_hr":4,"loss_margin_scope":"artificial crop borders only; real image edges supervised",
                "exposure":1,"gradient_clip_norm":1,
                "loss_outliers":{"threshold":1,"file":"loss-outliers.jsonl","policy":"diagnostic only; no skipped/clamped batches"},
                "resume":"Adam moments/step, full GPU state, consumed-batch sampler RNG/cursors and fixed schedule",
            }))?,
        )?;
        let prefetch = Prefetch::new(Arc::clone(&train), config, sampler.clone());
        let mut losses = std::fs::File::create(out.join("loss.csv"))?;
        let mut outliers = std::fs::File::create(out.join("loss-outliers.jsonl"))?;
        writeln!(
            losses,
            "update,loss,learning_rate,cold_windows,windows,cold_fraction"
        )?;
        let mut rows = std::fs::File::create(out.join("profile.csv"))?;
        writeln!(
            rows,
            "update,io_wait,worker_decode,upload,preparation_submit,step_wait,carry_wait,loss_read,total,supervised_pixel_gradients"
        )?;
        let mut totals = [0.0f64; 8];
        let mut steady = [0.0f64; 8];
        let mut supervised = 0_usize;
        let mut steady_supervised = 0_usize;
        for update in first_step + 1..=last_step {
            let start = Instant::now();
            let batch = prefetch.receive()?;
            let io = start.elapsed().as_secs_f64();
            let lr = learning_rate(rate, update, steps);
            let (loss, timing) = trainer.step(&batch, lr, weights)?;
            sampler = batch.next_sampler.clone();
            let cold = batch.windows.iter().filter(|w| w.reset).count();
            let fraction = sampler.cold_windows as f64 / sampler.windows as f64;
            let valid_pixels = batch
                .windows
                .iter()
                .map(|w| sampler.supervised_pixels(w.origin, config.scale, 4) * unroll)
                .sum::<usize>();
            supervised += valid_pixels;
            if update > first_step + 10 {
                steady_supervised += valid_pixels;
            }
            writeln!(
                losses,
                "{update},{loss},{lr},{cold},{batch_size},{fraction}"
            )?;
            // Observation only: the fixed threshold matches the Phase 3 spike
            // report. It never drops a batch, changes the objective or clips data.
            if loss > 1.0 {
                let mut diagnostic = loss_diagnostics(&mut trainer, &batch, config);
                diagnostic["update"] = update.into();
                diagnostic["mean_loss"] = loss.into();
                writeln!(outliers, "{}", serde_json::to_string(&diagnostic)?)?;
                outliers.flush()?;
            }
            let total = start.elapsed().as_secs_f64();
            let values = [
                io,
                batch.decode_seconds,
                timing.upload,
                timing.preparation_submit,
                timing.step_wait,
                timing.carry_wait,
                timing.loss_read,
                total,
            ];
            write!(rows, "{update}")?;
            for value in values {
                write!(rows, ",{value}")?;
            }
            writeln!(rows, ",{valid_pixels}")?;
            for i in 0..8 {
                totals[i] += values[i];
                if update > first_step + 10 {
                    steady[i] += values[i];
                }
            }
            if update == first_step + 1 || update % 16 == 0 {
                println!(
                    "update {update}/{steps}: loss {loss:.7}, {:.2} updates/s, cold {:.2}%",
                    (update - first_step) as f64 / totals[7],
                    fraction * 100.0
                );
            }
            if trainer.session.adam_step_count() as usize != update {
                return Err("expected exactly one Adam step per batch".into());
            }
            if update % 5000 == 0 || update == last_step {
                let directory = out.join("checkpoints").join(format!("step-{update:08}"));
                Checkpoint::save(
                    &directory,
                    &mut trainer,
                    settings.clone(),
                    train.identities(),
                    sampler.clone(),
                    means,
                )?;
            }
            if eval_every > 0 && update % eval_every == 0 && update < last_step {
                let dir = out.join(format!("dev-{update}"));
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
                println!("development {update}: {}", serde_json::to_string(&report)?);
            }
        }
        drop(prefetch);
        let pixels = batch_size * unroll * (crop * config.scale).pow(2) as usize;
        let baseline = 131072.0 / 0.438300;
        let report_timing = |count: usize, values: [f64; 8], supervised: usize| {
            serde_json::json!({
                "updates":count,"seconds":values[7],"updates_per_second":count as f64 / values[7],
                "nominal_pixel_gradients_per_second":count as f64*pixels as f64/values[7],
                "valid_pixel_gradients":supervised,
                "valid_pixel_gradients_per_second":supervised as f64/values[7],
                "speedup_vs_phase1_valid":supervised as f64/values[7]/baseline,
                "seconds_by_stage":{"io_wait":values[0],"worker_decode_overlapped":values[1],"upload":values[2],"preparation_submit":values[3],"step_wait":values[4],"carry_wait":values[5],"loss_read":values[6]},
            })
        };
        let profile = serde_json::json!({
            "scope":"host update wall time including I/O and CSV loss write; excludes initialization, checkpoint, evaluation and profile CSV write; worker decode overlaps GPU work",
            "device":context.device_information().device_name,
            "phase1_pixel_gradients_per_second":baseline,"nominal_pixels_per_update":pixels,
            "valid_pixels_per_update":supervised as f64 / (last_step-first_step) as f64,
            "all_updates":report_timing(last_step-first_step, totals, supervised),
            "after_first_10_updates":if last_step-first_step > 10 { report_timing(last_step-first_step-10, steady, steady_supervised) } else { serde_json::Value::Null },
            "cold_fraction":sampler.cold_windows as f64 / sampler.windows as f64,
        });
        std::fs::write(
            out.join("profile.json"),
            serde_json::to_vec_pretty(&profile)?,
        )?;
        println!("training profile: {}", serde_json::to_string(&profile)?);
        let final_dir = out.join("checkpoints").join(format!("step-{last_step:08}"));
        for name in [
            "model.safetensors",
            "model.transport.ron",
            "state.f32",
            "trainer.json",
        ] {
            std::fs::copy(final_dir.join(name), out.join(name))?;
        }
        // Independent storage is essential: reloading a shared session would
        // overwrite the live parameters and make this comparison tautological.
        learned = native::Native::new(Arc::clone(&context), config, low)?;
        learned.session.load_checkpoint(&checkpoint)?;
        let names: Vec<_> = trainer
            .network
            .params
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        for ((name, live), loaded) in names
            .iter()
            .zip(trainer.session.read_params(&names))
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
    fn alpha_histogram_preserves_endpoints_and_rejects_invalid_values() {
        let s = alpha_summary(&[0.0, 0.049, 0.5, 0.999, 1.0]).unwrap();
        let histogram = s["histogram"].as_array().unwrap();
        assert_eq!(histogram[0], 2);
        assert_eq!(histogram[10], 1);
        assert_eq!(histogram[19], 2);
        assert_eq!(
            histogram.iter().map(|n| n.as_u64().unwrap()).sum::<u64>(),
            5
        );
        for invalid in [f32::NAN, f32::INFINITY, -0.001, 1.001] {
            assert!(alpha_summary(&[invalid]).is_err());
        }
        assert!(alpha_summary(&[]).is_err());
    }

    #[test]
    fn outlier_summary_preserves_extremes_and_reports_nonfinite_values() {
        let s = finite_summary([0.0, 2.0, 10.0, f32::NAN, f32::INFINITY].into_iter());
        assert_eq!(s["finite"], 3);
        assert_eq!(s["nonfinite"], 2);
        assert_eq!(s["min"], 0.0);
        assert_eq!(s["max"], 10.0);
        assert_eq!(s["mean"], 4.0);
        let empty = finite_summary(std::iter::empty());
        assert!(empty["min"].is_null() && empty["max"].is_null() && empty["mean"].is_null());
    }

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
