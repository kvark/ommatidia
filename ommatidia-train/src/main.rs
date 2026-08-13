//! Train the ommatidia reconstruction network on a generated dataset.
//!
//! ```sh
//! cargo run --release -p ommatidia-train -- --data data/train.omd --steps 2000
//! ```
//!
//! The model shape is taken from the dataset where it can be: the scale factor
//! and the conditioning planes both come out of the file header, so a model
//! cannot be built that asks for channels the data does not carry.

mod batcher;
mod eval;

use std::path::PathBuf;

use ommatidia::batch::Crop;
use ommatidia::checkpoint;
use ommatidia::dataset::Reader;
use ommatidia::diffusion::Schedule;
use ommatidia::model::{self, Backbone, ModelConfig, Objective};

/// How a dataset is divided between fitting and scoring.
///
/// The tail of the file is held out. Samples are independent scenes, so a
/// contiguous split is as good as a shuffled one, and it has the advantage of
/// being reproducible without a seed.
#[derive(Clone, Copy, Debug)]
struct Split {
    train: usize,
    total: usize,
}

impl Split {
    /// Reserve `fraction` of the set for validation, always leaving at least
    /// one sample on each side.
    fn new(total: usize, fraction: f32) -> Self {
        let held = ((total as f32 * fraction).round() as usize).clamp(1, total.saturating_sub(1));
        Self {
            train: total - held,
            total,
        }
    }

    fn training(&self) -> std::ops::Range<usize> {
        0..self.train
    }

    fn validation(&self) -> std::ops::Range<usize> {
        self.train..self.total
    }
}

/// Samples drawn to measure the residual gain. Enough to average out the
/// content variation between scenes without reading the whole set.
const GAIN_PROBE_SAMPLES: usize = 16;

struct Args {
    data: PathBuf,
    out: PathBuf,
    steps: usize,
    batch: u32,
    tile: u32,
    learning_rate: f32,
    learning_rate_final: Option<f32>,
    grad_clip: f32,
    base_channels: u32,
    levels: usize,
    blocks_per_level: usize,
    num_groups: u32,
    backbone: Backbone,
    timesteps: usize,
    sampler_steps: usize,
    objective: Objective,
    seed: u64,
    log_every: usize,
    eval_out: Option<PathBuf>,
    color_only: bool,
    val_fraction: f32,
    eval_crops: usize,
    eval_every: usize,
    checkpoint_every: usize,
    eval_only: bool,
    allow_filtered_input: bool,
    device_id: Option<u32>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            data: PathBuf::from("data/train.omd"),
            out: PathBuf::from("runs/ommatidia"),
            steps: 1000,
            batch: 8,
            tile: 64,
            learning_rate: 2e-4,
            learning_rate_final: None,
            grad_clip: 1.0,
            base_channels: 24,
            levels: 3,
            blocks_per_level: 1,
            num_groups: 8,
            backbone: Backbone::Conv,
            timesteps: 1000,
            sampler_steps: 20,
            objective: Objective::Direct,
            seed: 0,
            log_every: 50,
            eval_out: None,
            color_only: false,
            val_fraction: 0.15,
            eval_crops: 64,
            eval_every: 0,
            checkpoint_every: 0,
            eval_only: false,
            allow_filtered_input: false,
            device_id: None,
        }
    }
}

const USAGE: &str = "\
train the ommatidia reconstruction network

usage: ommatidia-train [options]

  --data PATH          dataset to train on  [data/train.omd]
  --out STEM           checkpoint stem, gets .safetensors and .ron  [runs/ommatidia]
  --steps N            optimizer steps  [1000]
  --batch N            crops per step  [8]
  --tile N             square crop size, in input pixels  [64]
  --lr F               Adam learning rate  [2e-4]
  --lr-final F         decay the rate to this by the last step, on a cosine.
                       Worth setting on a long run: a rate that was right for
                       the first hour is too coarse to settle in the last one
  --grad-clip F        clip the gradient norm to this, 0 to disable  [1.0]
  --base-channels N    channel width of the first level  [24]
  --levels N           U-Net levels  [3]
  --blocks N           residual blocks per level  [1]
  --num-groups N       GroupNorm groups; must divide every level width  [8]
  --backbone KIND      conv or hybrid-window  [conv]
  --attention-window N local attention window at the bottleneck  [8]
  --attention-head-dim channels in each attention head  [16]
  --timesteps N        diffusion schedule length  [1000]
  --sampler-steps N    DDIM steps used when evaluating  [20]
  --objective KIND     direct or diffusion  [direct]
  --seed N             seed for init and batching  [0]
  --device-id ID       adapter ID for this standalone process (hex or decimal)
  --log-every N        steps between loss lines  [50]
  --eval-out DIR       write comparison PNGs of the first held-out crop
  --val-fraction F     share of the set held out for scoring  [0.15]
  --eval-crops N       cap on held-out crops scored  [64]
  --eval-every N       score the held-out set every N steps, 0 for only at
                       the end  [0]
  --checkpoint-every N save the checkpoint every N steps as well as at the
                       end, so a long run survives a crash  [0]
  --eval-only          load the checkpoint at --out and score it without
                       training, so a finished run can be re-examined under a
                       different --sampler-steps
  --color-only         condition on colour alone, ignoring the dataset's
                       G-buffer planes; the other half of that ablation is
                       simply leaving this off
  --allow-filtered-input
                       allow a legacy dataset whose input already passed
                       through Blade's SVGF denoiser; for historical
                       comparisons only, not a denoiser-replacement model
  -h, --help           this message
";

fn parse_args() -> Result<Args, String> {
    parse_from(std::env::args().skip(1))
}

fn parse_from(argv: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut args = Args::default();
    let mut argv = argv;
    while let Some(flag) = argv.next() {
        let mut value = || argv.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--data" => args.data = PathBuf::from(value()?),
            "--out" => args.out = PathBuf::from(value()?),
            "--steps" => args.steps = value()?.parse().map_err(|e| format!("--steps: {e}"))?,
            "--batch" => args.batch = value()?.parse().map_err(|e| format!("--batch: {e}"))?,
            "--tile" => args.tile = value()?.parse().map_err(|e| format!("--tile: {e}"))?,
            "--lr" => args.learning_rate = value()?.parse().map_err(|e| format!("--lr: {e}"))?,
            "--lr-final" => {
                args.learning_rate_final =
                    Some(value()?.parse().map_err(|e| format!("--lr-final: {e}"))?)
            }
            "--grad-clip" => {
                args.grad_clip = value()?.parse().map_err(|e| format!("--grad-clip: {e}"))?
            }
            "--base-channels" => {
                args.base_channels = value()?
                    .parse()
                    .map_err(|e| format!("--base-channels: {e}"))?
            }
            "--levels" => args.levels = value()?.parse().map_err(|e| format!("--levels: {e}"))?,
            "--blocks" => {
                args.blocks_per_level = value()?.parse().map_err(|e| format!("--blocks: {e}"))?
            }
            "--num-groups" => {
                args.num_groups = value()?.parse().map_err(|e| format!("--num-groups: {e}"))?
            }
            "--backbone" => {
                args.backbone = match value()?.as_str() {
                    "conv" => Backbone::Conv,
                    "hybrid-window" => match args.backbone {
                        Backbone::Conv => Backbone::HybridWindowAttention {
                            window: 8,
                            head_dim: 16,
                        },
                        configured @ Backbone::HybridWindowAttention { .. } => configured,
                    },
                    other => return Err(format!("unknown backbone {other:?}")),
                }
            }
            "--attention-window" => {
                let window = value()?
                    .parse()
                    .map_err(|e| format!("--attention-window: {e}"))?;
                let head_dim = match args.backbone {
                    Backbone::Conv => 16,
                    Backbone::HybridWindowAttention { head_dim, .. } => head_dim,
                };
                args.backbone = Backbone::HybridWindowAttention { window, head_dim };
            }
            "--attention-head-dim" => {
                let head_dim = value()?
                    .parse()
                    .map_err(|e| format!("--attention-head-dim: {e}"))?;
                let window = match args.backbone {
                    Backbone::Conv => 8,
                    Backbone::HybridWindowAttention { window, .. } => window,
                };
                args.backbone = Backbone::HybridWindowAttention { window, head_dim };
            }
            "--timesteps" => {
                args.timesteps = value()?.parse().map_err(|e| format!("--timesteps: {e}"))?
            }
            "--sampler-steps" => {
                args.sampler_steps = value()?
                    .parse()
                    .map_err(|e| format!("--sampler-steps: {e}"))?
            }
            "--objective" => {
                args.objective = match value()?.as_str() {
                    "diffusion" => Objective::Diffusion,
                    "direct" => Objective::Direct,
                    other => return Err(format!("unknown objective {other:?}")),
                }
            }
            "--seed" => args.seed = value()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--device-id" => args.device_id = Some(ommatidia::gpu::parse_device_id(&value()?)?),
            "--log-every" => {
                args.log_every = value()?.parse().map_err(|e| format!("--log-every: {e}"))?
            }
            "--eval-out" => args.eval_out = Some(PathBuf::from(value()?)),
            "--color-only" => args.color_only = true,
            "--eval-only" => args.eval_only = true,
            "--allow-filtered-input" => args.allow_filtered_input = true,
            "--val-fraction" => {
                args.val_fraction = value()?
                    .parse()
                    .map_err(|e| format!("--val-fraction: {e}"))?
            }
            "--eval-crops" => {
                args.eval_crops = value()?.parse().map_err(|e| format!("--eval-crops: {e}"))?
            }
            "--eval-every" => {
                args.eval_every = value()?.parse().map_err(|e| format!("--eval-every: {e}"))?
            }
            "--checkpoint-every" => {
                args.checkpoint_every = value()?
                    .parse()
                    .map_err(|e| format!("--checkpoint-every: {e}"))?
            }
            other => return Err(format!("unknown option {other:?}\n\n{USAGE}")),
        }
    }
    if args.steps == 0 {
        return Err("--steps must be positive".into());
    }
    if !(0.0..1.0).contains(&args.val_fraction) {
        return Err(format!(
            "--val-fraction must be in [0, 1), got {}",
            args.val_fraction
        ));
    }
    if args.eval_crops == 0 {
        return Err("--eval-crops must be positive".into());
    }
    Ok(args)
}

fn validate_input_source(layout: &ommatidia::dataset::Layout, args: &Args) -> Result<(), String> {
    if layout.lr_source != ommatidia::dataset::InputSource::Svgf || args.allow_filtered_input {
        return Ok(());
    }
    Err(format!(
        "{} contains {:?} low-resolution input; refusing to train a \
         reconstruction model on another denoiser's output. Regenerate it with the \
         current ommatidia-data, or pass --allow-filtered-input for a historical comparison.",
        args.data.display(),
        layout.lr_source,
    ))
}

fn main() {
    env_logger::init();
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };

    let mut reader = match Reader::open(&args.data) {
        Ok(reader) => reader,
        Err(e) => {
            eprintln!("cannot open {}: {e}", args.data.display());
            std::process::exit(1);
        }
    };
    let layout = *reader.layout();
    if let Err(message) = validate_input_source(&layout, &args) {
        eprintln!("{message}");
        std::process::exit(1);
    }
    if reader.is_empty() {
        eprintln!("{} holds no samples", args.data.display());
        std::process::exit(1);
    }
    let tile = args.tile.min(layout.lr_width).min(layout.lr_height);
    if tile != args.tile {
        println!(
            "tile clamped to {tile}, the dataset is only {}x{}",
            layout.lr_width, layout.lr_height
        );
    }

    // The residual is small, and how small depends on the content and the
    // scale factor, so it is measured rather than assumed. Without this the
    // diffusion objective trains to a low loss and samples to pure noise.
    let probe = GAIN_PROBE_SAMPLES.min(reader.len());
    let gain = {
        let samples: Vec<_> = (0..probe)
            .filter_map(|i| reader.sample(i * reader.len() / probe.max(1)).ok())
            .collect();
        ommatidia::batch::estimate_gain(samples, &layout)
    };
    println!(
        "residual gain {gain:.2} (standard deviation {:.4}), measured over {probe} samples",
        1.0 / gain
    );

    // Everything about the data comes from the data, so the network cannot ask
    // for a plane the dataset does not carry or upscale by the wrong factor.
    let config = ModelConfig {
        scale: layout.scale,
        tile,
        batch: args.batch,
        // Restricting rather than regenerating keeps an ablation honest: both
        // arms then see the same bytes, the same crops, and the same batch
        // order, and differ only in which channels reach the network.
        cond_planes: if args.color_only {
            ommatidia::PlaneSet::new().with(ommatidia::Plane::Color)
        } else {
            layout.lr_planes
        },
        base_channels: args.base_channels,
        level_multipliers: (0..args.levels).map(|i| 1 << i).collect(),
        blocks_per_level: args.blocks_per_level,
        num_groups: args.num_groups,
        backbone: args.backbone,
        residual_gain: gain,
        objective: args.objective,
        ..ModelConfig::default()
    };
    if let Err(message) = config.validate() {
        eprintln!("model configuration is invalid: {message}");
        std::process::exit(1);
    }

    let split = Split::new(reader.len(), args.val_fraction);
    println!(
        "{} samples for training, {} held out for scoring",
        split.training().len(),
        split.validation().len()
    );

    let schedule = Schedule::cosine(args.timesteps);

    if args.eval_only {
        // The configuration the checkpoint carries wins over anything rebuilt
        // from the flags, because the weights only mean anything in the graph
        // they were fitted in. What stays under the caller's control is
        // everything outside that graph — the sampler budget above all, which
        // is the reason to re-score a finished run at all.
        let (stored, paths) = match checkpoint::load_config(&args.out) {
            Ok(pair) => pair,
            Err(e) => {
                eprintln!("cannot read the checkpoint at {}: {e}", args.out.display());
                std::process::exit(1);
            }
        };
        let mut evaluator = Evaluator::new(
            &stored,
            ommatidia::gpu::create_context(args.device_id, false),
        );
        if let Err(e) = evaluator.session.load_checkpoint(&paths.weights) {
            eprintln!("cannot load {}: {e}", paths.weights.display());
            std::process::exit(1);
        }
        let mut batcher = batcher::Batcher::new(
            reader,
            stored,
            schedule.clone(),
            split.training(),
            args.seed,
        );
        println!(
            "scoring {} with {} sampler steps",
            paths.weights.display(),
            args.sampler_steps
        );
        evaluator.score(
            &mut batcher,
            split,
            &args,
            &schedule,
            args.eval_out.as_deref(),
        );
        return;
    }

    let model = model::build(&config, true).expect("validated above");
    println!(
        "{} samples, {}x{} -> {}x{}, {} conditioning channels, {} output channels",
        reader.len(),
        layout.lr_width,
        layout.lr_height,
        layout.hr_width(),
        layout.hr_height(),
        config.cond_channels(),
        config.target_channels(),
    );
    println!(
        "conditioning on {}",
        config
            .cond_planes
            .iter()
            .map(|p| format!("{p:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let parameter_count: usize = model.params.iter().map(|p| p.len).sum();
    println!(
        "{:?} objective, {:?} backbone, {} levels, {parameter_count} parameters",
        config.objective,
        config.backbone,
        config.levels()
    );
    // What this shape would cost at a real output extent, so a configuration
    // can be ruled out before it is trained rather than after.
    println!(
        "estimated arithmetic: {:.1} GFLOP per 1080p frame",
        config.flops(1920 * 1080)
    );

    print!("compiling... ");
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    let started = std::time::Instant::now();
    // One context, shared by the training session and the evaluator. Two
    // would contend for the same queue, and neither would land on the right
    // adapter without being told which one.
    let context = ommatidia::gpu::create_context(args.device_id, false);
    let mut session = ommatidia::gpu::training_session(&model.graph, context.clone());
    println!(
        "{:.1}s, {} dispatches on {}",
        started.elapsed().as_secs_f32(),
        session.plan().dispatches.len(),
        session.device_information().device_name,
    );
    model.initialize(&mut session, args.seed);
    session.set_adam(args.learning_rate, 0.9, 0.999, 1e-8);
    // On by default. A run measured in hours has many more chances to meet the
    // one batch that blows the gradient up, and the cost of that is the whole
    // run rather than one step. Clipping every fifth step amortises the extra
    // readback while still catching the collapse, which builds over thousands
    // of steps rather than one.
    if args.grad_clip > 0.0 {
        session.set_grad_clip_norm(args.grad_clip);
        session.set_grad_clip_every(5);
    }

    let mut batcher = batcher::Batcher::new(
        reader,
        config.clone(),
        schedule.clone(),
        split.training(),
        args.seed,
    );
    let diffusing = config.objective == Objective::Diffusion;

    // Built up front so scoring mid-run costs a parameter copy rather than a
    // compile and a checkpoint round trip.
    print!("building the evaluation session... ");
    let _ = std::io::stdout().flush();
    let mut evaluator = Evaluator::new(&config, context.clone());
    println!("done");

    let mut recent = Vec::new();
    let mut first_average = f32::NAN;
    let mut last_average = f32::NAN;
    let training_started = std::time::Instant::now();

    for step in 0..args.steps {
        if let Some(final_rate) = args.learning_rate_final {
            // Cosine from the initial rate down to the final one. Nothing
            // exotic; the point is only that the last hour of a long run
            // settles rather than keeps bouncing at the rate that suited the
            // first one.
            let progress = step as f32 / (args.steps.max(2) - 1) as f32;
            let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            session.set_learning_rate(final_rate + (args.learning_rate - final_rate) * cosine);
        }

        let batch = batcher.next().expect("cannot read a batch");
        session.set_input("cond", &batch.cond);
        session.set_input("target", &batch.target);
        if diffusing {
            session.set_input("x_t", &batch.x_t);
            session.set_input("t_emb", &batch.t_emb);
        }

        session.step();
        session.wait();
        let loss = session.read_loss();
        if !loss.is_finite() {
            eprintln!("loss went non-finite at step {step}, stopping");
            std::process::exit(1);
        }
        recent.push(loss);

        if (step + 1) % args.log_every == 0 || step + 1 == args.steps {
            let average = recent.iter().sum::<f32>() / recent.len() as f32;
            recent.clear();
            if first_average.is_nan() {
                first_average = average;
            }
            last_average = average;
            let rate = (step + 1) as f32 / training_started.elapsed().as_secs_f32();
            let elapsed = training_started.elapsed().as_secs_f32();
            println!(
                "step {:>7}: loss {average:.6}  ({rate:.1} steps/s, {:.0}m elapsed)",
                step + 1,
                elapsed / 60.0,
            );
        }

        let last = step + 1 == args.steps;
        if args.eval_every > 0 && (step + 1) % args.eval_every == 0 && !last {
            evaluator.sync(&session);
            evaluator.score(&mut batcher, split, &args, &schedule, None);
        }
        if args.checkpoint_every > 0 && (step + 1) % args.checkpoint_every == 0 && !last {
            // A long run should not lose everything to a crash, and an
            // intermediate checkpoint is also what makes an overtrained run
            // recoverable.
            if let Err(e) = checkpoint::save(&mut session, &config, &args.out) {
                eprintln!("cannot save the checkpoint: {e}");
            }
        }
    }

    println!(
        "loss {first_average:.6} -> {last_average:.6} over {} steps in {:.1}s",
        args.steps,
        training_started.elapsed().as_secs_f32()
    );

    match checkpoint::save(&mut session, &config, &args.out) {
        Ok(paths) => println!(
            "wrote {} and {}",
            paths.weights.display(),
            paths.config.display()
        ),
        Err(e) => {
            eprintln!("cannot save the checkpoint: {e}");
            std::process::exit(1);
        }
    }

    drop(session);
    evaluator.score(
        &mut batcher,
        split,
        &args,
        &schedule,
        args.eval_out.as_deref(),
    );
}

/// Holds an inference session alongside the training one, so the held-out set
/// can be scored mid-run without a checkpoint round trip.
///
/// The training graph ends at the loss and has the training batch size baked
/// in, so sampling needs its own. Parameters do not depend on the batch, which
/// is what lets them be copied straight across.
struct Evaluator {
    session: meganeura::Session,
    config: ModelConfig,
    names: Vec<String>,
    scratch: Vec<f32>,
}

impl Evaluator {
    /// Build the inference session for `config`.
    ///
    /// The parameter set is taken from the inference model rather than from a
    /// training one, because they are the same: only the batch differs between
    /// the two graphs, and no parameter's shape depends on it. That is also
    /// what lets a checkpoint be scored with no training session in sight.
    fn new(config: &ModelConfig, context: std::sync::Arc<blade_graphics::Context>) -> Self {
        let mut eval_config = config.clone();
        eval_config.batch = 1;
        let eval_model =
            model::build(&eval_config, false).expect("the caller validated this config");
        let session = ommatidia::gpu::inference_session(&eval_model.graph, context);
        let names = eval_model.params.iter().map(|p| p.name.clone()).collect();
        let widest = eval_model.params.iter().map(|p| p.len).max().unwrap_or(0);
        Self {
            session,
            config: eval_config,
            names,
            scratch: vec![0.0; widest],
        }
    }

    /// Copy every parameter from the training session into the inference one.
    fn sync(&mut self, from: &meganeura::Session) {
        for name in &self.names {
            let Some(len) = from.param_size(name) else {
                continue;
            };
            let slice = &mut self.scratch[..len];
            from.read_param(name, slice);
            self.session.upload_param(name, slice);
        }
    }

    /// Reconstruct the held-out samples and report against nearest upsampling.
    ///
    /// Over a grid of crops across every validation sample, not one crop of one
    /// training sample: a single tile is far too small and too lucky to
    /// separate two configurations, and scoring on data the network was fitted
    /// to measures memorisation rather than reconstruction.
    fn score(
        &mut self,
        batcher: &mut batcher::Batcher,
        split: Split,
        args: &Args,
        schedule: &Schedule,
        dir: Option<&std::path::Path>,
    ) -> Option<f32> {
        let layout = *batcher.layout();
        // Non-overlapping tiles, so no pixel is counted twice.
        let crops = Crop::grid(&layout, self.config.tile, self.config.tile);
        if crops.is_empty() {
            eprintln!("the tile is larger than a sample, nothing to evaluate");
            return None;
        }

        let mut baseline_total = 0.0f64;
        let mut network_total = 0.0f64;
        let mut baseline_ssim_total = 0.0f64;
        let mut network_ssim_total = 0.0f64;
        let mut counted = 0usize;
        let started = std::time::Instant::now();

        'outer: for index in split.validation() {
            let sample = match batcher.reader().sample(index) {
                Ok(sample) => sample,
                Err(e) => {
                    eprintln!("cannot read validation sample {index}: {e}");
                    break;
                }
            };
            for &crop in &crops {
                if counted >= args.eval_crops {
                    break 'outer;
                }
                let predicted = eval::reconstruct(
                    &mut self.session,
                    &self.config,
                    schedule,
                    &sample,
                    &layout,
                    crop,
                    args.sampler_steps,
                    // Vary the sampler noise per crop, so the score is not one
                    // lucky or unlucky draw repeated.
                    args.seed.wrapping_add(counted as u64),
                );
                let low = ommatidia::batch::crop_color(&sample, &layout, crop);
                let reference = ommatidia::batch::crop_reference(&sample, &layout, crop);
                let baseline = eval::nearest(
                    &low,
                    crop.tile as usize,
                    crop.tile as usize,
                    self.config.scale as usize,
                );

                baseline_total += eval::error(&baseline, &reference) as f64;
                network_total += eval::error(&predicted, &reference) as f64;
                let extent = (crop.tile * self.config.scale) as usize;
                baseline_ssim_total += eval::ssim(&baseline, &reference, extent, extent) as f64;
                network_ssim_total += eval::ssim(&predicted, &reference, extent, extent) as f64;

                // The first crop also goes out as images, for eyeballing.
                if counted == 0
                    && let Some(dir) = dir
                {
                    let hr_extent = crop.tile * self.config.scale;
                    for (name, image, width) in [
                        ("input", &low, crop.tile),
                        ("nearest", &baseline, hr_extent),
                        ("predicted", &predicted, hr_extent),
                        ("reference", &reference, hr_extent),
                    ] {
                        let path = dir.join(format!("{name}.png"));
                        if let Err(e) = eval::write_png(&path, image, width, width) {
                            eprintln!("cannot write {}: {e}", path.display());
                        }
                    }
                }
                counted += 1;
            }
        }

        if counted == 0 {
            eprintln!("no validation crops were evaluated");
            return None;
        }
        let baseline_error = baseline_total / counted as f64;
        let network_error = network_total / counted as f64;
        let baseline_psnr = -10.0 * baseline_error.log10();
        let network_psnr = -10.0 * network_error.log10();
        let baseline_ssim = baseline_ssim_total / counted as f64;
        let network_ssim = network_ssim_total / counted as f64;
        let gain = if network_error > 0.0 {
            10.0 * (baseline_error / network_error).log10()
        } else {
            f64::INFINITY
        };
        println!(
            "  held-out over {counted} crops in {:.1}s:\n  \
             nearest  MSE {baseline_error:.6}, PSNR {baseline_psnr:.2} dB, SSIM {baseline_ssim:.4}\n  \
             network  MSE {network_error:.6}, PSNR {network_psnr:.2} dB, SSIM {network_ssim:.4} \
             ({gain:+.2} dB PSNR gain)",
            started.elapsed().as_secs_f32()
        );
        Some(gain as f32)
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    fn parse(words: &[&str]) -> Result<Args, String> {
        parse_from(words.iter().map(|w| w.to_string()))
    }

    /// Every flag the usage text advertises has to actually be accepted.
    ///
    /// Adding the field, the default, and the usage line while forgetting the
    /// match arm compiles perfectly and fails only when someone passes the
    /// flag — which, on a run meant to last hours, is discovered late.
    #[test]
    fn every_documented_flag_is_parsed() {
        // A value that parses as a number, a float, and a path alike.
        const VALUE: &str = "1";
        for line in USAGE.lines() {
            let trimmed = line.trim_start();
            let Some(rest) = trimmed.strip_prefix("--") else {
                continue;
            };
            let flag = format!("--{}", rest.split_whitespace().next().unwrap());
            if flag == "--help" {
                continue; // exits the process
            }
            // Flags whose value is constrained beyond "parses as a number".
            // Arity is not listed: a flag is tried with a value and then
            // without, so a new switch needs no entry here to stay covered.
            let candidates: Vec<Vec<&str>> = match flag.as_str() {
                "--objective" => vec![vec!["--objective", "direct"]],
                "--backbone" => vec![vec!["--backbone", "hybrid-window"]],
                "--val-fraction" => vec![vec!["--val-fraction", "0.1"]],
                _ => vec![vec![&flag, VALUE], vec![&flag]],
            };
            let accepted = candidates.iter().any(|words| parse(words).is_ok());
            assert!(
                accepted,
                "{flag} is documented but not accepted, with a value or without: {:?}",
                parse(&candidates[0]).err()
            );
        }
    }

    #[test]
    fn the_periodic_flags_reach_the_fields() {
        let args = parse(&["--eval-every", "250", "--checkpoint-every", "500"]).unwrap();
        assert_eq!(args.eval_every, 250);
        assert_eq!(args.checkpoint_every, 500);
        // And stay off unless asked for, so a short run pays nothing.
        let plain = parse(&[]).unwrap();
        assert_eq!(plain.eval_every, 0);
        assert_eq!(plain.checkpoint_every, 0);
    }

    #[test]
    fn defaults_select_the_semi_realtime_direct_model() {
        let args = parse(&[]).unwrap();
        assert_eq!(args.objective, Objective::Direct);
        assert_eq!(args.base_channels, 24);
        assert_eq!(args.levels, 3);
        assert_eq!(args.blocks_per_level, 1);
        assert_eq!(args.batch, 8);
        assert_eq!(args.backbone, Backbone::Conv);
        assert!(!args.allow_filtered_input);
    }

    #[test]
    fn attention_flags_select_the_hybrid_backbone() {
        for argv in [
            [
                "--backbone",
                "hybrid-window",
                "--attention-window",
                "4",
                "--attention-head-dim",
                "8",
            ],
            [
                "--attention-head-dim",
                "8",
                "--attention-window",
                "4",
                "--backbone",
                "hybrid-window",
            ],
        ] {
            let args = parse(&argv).unwrap();
            assert_eq!(
                args.backbone,
                Backbone::HybridWindowAttention {
                    window: 4,
                    head_dim: 8,
                }
            );
        }
    }

    #[test]
    fn filtered_training_data_requires_an_explicit_override() {
        let mut layout = ommatidia::dataset::Layout {
            scale: 2,
            lr_width: 8,
            lr_height: 8,
            lr_source: ommatidia::dataset::InputSource::Svgf,
            lr_planes: ommatidia::PlaneSet::new().with(ommatidia::Plane::Color),
            hr_planes: ommatidia::PlaneSet::new().with(ommatidia::Plane::Color),
        };
        let plain = parse(&[]).unwrap();
        assert!(validate_input_source(&layout, &plain).is_err());

        let overridden = parse(&["--allow-filtered-input"]).unwrap();
        assert!(validate_input_source(&layout, &overridden).is_ok());

        layout.lr_source = ommatidia::dataset::InputSource::RawRestir;
        assert!(validate_input_source(&layout, &plain).is_ok());
    }

    #[test]
    fn an_unknown_flag_is_rejected() {
        assert!(parse(&["--not-a-flag", "1"]).is_err());
    }
}
