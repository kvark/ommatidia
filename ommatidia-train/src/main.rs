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

use ommatidia::batch::{self, Crop};
use ommatidia::checkpoint;
use ommatidia::dataset::Reader;
use ommatidia::diffusion::Schedule;
use ommatidia::model::{self, ModelConfig, Objective, Prediction, ReconstructionBase};

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
    fn new(total: usize, sequence_length: usize, fraction: f32) -> Self {
        assert!(total.is_multiple_of(sequence_length));
        let sequences = total / sequence_length;
        let held_sequences =
            ((sequences as f32 * fraction).round() as usize).clamp(1, sequences.saturating_sub(1));
        let held = held_sequences * sequence_length;
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
    timesteps: usize,
    sampler_steps: usize,
    objective: Objective,
    prediction: Prediction,
    reconstruction_base: ReconstructionBase,
    kernel_radius: u32,
    demodulate: bool,
    demodulation_offset: f32,
    head_kernel: u32,
    temporal_weight: f32,
    teacher_every: usize,
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
    history_frames: u32,
    temporal_features: ommatidia::temporal::Features,
}

const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPSILON: f32 = 1e-8;

fn cosine_learning_rate(initial: f32, final_rate: f32, step: usize, steps: usize) -> f32 {
    let progress = step as f32 / (steps.max(2) - 1) as f32;
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    final_rate + (initial - final_rate) * cosine
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
            base_channels: 8,
            levels: 3,
            blocks_per_level: 1,
            num_groups: 8,
            timesteps: 1000,
            sampler_steps: 20,
            objective: Objective::Direct,
            prediction: Prediction::SubpixelResidual,
            reconstruction_base: ReconstructionBase::GuidedBilinear,
            kernel_radius: 2,
            demodulate: false,
            demodulation_offset: 0.25,
            head_kernel: 3,
            temporal_weight: 0.0,
            teacher_every: 250,
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
            history_frames: 1,
            temporal_features: ommatidia::temporal::Features::Variance,
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
  --base-channels N    channel width of the first level  [8]
  --levels N           U-Net levels  [3]
  --blocks N           residual blocks per level  [1]
  --num-groups N       GroupNorm groups; must divide every level width  [8]
  --timesteps N        diffusion schedule length  [1000]
  --sampler-steps N    DDIM steps used when evaluating  [20]
  --objective KIND     direct or diffusion  [direct]
  --prediction KIND    subpixel or low-color residual, or kernel for the
                       single-operation sample gather  [subpixel]
  --reconstruction-base KIND
                       nearest, bilinear, guided, hr-guided, or sample, which
                       is the only one with no separate denoise  [guided]
  --kernel-radius N    half-width of the neighbourhood a kernel gathers, in
                       input pixels  [2]
  --demodulate         gather radiance divided by albedo and multiply the exact
                       output-resolution albedo back, so the texture is put
                       back rather than reconstructed. Kernel checkpoints only
  --temporal-weight F  weight of the temporal term in the loss. A per-frame
                       squared error is indifferent to whether consecutive
                       frames agree, so without this a reconstruction fitted to
                       one will flicker whenever its input does  [0]
  --teacher-every N    steps between resynchronising the detached copy of the
                       network that the temporal target is built from  [250]
  --head-kernel N      kernel size of the output convolution. A kernel head is
                       wide, so at 3 it is a quarter of the arithmetic; the
                       features it reads already have a wide receptive field  [3]
  --demodulation-offset F
                       added to the albedo on both sides, bounding how far a
                       pixel can be rescaled. 0.05 allows 20x and loses 1.5 dB
                       to the compressed gather; 0.25 allows 4x  [0.25]
  --seed N             seed for init and batching  [0]
  --device-id ID       adapter ID for this standalone process (hex or decimal)
  --history-frames N   surface-reprojected sparse frames, 1 for spatial [1]
  --temporal-features KIND
                       basic or variance history conditioning [variance]
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
            "--prediction" => {
                args.prediction = match value()?.as_str() {
                    "subpixel" => Prediction::SubpixelResidual,
                    "low-color" => Prediction::LowResolutionResidual,
                    "kernel" => Prediction::SubpixelKernel,
                    other => return Err(format!("unknown prediction {other:?}")),
                }
            }
            "--reconstruction-base" => {
                args.reconstruction_base = match value()?.as_str() {
                    "nearest" => ReconstructionBase::Nearest,
                    "bilinear" => ReconstructionBase::Bilinear,
                    "guided" => ReconstructionBase::GuidedBilinear,
                    "hr-guided" => ReconstructionBase::HighResolutionGuided,
                    "sample" => ReconstructionBase::Sample,
                    other => return Err(format!("unknown reconstruction base {other:?}")),
                }
            }
            "--kernel-radius" => {
                args.kernel_radius = value()?
                    .parse()
                    .map_err(|e| format!("--kernel-radius: {e}"))?
            }
            "--head-kernel" => {
                args.head_kernel = value()?
                    .parse()
                    .map_err(|e| format!("--head-kernel: {e}"))?
            }
            "--temporal-weight" => {
                args.temporal_weight = value()?
                    .parse()
                    .map_err(|e| format!("--temporal-weight: {e}"))?
            }
            "--teacher-every" => {
                args.teacher_every = value()?
                    .parse()
                    .map_err(|e| format!("--teacher-every: {e}"))?
            }
            "--demodulate" => args.demodulate = true,
            "--demodulation-offset" => {
                args.demodulation_offset = value()?
                    .parse()
                    .map_err(|e| format!("--demodulation-offset: {e}"))?
            }
            "--seed" => args.seed = value()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--device-id" => args.device_id = Some(ommatidia::gpu::parse_device_id(&value()?)?),
            "--history-frames" => {
                args.history_frames = value()?
                    .parse()
                    .map_err(|e| format!("--history-frames: {e}"))?
            }
            "--temporal-features" => {
                args.temporal_features = match value()?.as_str() {
                    "basic" => ommatidia::temporal::Features::Basic,
                    "variance" => ommatidia::temporal::Features::Variance,
                    other => return Err(format!("unknown temporal features {other:?}")),
                }
            }
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
    if args.history_frames == 0 {
        return Err("--history-frames must be positive".into());
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
    if reader.sequence_length() > 1 && !args.eval_only && args.history_frames < 2 {
        eprintln!(
            "{} contains {}-frame sequences; select --history-frames 2 or greater",
            args.data.display(),
            reader.sequence_length(),
        );
        std::process::exit(1);
    }
    if reader.sequence_length() == 1 && args.history_frames > 1 {
        eprintln!("--history-frames needs a sequence dataset");
        std::process::exit(1);
    }
    if let Err(message) = validate_input_source(&layout, &args) {
        eprintln!("{message}");
        std::process::exit(1);
    }
    if reader.is_empty() {
        eprintln!("{} holds no samples", args.data.display());
        std::process::exit(1);
    }
    if reader.len() / reader.sequence_length() < 2 {
        eprintln!("training and validation need at least two independent sequences");
        std::process::exit(1);
    }
    let tile = args.tile.min(layout.lr_width).min(layout.lr_height);
    if tile != args.tile {
        println!(
            "tile clamped to {tile}, the dataset is only {}x{}",
            layout.lr_width, layout.lr_height
        );
    }
    // Re-scoring takes the shape of the graph from the checkpoint, not from the
    // flags, and it has to take all of it: reading the base from the sidecar
    // while leaving the prediction on its default is a configuration that
    // describes no checkpoint at all.
    let stored = args
        .eval_only
        .then(|| checkpoint::load_config(&args.out).ok())
        .flatten()
        .map(|(config, _)| config);
    let (prediction, kernel_radius, demodulate, demodulation_offset) = match &stored {
        Some(config) => (
            config.prediction,
            config.kernel_radius,
            config.demodulate,
            config.demodulation_offset,
        ),
        None => (
            args.prediction,
            args.kernel_radius,
            args.demodulate,
            args.demodulation_offset,
        ),
    };
    let reconstruction_base = match &stored {
        Some(config) => config.reconstruction_base,
        None if args.color_only => ReconstructionBase::Bilinear,
        None => args.reconstruction_base,
    };
    if reconstruction_base == ReconstructionBase::HighResolutionGuided {
        for plane in [
            ommatidia::Plane::Depth,
            ommatidia::Plane::Normal,
            ommatidia::Plane::DiffuseAlbedo,
        ] {
            if !layout.hr_planes.contains(plane) {
                eprintln!("high-resolution guided reconstruction needs the HR {plane:?} plane");
                std::process::exit(1);
            }
        }
    }

    // Everything about the data comes from the data, so the network cannot ask
    // for a plane the dataset does not carry or upscale by the wrong factor.
    let temporal = (args.history_frames > 1).then_some(ommatidia::temporal::Config {
        frames: args.history_frames,
        rejection: ommatidia::temporal::RejectionConfig::default(),
        features: args.temporal_features,
    });
    let mut config = ModelConfig {
        scale: layout.scale,
        tile,
        batch: args.batch,
        // Restricting rather than regenerating keeps an ablation honest: both
        // arms then see the same bytes, the same crops, and the same batch
        // order, and differ only in which channels reach the network.
        cond_planes: if args.color_only {
            ommatidia::PlaneSet::new().with(ommatidia::Plane::Color)
        } else if temporal.is_some() {
            layout.lr_planes.without(ommatidia::Plane::Motion)
        } else {
            layout.lr_planes
        },
        base_channels: args.base_channels,
        level_multipliers: (0..args.levels).map(|i| 1 << i).collect(),
        blocks_per_level: args.blocks_per_level,
        num_groups: args.num_groups,
        residual_gain: 1.0,
        objective: args.objective,
        prediction,
        kernel_radius,
        demodulate,
        demodulation_offset,
        head_kernel: args.head_kernel,
        temporal_weight: args.temporal_weight,
        reconstruction_base,
        temporal,
        ..ModelConfig::default()
    };
    let split = Split::new(reader.len(), reader.sequence_length(), args.val_fraction);
    // The residual is small, and how small depends on the content and the
    // scale factor, so it is measured rather than assumed. Without this the
    // diffusion objective trains to a low loss and samples to pure noise.
    let probe = GAIN_PROBE_SAMPLES.min(reader.len());
    // A kernel checkpoint has no residual, so there is no scale to measure.
    if !args.eval_only && config.prediction != Prediction::SubpixelKernel {
        let samples: Vec<_> = if let Some(temporal) = config.temporal {
            let sequence_length = reader.sequence_length();
            let sequences = split.training().len() / sequence_length;
            (0..probe.min(sequences))
                .filter_map(|i| {
                    let sequence = i * sequences / probe.min(sequences).max(1);
                    let index = sequence * sequence_length + sequence_length - 1;
                    ommatidia::temporal::prepare(&mut reader, index, temporal)
                        .ok()
                        .map(|prepared| prepared.sample)
                })
                .collect()
        } else {
            (0..probe)
                .filter_map(|i| reader.sample(i * reader.len() / probe.max(1)).ok())
                .collect()
        };
        config.residual_gain = ommatidia::batch::estimate_gain(samples, &layout, &config);
        println!(
            "residual gain {:.2} (standard deviation {:.4}), measured over {probe} samples",
            config.residual_gain,
            1.0 / config.residual_gain
        );
    }
    if let Err(message) = config.validate() {
        eprintln!("model configuration is invalid: {message}");
        std::process::exit(1);
    }

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
        "{:?} objective, {} levels, {parameter_count} parameters",
        config.objective,
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
    session.set_adam(args.learning_rate, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);

    // The temporal loss compares this frame against the network's own answer
    // for the previous one. That answer has to come from somewhere the gradient
    // does not flow through, and a reprojection is not something the graph can
    // express, so it comes from a detached copy run on the host side and
    // resynchronised every `--teacher-every` steps. At step zero the copy is
    // initialised identically, so the target is self-consistent from the start
    // rather than arbitrary.
    let mut teacher = config.temporal_weight.ne(&0.0).then(|| {
        let inference = model::build_ending(&config, model::Ending::Image, [config.tile; 2])
            .expect("the training configuration already validated");
        let mut session = ommatidia::gpu::inference_session(&inference.graph, context.clone());
        inference.initialize(&mut session, args.seed);
        session
    });
    let teacher_checkpoint = std::env::temp_dir().join("ommatidia-teacher.safetensors");
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
            let rate = cosine_learning_rate(args.learning_rate, final_rate, step, args.steps);
            // `set_learning_rate` configures SGD in Meganeura. Reconfigure
            // Adam itself so the schedule preserves its moment estimates.
            session.set_adam(rate, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);
        }

        let batch = batcher.next().expect("cannot read a batch");
        session.set_input("cond", &batch.cond);
        session.set_input("target", &batch.target);
        if !batch.taps.is_empty() {
            session.set_input("taps", &batch.taps);
        }
        if let (Some(teacher), Some(temporal)) = (teacher.as_mut(), batch.temporal.as_ref()) {
            if step.is_multiple_of(args.teacher_every)
                && let Err(e) = session
                    .save_checkpoint(&teacher_checkpoint)
                    .and_then(|()| teacher.load_checkpoint(&teacher_checkpoint))
            {
                eprintln!("cannot resynchronise the temporal teacher: {e}");
                std::process::exit(1);
            }
            teacher.set_input("cond", &temporal.cond);
            teacher.set_input("taps", &temporal.taps);
            teacher.step();
            teacher.wait();
            let previous = teacher.read_output(config.loss_len());

            let tile = config.tile as usize;
            let scale = config.scale as usize;
            let per_slot = (config.image_channels() * config.tile * config.tile) as usize;
            let mut target = vec![0.0; config.loss_len()];
            let mut mask = vec![0.0; config.loss_len()];
            for slot in 0..config.batch as usize {
                let span = slot * per_slot..(slot + 1) * per_slot;
                let texels = slot * tile * tile..(slot + 1) * tile * tile;
                let (slot_target, slot_mask) = batch::temporal_target(
                    &previous[span.clone()],
                    &temporal.reference_change[span.clone()],
                    &temporal.motion[texels.start * 2..texels.end * 2],
                    &temporal.valid[texels],
                    tile,
                    scale,
                );
                target[span.clone()].copy_from_slice(&slot_target);
                mask[span].copy_from_slice(&slot_mask);
            }
            session.set_input("temporal_target", &target);
            session.set_input("temporal_mask", &mask);
        }
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

    // The evaluator is a separate inference graph. Without this final copy it
    // would score initialization (or the last periodic evaluation) immediately
    // after correctly saving the trained checkpoint.
    evaluator.sync(&session);
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

struct TemporalFrame {
    network: Vec<f32>,
    base: Vec<f32>,
    reference: Vec<f32>,
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

    /// Reconstruct the held-out samples and report against deterministic bases.
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

        let mut nearest_scores = eval::Scores::default();
        let mut bilinear_scores = eval::Scores::default();
        let mut network_scores = eval::Scores::default();
        let mut guided_scores = eval::Scores::default();
        let mut hr_guided_scores = eval::Scores::default();
        // Detail is only meaningful against the canonical frame's own.
        let mut reference_detail = 0.0f64;
        let has_guides = [
            ommatidia::Plane::Depth,
            ommatidia::Plane::Normal,
            ommatidia::Plane::DiffuseAlbedo,
        ]
        .into_iter()
        .all(|plane| layout.lr_planes.contains(plane));
        let has_hr_guides = [
            ommatidia::Plane::Depth,
            ommatidia::Plane::Normal,
            ommatidia::Plane::DiffuseAlbedo,
        ]
        .into_iter()
        .all(|plane| layout.hr_planes.contains(plane));
        let mut temporal_base_total = 0.0f64;
        let mut temporal_network_total = 0.0f64;
        let mut temporal_counted = 0usize;
        let mut temporal_values = 0usize;
        let mut moving_base_total = 0.0f64;
        let mut moving_network_total = 0.0f64;
        let mut moving_values = 0usize;
        let mut previous_temporal: Vec<Option<TemporalFrame>> =
            (0..crops.len()).map(|_| None).collect();
        let mut counted = 0usize;
        let started = std::time::Instant::now();

        'outer: for index in split.validation() {
            if !batcher.has_history(index) {
                previous_temporal.iter_mut().for_each(|slot| *slot = None);
                continue;
            }
            let input = match batcher.sample(index) {
                Ok(sample) => sample,
                Err(e) => {
                    eprintln!("cannot read validation sample {index}: {e}");
                    break;
                }
            };
            let sample = input.sample();
            for (crop_index, &crop) in crops.iter().enumerate() {
                if counted >= args.eval_crops {
                    break 'outer;
                }
                let guided = has_guides
                    .then(|| batch::guided_base(sample, &layout, crop, self.config.guide));
                let hr_guided = has_hr_guides.then(|| {
                    batch::high_resolution_guided_base(sample, &layout, crop, self.config.guide)
                });
                let model_base = match self.config.reconstruction_base {
                    ReconstructionBase::GuidedBilinear => guided.as_deref(),
                    ReconstructionBase::HighResolutionGuided => hr_guided.as_deref(),
                    ReconstructionBase::Nearest
                    | ReconstructionBase::Bilinear
                    | ReconstructionBase::Sample => None,
                };
                let predicted = eval::reconstruct(
                    &mut self.session,
                    &self.config,
                    schedule,
                    &input,
                    &layout,
                    crop,
                    model_base,
                    args.sampler_steps,
                    // Vary the sampler noise per crop, so the score is not one
                    // lucky or unlucky draw repeated.
                    args.seed.wrapping_add(counted as u64),
                );
                let low = ommatidia::batch::crop_color(sample, &layout, crop);
                let reference = ommatidia::batch::crop_reference(sample, &layout, crop);
                let baseline = eval::nearest(
                    &low,
                    crop.tile as usize,
                    crop.tile as usize,
                    self.config.scale as usize,
                );
                let bilinear = eval::bilinear(
                    &low,
                    crop.tile as usize,
                    crop.tile as usize,
                    self.config.scale as usize,
                );
                let extent = (crop.tile * self.config.scale) as usize;
                nearest_scores.add(&baseline, &reference, extent);
                bilinear_scores.add(&bilinear, &reference, extent);
                network_scores.add(&predicted, &reference, extent);
                reference_detail += ommatidia::metrics::detail(&reference, extent, extent);
                if let Some(guided) = &guided {
                    guided_scores.add(guided, &reference, extent);
                }
                if let Some(hr_guided) = &hr_guided {
                    hr_guided_scores.add(hr_guided, &reference, extent);
                }

                let temporal_base = match self.config.reconstruction_base {
                    ReconstructionBase::Nearest => &baseline,
                    ReconstructionBase::Bilinear => &bilinear,
                    ReconstructionBase::GuidedBilinear => {
                        guided.as_ref().expect("guided model has a guide")
                    }
                    ReconstructionBase::HighResolutionGuided => {
                        hr_guided.as_ref().expect("HR-guided model has an HR guide")
                    }
                    // A kernel checkpoint has no base of its own, so it is held
                    // against the best deterministic reconstruction the dataset
                    // can support. Anything weaker would flatter it.
                    ReconstructionBase::Sample => {
                        hr_guided.as_ref().or(guided.as_ref()).unwrap_or(&bilinear)
                    }
                };
                if let Some(temporal) = self.config.temporal
                    && let Some((motion, valid)) =
                        eval::temporal_evidence(&input, &layout, crop, temporal.frames)
                {
                    if let Some(previous) = &previous_temporal[crop_index] {
                        let extent = crop.tile as usize;
                        let scale = self.config.scale as usize;
                        let network_error = ommatidia::metrics::temporal_error(
                            [&predicted, &previous.network],
                            [&reference, &previous.reference],
                            &motion,
                            &valid,
                            [extent, extent],
                            scale,
                        );
                        let base_error = ommatidia::metrics::temporal_error(
                            [temporal_base, &previous.base],
                            [&reference, &previous.reference],
                            &motion,
                            &valid,
                            [extent, extent],
                            scale,
                        );
                        if let (Some(network_error), Some(base_error)) = (network_error, base_error)
                        {
                            assert_eq!(network_error.values, base_error.values);
                            temporal_network_total += network_error.squared_sum;
                            temporal_base_total += base_error.squared_sum;
                            temporal_values += network_error.values;
                            temporal_counted += 1;
                        }
                        let moving_valid: Vec<_> = valid
                            .iter()
                            .zip(motion.chunks_exact(2))
                            .map(|(&valid, vector)| valid && (vector[0] != 0.0 || vector[1] != 0.0))
                            .collect();
                        let moving_network = ommatidia::metrics::temporal_error(
                            [&predicted, &previous.network],
                            [&reference, &previous.reference],
                            &motion,
                            &moving_valid,
                            [extent, extent],
                            scale,
                        );
                        let moving_base = ommatidia::metrics::temporal_error(
                            [temporal_base, &previous.base],
                            [&reference, &previous.reference],
                            &motion,
                            &moving_valid,
                            [extent, extent],
                            scale,
                        );
                        if let (Some(network_error), Some(base_error)) =
                            (moving_network, moving_base)
                        {
                            assert_eq!(network_error.values, base_error.values);
                            moving_network_total += network_error.squared_sum;
                            moving_base_total += base_error.squared_sum;
                            moving_values += network_error.values;
                        }
                    }
                    previous_temporal[crop_index] = Some(TemporalFrame {
                        network: predicted.clone(),
                        base: temporal_base.clone(),
                        reference: reference.clone(),
                    });
                }

                // The first crop also goes out as images, for eyeballing.
                if counted == 0
                    && let Some(dir) = dir
                {
                    let hr_extent = crop.tile * self.config.scale;
                    for (name, image, width) in [
                        ("input", &low, crop.tile),
                        ("nearest", &baseline, hr_extent),
                        ("bilinear", &bilinear, hr_extent),
                        ("predicted", &predicted, hr_extent),
                        ("reference", &reference, hr_extent),
                    ] {
                        let path = dir.join(format!("{name}.png"));
                        if let Err(e) = eval::write_png(&path, image, width, width) {
                            eprintln!("cannot write {}: {e}", path.display());
                        }
                    }
                    if let Some(guided) = &guided {
                        let path = dir.join("guided.png");
                        if let Err(e) = eval::write_png(&path, guided, hr_extent, hr_extent) {
                            eprintln!("cannot write {}: {e}", path.display());
                        }
                    }
                    if let Some(hr_guided) = &hr_guided {
                        let path = dir.join("hr-guided.png");
                        if let Err(e) = eval::write_png(&path, hr_guided, hr_extent, hr_extent) {
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
        let (base_name, base_error) = match self.config.reconstruction_base {
            ReconstructionBase::Nearest => ("nearest", nearest_scores.mse()),
            ReconstructionBase::Bilinear => ("bilinear", bilinear_scores.mse()),
            ReconstructionBase::GuidedBilinear => {
                assert!(has_guides, "guided models require guide planes");
                ("guided", guided_scores.mse())
            }
            ReconstructionBase::HighResolutionGuided => ("HR guide", hr_guided_scores.mse()),
            ReconstructionBase::Sample if has_hr_guides => ("HR guide", hr_guided_scores.mse()),
            ReconstructionBase::Sample if has_guides => ("guided", guided_scores.mse()),
            ReconstructionBase::Sample => ("bilinear", bilinear_scores.mse()),
        };
        let network_error = network_scores.mse();
        let gain = if network_error > 0.0 {
            10.0 * (base_error / network_error).log10()
        } else {
            f64::INFINITY
        };
        println!(
            "  held-out over {counted} crops in {:.1}s:",
            started.elapsed().as_secs_f32()
        );
        println!("  {}", nearest_scores.line("nearest", reference_detail));
        println!("  {}", bilinear_scores.line("bilinear", reference_detail));
        if has_guides {
            println!("  {}", guided_scores.line("guided", reference_detail));
        }
        if has_hr_guides {
            println!("  {}", hr_guided_scores.line("HR guide", reference_detail));
        }
        println!(
            "  {} ({gain:+.2} dB versus {base_name})",
            network_scores.line("network", reference_detail)
        );
        if temporal_counted != 0 {
            let base = temporal_base_total / temporal_values as f64;
            let network = temporal_network_total / temporal_values as f64;
            let gain = 10.0 * (base / network).log10();
            println!(
                "  temporal over {temporal_counted} reprojected crop pairs:\n  \
                 {base_name:<9} delta MSE {base:.6}\n  \
                 network   delta MSE {network:.6} ({gain:+.2} dB versus {base_name})"
            );
            if moving_values != 0 {
                let moving_base = moving_base_total / moving_values as f64;
                let moving_network = moving_network_total / moving_values as f64;
                let moving_gain = 10.0 * (moving_base / moving_network).log10();
                let coverage = 100.0 * moving_values as f64 / temporal_values as f64;
                println!(
                    "  nonzero-motion pixels ({coverage:.1}% of valid):\n  \
                     {base_name:<9} delta MSE {moving_base:.6}\n  \
                     network   delta MSE {moving_network:.6} \
                     ({moving_gain:+.2} dB versus {base_name})"
                );
            }
        }
        Some(gain as f32)
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    fn parse(words: &[&str]) -> Result<Args, String> {
        parse_from(words.iter().map(|w| w.to_string()))
    }

    #[test]
    fn cosine_rate_reaches_both_endpoints() {
        assert!((cosine_learning_rate(2e-4, 2e-5, 0, 4000) - 2e-4).abs() < 1e-10);
        assert!((cosine_learning_rate(2e-4, 2e-5, 3999, 4000) - 2e-5).abs() < 1e-10);
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
                "--prediction" => vec![
                    vec!["--prediction", "subpixel"],
                    vec!["--prediction", "kernel", "--reconstruction-base", "sample"],
                ],
                "--head-kernel" => vec![vec!["--head-kernel", "1"]],
                "--teacher-every" => vec![vec!["--teacher-every", "100"]],
                "--temporal-weight" => vec![vec![
                    "--temporal-weight",
                    "1",
                    "--history-frames",
                    "4",
                    "--prediction",
                    "kernel",
                    "--reconstruction-base",
                    "sample",
                ]],
                "--demodulation-offset" => vec![vec![
                    "--demodulation-offset",
                    "0.25",
                    "--demodulate",
                    "--prediction",
                    "kernel",
                    "--reconstruction-base",
                    "sample",
                ]],
                "--demodulate" => vec![vec![
                    "--demodulate",
                    "--prediction",
                    "kernel",
                    "--reconstruction-base",
                    "sample",
                ]],
                "--reconstruction-base" => {
                    vec![vec!["--reconstruction-base", "guided"]]
                }
                "--temporal-features" => vec![vec!["--temporal-features", "variance"]],
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
        assert_eq!(args.base_channels, 8);
        assert_eq!(args.levels, 3);
        assert_eq!(args.blocks_per_level, 1);
        assert_eq!(args.batch, 8);
        assert_eq!(
            args.temporal_features,
            ommatidia::temporal::Features::Variance
        );
        assert!(!args.allow_filtered_input);
    }

    #[test]
    fn validation_never_cuts_a_frame_sequence() {
        let split = Split::new(40, 4, 0.15);
        assert_eq!(split.training(), 0..32);
        assert_eq!(split.validation(), 32..40);
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
