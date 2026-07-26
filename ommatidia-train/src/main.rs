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
use ommatidia::model::{self, ModelConfig, Objective};

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
    base_channels: u32,
    levels: usize,
    blocks_per_level: usize,
    timesteps: usize,
    sampler_steps: usize,
    objective: Objective,
    seed: u64,
    log_every: usize,
    eval_out: Option<PathBuf>,
    color_only: bool,
    val_fraction: f32,
    eval_crops: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            data: PathBuf::from("data/train.omd"),
            out: PathBuf::from("runs/ommatidia"),
            steps: 1000,
            batch: 4,
            tile: 64,
            learning_rate: 2e-4,
            base_channels: 64,
            levels: 3,
            blocks_per_level: 2,
            timesteps: 1000,
            sampler_steps: 20,
            objective: Objective::Diffusion,
            seed: 0,
            log_every: 50,
            eval_out: None,
            color_only: false,
            val_fraction: 0.15,
            eval_crops: 64,
        }
    }
}

const USAGE: &str = "\
train the ommatidia reconstruction network

usage: ommatidia-train [options]

  --data PATH          dataset to train on  [data/train.omd]
  --out STEM           checkpoint stem, gets .safetensors and .ron  [runs/ommatidia]
  --steps N            optimizer steps  [1000]
  --batch N            crops per step  [4]
  --tile N             square crop size, in input pixels  [64]
  --lr F               Adam learning rate  [2e-4]
  --base-channels N    channel width of the first level  [64]
  --levels N           U-Net levels  [3]
  --blocks N           residual blocks per level  [2]
  --timesteps N        diffusion schedule length  [1000]
  --sampler-steps N    DDIM steps used when evaluating  [20]
  --objective KIND     diffusion or direct  [diffusion]
  --seed N             seed for init and batching  [0]
  --log-every N        steps between loss lines  [50]
  --eval-out DIR       write comparison PNGs of the first held-out crop
  --val-fraction F     share of the set held out for scoring  [0.15]
  --eval-crops N       cap on held-out crops scored  [64]
  --color-only         condition on colour alone, ignoring the dataset's
                       G-buffer planes; the other half of that ablation is
                       simply leaving this off
  -h, --help           this message
";

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut argv = std::env::args().skip(1);
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
            "--base-channels" => {
                args.base_channels = value()?
                    .parse()
                    .map_err(|e| format!("--base-channels: {e}"))?
            }
            "--levels" => args.levels = value()?.parse().map_err(|e| format!("--levels: {e}"))?,
            "--blocks" => {
                args.blocks_per_level = value()?.parse().map_err(|e| format!("--blocks: {e}"))?
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
            "--log-every" => {
                args.log_every = value()?.parse().map_err(|e| format!("--log-every: {e}"))?
            }
            "--eval-out" => args.eval_out = Some(PathBuf::from(value()?)),
            "--color-only" => args.color_only = true,
            "--val-fraction" => {
                args.val_fraction = value()?
                    .parse()
                    .map_err(|e| format!("--val-fraction: {e}"))?
            }
            "--eval-crops" => {
                args.eval_crops = value()?.parse().map_err(|e| format!("--eval-crops: {e}"))?
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

    print!("compiling... ");
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    let started = std::time::Instant::now();
    let mut session = meganeura::build_session(&model.graph);
    println!(
        "{:.1}s, {} dispatches on {}",
        started.elapsed().as_secs_f32(),
        session.plan().dispatches.len(),
        session.device_information().device_name,
    );
    model.initialize(&mut session, args.seed);
    session.set_adam(args.learning_rate, 0.9, 0.999, 1e-8);

    let mut batcher = batcher::Batcher::new(
        reader,
        config.clone(),
        schedule.clone(),
        split.training(),
        args.seed,
    );
    let diffusing = config.objective == Objective::Diffusion;

    let mut recent = Vec::new();
    let mut first_average = f32::NAN;
    let mut last_average = f32::NAN;
    let training_started = std::time::Instant::now();

    for step in 0..args.steps {
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
            println!(
                "step {:>6}: loss {average:.6}  ({rate:.1} steps/s)",
                step + 1
            );
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
    evaluate(
        &args,
        &config,
        &schedule,
        &mut batcher,
        split,
        args.eval_out.as_deref(),
    );
}

/// Reconstruct the held-out samples and report how the network compares to
/// nearest upsampling.
///
/// Over a grid of crops across every validation sample, not one crop of one
/// training sample: a single tile is far too small and too lucky to separate
/// two configurations, and scoring on data the network was fitted to measures
/// memorisation rather than reconstruction.
///
/// A separate inference session, because the training graph ends at the loss
/// and has the training batch size baked in, while sampling needs the
/// prediction and runs one crop at a time. Parameters do not depend on the
/// batch, so the checkpoint loads straight into it.
fn evaluate(
    args: &Args,
    config: &ModelConfig,
    schedule: &Schedule,
    batcher: &mut batcher::Batcher,
    split: Split,
    dir: Option<&std::path::Path>,
) {
    let mut eval_config = config.clone();
    eval_config.batch = 1;
    let model = model::build(&eval_config, false).expect("the training config already validated");

    print!("evaluating: compiling inference graph... ");
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    let mut session = meganeura::build_inference_session(&model.graph);
    let paths = checkpoint::Paths::from_stem(&args.out);
    if let Err(e) = session.load_checkpoint(&paths.weights) {
        eprintln!("\ncannot load {}: {e}", paths.weights.display());
        return;
    }
    println!("done");

    let layout = *batcher.layout();
    // Non-overlapping tiles, so no pixel is counted twice.
    let crops = Crop::grid(&layout, eval_config.tile, eval_config.tile);
    if crops.is_empty() {
        eprintln!("the tile is larger than a sample, nothing to evaluate");
        return;
    }

    let mut baseline_total = 0.0f64;
    let mut network_total = 0.0f64;
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
                &mut session,
                &eval_config,
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
                eval_config.scale as usize,
            );

            baseline_total += eval::error(&baseline, &reference) as f64;
            network_total += eval::error(&predicted, &reference) as f64;

            // The first crop also goes out as images, for eyeballing.
            if counted == 0
                && let Some(dir) = dir
            {
                let hr_extent = crop.tile * eval_config.scale;
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
                println!("wrote comparison images to {}", dir.display());
            }
            counted += 1;
        }
    }

    if counted == 0 {
        eprintln!("no validation crops were evaluated");
        return;
    }
    let baseline_error = baseline_total / counted as f64;
    let network_error = network_total / counted as f64;
    println!(
        "held-out reconstruction over {counted} crops from {} samples ({:.1}s):",
        split.validation().len(),
        started.elapsed().as_secs_f32()
    );
    println!("  nearest {baseline_error:.6}, network {network_error:.6}");
    if network_error < baseline_error {
        let gain = 10.0 * (baseline_error / network_error).log10();
        println!("  the network beats nearest upsampling by {gain:.2} dB");
    } else {
        println!(
            "  the network is not beating nearest upsampling — expected early \
             in training, since the head starts at zero"
        );
    }
}
