//! Frozen-batch diagnostics, not a trainer or checkpoint-promotion path.
//! No new runtime architecture: append isolated losses to the existing graphs.
use ommatidia::{gpu, transport::graph::Network};
use serde_json::{Value, json};
use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};
#[path = "fit/field.rs"]
mod field;
#[path = "fit/selector.rs"]
mod selector;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Options {
    task: String,
    data: PathBuf,
    out: PathBuf,
    steps: usize,
    seed: u64,
    frame: usize,
}
fn parse() -> Result<Option<Options>> {
    let mut o = Options {
        task: String::new(),
        data: PathBuf::new(),
        out: PathBuf::new(),
        steps: 512,
        seed: 7,
        frame: 0,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!(
                "fit --task selector-lobes|selector-rgb|source-depth|volume-depth|surface-appearance --data CAPTURE.omd --out DIR [--steps 512 --seed 7 --frame 0]\nFrozen fitting observations only; surface-appearance deliberately uses true positions. Diagnostic weights are NOT trained deployment models."
            );
            return Ok(None);
        }
        let v = args.next().ok_or(format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--task" => o.task = v,
            "--data" => o.data = v.into(),
            "--out" => o.out = v.into(),
            "--steps" => o.steps = v.parse()?,
            "--seed" => o.seed = v.parse()?,
            "--frame" => o.frame = v.parse()?,
            _ => return Err(format!("unknown flag: {flag}").into()),
        }
    }
    if ![
        "selector-lobes",
        "selector-rgb",
        "source-depth",
        "volume-depth",
        "surface-appearance",
    ]
    .contains(&o.task.as_str())
        || !o.data.is_file()
        || o.out.as_os_str().is_empty()
        || o.steps == 0
        || o.steps > 16384
    {
        return Err(
            "require a supported task, existing capture, fresh output directory and 1..16384 steps"
                .into(),
        );
    }
    if o.out.exists() {
        return Err("refusing to overwrite a diagnostic run".into());
    }
    Ok(Some(o))
}
fn write(path: impl AsRef<Path>, value: &Value) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    writeln!(f)?;
    Ok(())
}
fn output(s: &meganeura::Session, index: usize, len: usize) -> Vec<f32> {
    let mut v = vec![0.0; len];
    s.read_output_by_index(index, &mut v);
    v
}
fn copy(a: &meganeura::Session, b: &mut meganeura::Session, m: &Network) {
    for p in &m.params {
        if a.has_parameter(&p.name) && b.has_parameter(&p.name) {
            let mut v = vec![0.0; p.len];
            a.read_param(&p.name, &mut v);
            b.set_parameter(&p.name, &v);
        }
    }
}
fn initialize(m: &Network, s: &mut meganeura::Session, seed: u64) {
    // Preserve random draws even when an isolated branch was pruned.
    let mut rng = ommatidia::rng::Rng::new(seed);
    for p in &m.params {
        use ommatidia::model::InitKind;
        let v = match &p.kind {
            InitKind::Zeros => vec![0.0; p.len],
            InitKind::Ones => vec![1.0; p.len],
            InitKind::Values(v) => v.clone(),
            InitKind::Kaiming { fan_in } => (0..p.len)
                .map(|_| rng.normal() * (2.0 / *fan_in as f32).sqrt())
                .collect(),
        };
        if s.has_parameter(&p.name) {
            s.set_parameter(&p.name, &v);
        }
    }
}
fn mse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64
}
fn image_score(a: &[f32], b: &[f32]) -> Value {
    json!({"linear_mse":mse(a,b),"compressed_psnr":-10.0*(ommatidia::metrics::error(a,b) as f64).max(1e-20).log10(),"energy_ratio":a.iter().map(|v|*v as f64).sum::<f64>() / b.iter().map(|v|*v as f64).sum::<f64>().max(1e-20)})
}
fn png(path: &Path, rgb: &[f32], extent: [u32; 2]) -> Result<()> {
    if rgb.len() != 3 * (extent[0] * extent[1]) as usize || rgb.iter().any(|v| !v.is_finite()) {
        return Err("invalid diagnostic RGB".into());
    }
    let bytes: Vec<u8> = rgb
        .iter()
        .map(|v| {
            let t = v.max(0.0) / (1.0 + v.max(0.0));
            let srgb = if t <= 0.0031308 {
                12.92 * t
            } else {
                1.055 * t.powf(1.0 / 2.4) - 0.055
            };
            (255.0 * srgb.clamp(0.0, 1.0)).round() as u8
        })
        .collect();
    let mut e = png::Encoder::new(std::fs::File::create(path)?, extent[0], extent[1]);
    e.set_color(png::ColorType::Rgb);
    e.set_depth(png::BitDepth::Eight);
    e.write_header()?.write_image_data(&bytes)?;
    Ok(())
}

/// Probe a separate training session so finite differences cannot alter optimizer state.
/// Directional perturbations use normalized gradient directions (no cancellation),
/// with three declared step sizes and both forward losses retained in the report.
fn gradients(
    m: &Network,
    source: &meganeura::Session,
    ctx: Arc<blade_graphics::Context>,
    feed: impl Fn(&mut meganeura::Session),
    names: &[&str],
) -> Result<Value> {
    let mut train = gpu::training_session(&m.graph, Arc::clone(&ctx));
    copy(source, &mut train, m);
    feed(&mut train);
    train.set_adam(0.0, 0.9, 0.999, 1e-8);
    train.step();
    train.wait();
    let mut forward = gpu::inference_session(&m.graph, ctx);
    copy(source, &mut forward, m);
    feed(&mut forward);
    forward.step();
    forward.wait();
    let forward_loss = forward.read_loss();
    let train_loss = train.read_loss();
    if !train_loss.is_finite()
        || (train_loss - forward_loss).abs() > 2e-4 * (1.0 + train_loss.abs())
    {
        return Err(
            format!("training/inference loss mismatch: {train_loss}/{forward_loss}").into(),
        );
    }
    let mut rows = Vec::new();
    for name in names {
        let p = m
            .params
            .iter()
            .find(|p| p.name == *name)
            .ok_or(format!("unknown gradient probe {name}"))?;
        if !train.has_param_grad(name) {
            rows.push(json!({"parameter":name,"connected":false}));
            continue;
        }
        let mut g = vec![0.0; p.len];
        train.read_param_grad(name, &mut g);
        if g.iter().any(|v| !v.is_finite()) {
            return Err(format!("nonfinite gradient: {name}").into());
        }
        let norm = g.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let mut values = vec![0.0; p.len];
        source.read_param(name, &mut values);
        let mut differences = Vec::new();
        if norm > 1e-9 {
            let direction: Vec<_> = g.iter().map(|v| (*v as f64 / norm) as f32).collect();
            for eps in [0.01f32, 0.003, 0.001] {
                let mut loss = [0.0f32; 2];
                for (side, sign) in [-1.0f32, 1.0].into_iter().enumerate() {
                    let perturbed: Vec<_> = values
                        .iter()
                        .zip(&direction)
                        .map(|(v, d)| v + sign * eps * d)
                        .collect();
                    forward.set_parameter(name, &perturbed);
                    forward.step();
                    forward.wait();
                    loss[side] = forward.read_loss();
                }
                let fd = (loss[1] as f64 - loss[0] as f64) / (2.0 * eps as f64);
                differences.push(json!({"epsilon":eps,"minus":loss[0],"plus":loss[1],"finite_difference":fd,"autodiff":norm,"relative_error":(fd-norm).abs()/norm.max(1e-8)}));
            }
            forward.set_parameter(name, &values);
        }
        rows.push(
            json!({"parameter":name,"connected":true,"norm":norm,"finite_differences":differences}),
        );
    }
    Ok(json!({"forward_loss":forward_loss,"training_loss":train_loss,"parameters":rows}))
}

/// Save/reload before final scoring. Each curve value is the pre-update forward
/// loss; the final returned inference session uses the last updated checkpoint.
fn optimize(
    m: &Network,
    ctx: Arc<blade_graphics::Context>,
    feed: impl Fn(&mut meganeura::Session),
    dir: &Path,
    seed: u64,
    steps: usize,
    rate: f32,
) -> Result<(meganeura::Session, Value)> {
    std::fs::create_dir_all(dir)?;
    let mut session = gpu::training_session(&m.graph, Arc::clone(&ctx));
    initialize(m, &mut session, seed);
    feed(&mut session);
    let mut curve = std::fs::File::create(dir.join("loss.csv"))?;
    writeln!(curve, "step,loss")?;
    let mut first = 0.0;
    for step in 0..steps {
        session.set_adam(rate, 0.9, 0.999, 1e-8);
        session.step();
        session.wait();
        let loss = session.read_loss();
        if !loss.is_finite() {
            return Err(format!("nonfinite loss at {step}").into());
        }
        if step == 0 {
            first = loss;
        }
        writeln!(curve, "{step},{loss:.10e}")?;
        if step % 128 == 0 || step + 1 == steps {
            println!("{} {step}/{steps} loss {loss:.8}", dir.display());
        }
    }
    session.save_checkpoint(&dir.join("diagnostic.safetensors"))?;
    let mut inf = gpu::inference_session(&m.graph, ctx);
    inf.load_checkpoint(&dir.join("diagnostic.safetensors"))?;
    feed(&mut inf);
    inf.step();
    inf.wait();
    let final_loss = inf.read_loss();
    if !final_loss.is_finite() {
        return Err("nonfinite reloaded objective".into());
    }
    Ok((
        inf,
        json!({"initial_loss":first,"final_loss":final_loss,"steps":steps,"learning_rate":rate,"seed":seed,"checkpoint_role":"diagnostic-only; no promotion"}),
    ))
}
fn main() -> Result<()> {
    env_logger::init();
    let Some(o) = parse()? else {
        return Ok(());
    };
    std::fs::create_dir_all(&o.out)?;
    let context = gpu::create_context(None, false);
    let device = context.device_information().device_name.clone();
    let result = if o.task.starts_with("selector-") {
        selector::run(&o, context)?
    } else {
        field::run(&o, context)?
    };
    write(
        o.out.join("quality.json"),
        &json!({"role":"frozen-batch construction diagnostic, not validation or deployment","task":o.task,"data":o.data,"frame":o.frame,"device":device,"result":result}),
    )?;
    Ok(())
}
