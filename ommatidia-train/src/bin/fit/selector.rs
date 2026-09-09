//! Freeze native GPU candidates and observed conditioning; never teacher-force history.
use super::*;
use meganeura::graph::Op;
use ommatidia::{
    dataset,
    model::{InitKind, ParamInit},
    transport::{CANDIDATES, Config, FEATURES, Frame, SCALES, Target, graph, native, oracle},
};

struct Batch {
    config: Config,
    frame: Frame,
    target: Target,
    features: Vec<f32>,
    candidates: oracle::Candidates,
    albedo: Vec<f32>,
    emission: Vec<f32>,
    rgb: Vec<f32>,
}
fn read_input(s: &meganeura::Session, name: &str, len: usize) -> Vec<f32> {
    let b = s
        .plan()
        .input_buffers
        .iter()
        .find(|(n, _)| n == name)
        .expect("native input contract")
        .1;
    let mut v = vec![0.0; len];
    s.read_buffer(b, &mut v);
    v
}
impl Batch {
    fn n(&self) -> usize {
        self.frame.surfaces.len()
    }
    fn point(&self, k: usize, l: usize, i: usize) -> [f32; 3] {
        let n = self.n();
        std::array::from_fn(|c| {
            let j = (3 * l + c) * n + i;
            if k == SCALES {
                self.candidates.history[j]
            } else {
                self.candidates.spatial[k * 6 * n + j]
            }
        })
    }
    fn compose(&self, lobes: &[f32]) -> Vec<f32> {
        let n = self.n();
        (0..3 * n)
            .map(|i| lobes[i] * self.albedo[i] + lobes[3 * n + i] + self.emission[i])
            .collect()
    }
    fn pixels(&self, rgb: &[f32]) -> Vec<f32> {
        let [w, h] = self.frame.low.map(|v| v * self.config.scale);
        let mut out = vec![0.0; rgb.len()];
        for y in 0..h as usize {
            for x in 0..w as usize {
                for c in 0..3 {
                    out[3 * (y * w as usize + x) + c] =
                        rgb[self.config.index(self.frame.low, c, x, y)];
                }
            }
        }
        out
    }
    fn feed(&self, s: &mut meganeura::Session, rgb: bool) {
        for (name, v) in [
            ("f0.features", &self.features),
            ("f0.candidates", &self.candidates.spatial),
            ("f0.prior", &self.candidates.prior),
            ("f0.history", &self.candidates.history),
        ] {
            if s.has_input(name) {
                s.set_input(name, v);
            }
        }
        if rgb {
            s.set_input("fit.target", &self.rgb);
            s.set_input("fit.albedo", &self.albedo);
            s.set_input("fit.emission", &self.emission);
        } else {
            s.set_input("fit.target", &self.target.lobes);
        }
    }
    fn reference(&self, logits: &[f32], rgb: bool) -> (f64, Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = self.n();
        assert_eq!(logits.len(), 2 * n * CANDIDATES);
        let mut weights = vec![0.0; logits.len()];
        let mut lobes = vec![0.0; 6 * n];
        let mut den = vec![0.0f64; 2 * n];
        for l in 0..2 {
            for i in 0..n {
                let a = l * n + i;
                let center = (0..CANDIDATES)
                    .filter(|&k| self.candidates.prior[k * 2 * n + a] > 0.0)
                    .map(|k| logits[k * 2 * n + a] as f64)
                    .fold(f64::NEG_INFINITY, f64::max);
                let multiplier = |z: f64| match self.config.mixture {
                    ommatidia::transport::mixture::Mode::Softplus => (z.max(0.0)
                        + (-z.abs()).exp().ln_1p())
                    .max(ommatidia::transport::MIN_MULTIPLIER as f64),
                    ommatidia::transport::mixture::Mode::MaskedSoftmax => {
                        ((z - center) * ommatidia::transport::mixture::SOFTMAX_GAIN as f64).exp()
                    }
                };
                let total = (0..CANDIDATES)
                    .map(|k| {
                        let z = logits[k * 2 * n + a] as f64;
                        let sp = if self.candidates.prior[k * 2 * n + a] > 0.0 {
                            multiplier(z)
                        } else {
                            0.0
                        };
                        self.candidates.prior[k * 2 * n + a] as f64 * sp
                    })
                    .sum::<f64>();
                den[a] = total;
                for k in 0..CANDIDATES {
                    let j = k * 2 * n + a;
                    let z = logits[j] as f64;
                    let sp = if self.candidates.prior[j] > 0.0 {
                        multiplier(z)
                    } else {
                        0.0
                    };
                    weights[j] = (self.candidates.prior[j] as f64 * sp / total) as f32;
                    for c in 0..3 {
                        lobes[(3 * l + c) * n + i] += weights[j] * self.point(k, l, i)[c];
                    }
                }
            }
        }
        let prediction = if rgb {
            self.compose(&lobes)
        } else {
            lobes.clone()
        };
        let target = if rgb { &self.rgb } else { &self.target.lobes };
        let cost = mse(&prediction, target);
        let mut grad = vec![0.0; logits.len()];
        for l in 0..2 {
            for i in 0..n {
                for k in 0..CANDIDATES {
                    let j = k * 2 * n + l * n + i;
                    let z = logits[j] as f64;
                    let active = z.max(0.0) + (-z.abs()).exp().ln_1p()
                        > ommatidia::transport::MIN_MULTIPLIER as f64;
                    let dp = if self.config.mixture
                        == ommatidia::transport::mixture::Mode::MaskedSoftmax
                    {
                        weights[j] as f64 * ommatidia::transport::mixture::SOFTMAX_GAIN as f64
                    } else if active {
                        self.candidates.prior[j] as f64 / (1.0 + (-z).exp()) / den[l * n + i]
                    } else {
                        0.0
                    };
                    let mut g = 0.0;
                    for c in 0..3 {
                        let idx = if rgb { c * n + i } else { (3 * l + c) * n + i };
                        let material = if rgb && l == 0 {
                            self.albedo[c * n + i]
                        } else {
                            1.0
                        };
                        g += 2.0 * (prediction[idx] as f64 - target[idx] as f64)
                            / prediction.len() as f64
                            * material as f64
                            * (self.point(k, l, i)[c] as f64 - lobes[(3 * l + c) * n + i] as f64);
                    }
                    grad[j] = (g * dp) as f32;
                }
            }
        }
        (cost, lobes, weights, grad)
    }
}

/// Replace only the native selector's final convolution with a diagnostic tensor.
/// Everything after it is the exact native mixing graph, not a duplicate estimator.
fn model(config: Config, low: [u32; 2], free: bool, rgb: bool) -> Result<Network> {
    let mut m = graph::build(config, low, 0)?;
    let n = (low[0] * low[1] * config.scale.pow(2)) as usize;
    let head = m
        .graph
        .nodes()
        .iter()
        .find(|v| matches!(&v.op,Op::Parameter{name} if name=="head.lobe_candidates"))
        .ok_or("native head parameter missing")?
        .id;
    let ids: Vec<_> = m
        .graph
        .nodes()
        .iter()
        .filter(|v| v.inputs.get(1) == Some(&head) && matches!(v.op, Op::Conv2d { .. }))
        .map(|n| n.id)
        .collect();
    if ids.len() != 1 {
        return Err("native head graph changed; update the diagnostic".into());
    }
    let logits = ids[0];
    if free {
        let node = &mut m.graph.nodes_mut()[logits as usize];
        node.op = Op::Parameter {
            name: "fit.logits".into(),
        };
        node.inputs.clear();
        m.params = vec![ParamInit {
            name: "fit.logits".into(),
            len: 2 * n * CANDIDATES,
            kind: InitKind::Zeros,
        }];
    }
    let image = m.graph.outputs()[0];
    let weights = m.graph.outputs()[1];
    let (prediction, channels) = if rgb {
        let a = m.graph.input("fit.albedo", &[3 * n]);
        let e = m.graph.input("fit.emission", &[3 * n]);
        let d = m.graph.split_a(image, 1, 3, 3, n as u32);
        let s = m.graph.split_b(image, 1, 3, 3, n as u32);
        let d = m.graph.mul(d, a);
        let sum = m.graph.add(d, s);
        (m.graph.add(sum, e), 3)
    } else {
        (image, 6)
    };
    let truth = m.graph.input("fit.target", &[channels * n]);
    let loss = m.graph.mse_loss(prediction, truth);
    m.graph.set_outputs(vec![loss, image, weights, logits]);
    Ok(m)
}

/// Fully corrective active hull projection in RGB. The active set has at most
/// four vertices, so the existing exhaustive <=6-vertex solver stays the kernel.
/// A dual gap is returned and checked rather than claiming an exact large hull.
fn project_many(points: &[[f32; 3]], target: [f32; 3]) -> Result<oracle::Projection> {
    if points.is_empty()
        || points
            .iter()
            .flatten()
            .chain(&target)
            .any(|v| !v.is_finite())
    {
        return Err("invalid RGB hull".into());
    }
    let mut active = vec![0usize];
    let mut best = oracle::project(&[points[0]], target)?;
    for _ in 0..256 {
        let gradient = std::array::from_fn::<_, 3, _>(|c| best.point[c] as f64 - target[c] as f64);
        let (index, gap) = (0..points.len())
            .map(|i| {
                (
                    i,
                    (0..3)
                        .map(|c| gradient[c] * (best.point[c] as f64 - points[i][c] as f64))
                        .sum::<f64>(),
                )
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        best.dual_gap = gap.max(0.0);
        if best.dual_gap <= 1e-7 * (1.0 + best.squared_error) || active.contains(&index) {
            break;
        }
        let keep: Vec<_> = active
            .iter()
            .zip(&best.weights)
            .filter(|(_, w)| **w > 0.0)
            .map(|(i, _)| *i)
            .collect();
        active = keep;
        active.push(index);
        best = oracle::project(
            &active.iter().map(|i| points[*i]).collect::<Vec<_>>(),
            target,
        )?;
    }
    let gradient = std::array::from_fn::<_, 3, _>(|c| best.point[c] as f64 - target[c] as f64);
    best.dual_gap = points
        .iter()
        .map(|p| {
            (0..3)
                .map(|c| gradient[c] * (best.point[c] as f64 - p[c] as f64))
                .sum::<f64>()
        })
        .fold(0.0, f64::max);
    if best.dual_gap > 2e-5 * (1.0 + best.squared_error) {
        return Err(format!("RGB hull failed optimality certificate: {}", best.dual_gap).into());
    }
    let mut weights = vec![0.0; points.len()];
    for (i, w) in active.into_iter().zip(best.weights) {
        weights[i] = w;
    }
    best.weights = weights;
    Ok(best)
}
fn optimum(b: &Batch, rgb: bool) -> Result<(Vec<f32>, f64, f64)> {
    if !rgb {
        let p = oracle::reconstruct(&b.candidates, &b.target.lobes)?;
        let cost = mse(&p.lobes, &b.target.lobes);
        return Ok((p.lobes, cost, p.max_dual_gap));
    }
    let n = b.n();
    let mut lobes = vec![0.0; 6 * n];
    let mut gap = 0.0f64;
    for i in 0..n {
        let legal = |l| {
            (0..CANDIDATES)
                .filter(|k| b.candidates.prior[k * 2 * n + l * n + i] > 0.0)
                .collect::<Vec<_>>()
        };
        let pairs: Vec<_> = legal(0)
            .into_iter()
            .flat_map(|d| legal(1).into_iter().map(move |s| (d, s)))
            .collect();
        let points: Vec<_> = pairs
            .iter()
            .map(|(d, s)| {
                std::array::from_fn(|c| {
                    b.point(*d, 0, i)[c] * b.albedo[c * n + i] + b.point(*s, 1, i)[c]
                })
            })
            .collect();
        let target = std::array::from_fn(|c| b.rgb[c * n + i] - b.emission[c * n + i]);
        let p = project_many(&points, target)?;
        gap = gap.max(p.dual_gap);
        for ((d, s), w) in pairs.iter().zip(p.weights) {
            for c in 0..3 {
                lobes[c * n + i] += w * b.point(*d, 0, i)[c];
                lobes[(3 + c) * n + i] += w * b.point(*s, 1, i)[c];
            }
        }
    }
    let cost = mse(&b.compose(&lobes), &b.rgb);
    Ok((lobes, cost, gap))
}
fn checks(b: &Batch, ctx: Arc<blade_graphics::Context>, rgb: bool) -> Result<Value> {
    let m = model(b.config, b.frame.low, true, rgb)?;
    let mut s = gpu::training_session(&m.graph, ctx);
    initialize(&m, &mut s, 7);
    b.feed(&mut s, rgb);
    let z: Vec<_> = (0..2 * b.n() * CANDIDATES)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.03)
        .collect();
    s.set_parameter("fit.logits", &z);
    s.set_adam(0.0, 0.9, 0.999, 1e-8);
    s.step();
    s.wait();
    let (loss, image, weights, grad) = b.reference(&z, rgb);
    let mut actual = vec![0.0; z.len()];
    s.read_param_grad("fit.logits", &mut actual);
    let error = mse(&actual, &grad).sqrt();
    let scale =
        grad.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt() / (grad.len() as f64).sqrt();
    let discrepancy = error / scale.max(1e-10);
    if discrepancy > 0.005
        || (loss - s.read_loss() as f64).abs() > 2e-5 * (1.0 + loss)
        || mse(&output(&s, 1, image.len()), &image) > 1e-8
        || mse(&output(&s, 2, weights.len()), &weights) > 1e-10
    {
        return Err(format!(
            "native/free CPU or analytic gradient disagreement: grad {discrepancy}, loss {loss}/{}",
            s.read_loss()
        )
        .into());
    }
    Ok(
        json!({"analytic_gradient_relative_l2":discrepancy,"cpu_loss":loss,"gpu_loss":s.read_loss()}),
    )
}
pub(super) fn run(o: &Options, ctx: Arc<blade_graphics::Context>) -> Result<Value> {
    let config = Config {
        version: o.mixture.version(),
        mixture: o.mixture,
        ..Config::default()
    };
    let mut reader = dataset::Reader::open(&o.data)?;
    let layout = *reader.layout();
    if o.frame >= reader.len() {
        return Err("frame outside capture".into());
    }
    let provenance: Value =
        serde_json::from_slice(&std::fs::read(o.data.with_extension("transport.json"))?)?;
    if provenance["matching_path_depth"] != true
        || provenance["input_estimator"] != "independent-paths"
    {
        return Err("need matched independent path capture".into());
    }
    let mut runtime = native::Native::new(
        Arc::clone(&ctx),
        Config {
            version: 1,
            mixture: Default::default(),
            ..config
        },
        [layout.lr_width, layout.lr_height],
    )?;
    let start = if reader.sequence_length() > 0 {
        o.frame / reader.sequence_length() * reader.sequence_length()
    } else {
        o.frame
    };
    let mut native_rgb = Vec::new();
    for index in start..=o.frame {
        let f = Frame::from_sample(&reader.sample(index)?, layout, config)?;
        native_rgb = runtime.process(&f)?;
    }
    let sample = reader.sample(o.frame)?;
    let frame = Frame::from_sample(&sample, layout, config)?;
    let target = Target::from_sample(&sample, layout, config)?;
    let n = frame.surfaces.len();
    let mut b = Batch {
        config,
        frame,
        target,
        features: read_input(&runtime.session, "f0.features", FEATURES * n),
        candidates: runtime.read_candidates(),
        albedo: vec![0.0; 3 * n],
        emission: vec![0.0; 3 * n],
        rgb: vec![0.0; 3 * n],
    };
    let w = (b.frame.low[0] * config.scale) as usize;
    for (p, s) in b.frame.surfaces.iter().enumerate() {
        for c in 0..3 {
            let j = config.index(b.frame.low, c, p % w, p / w);
            b.albedo[j] = s.albedo_roughness[c];
            b.emission[j] = s.emission[c];
            b.rgb[j] = b.target.rgb[3 * p + c];
        }
    }
    let rgb = o.task == "selector-rgb";
    let analytic = checks(&b, Arc::clone(&ctx), rgb)?;
    let (_, initial, _, _) = b.reference(&vec![0.0; 2 * n * CANDIDATES], rgb);
    let parity = mse(&b.pixels(&b.compose(&initial)), &native_rgb);
    if parity > 1e-8 {
        return Err(format!("zero-head candidate replay differs from native: {parity}").into());
    }
    let (optimal, bound, gap) = optimum(&b, rgb)?;
    let extent = b.frame.low.map(|v| v * config.scale);
    png(&o.out.join("reference.png"), &b.target.rgb, extent)?;
    png(&o.out.join("prior.png"), &native_rgb, extent)?;
    png(
        &o.out.join("conditional-optimum.png"),
        &b.pixels(&b.compose(&optimal)),
        extent,
    )?;
    // This file is explicitly fitting data, not an inference observation export.
    write(
        o.out.join("frozen-batch.json"),
        &json!({"role":"privileged fitting diagnostic","low":b.frame.low,"features":b.features,"spatial":b.candidates.spatial,"history":b.candidates.history,"prior":b.candidates.prior,"target_lobes":b.target.lobes,"albedo":b.albedo,"emission":b.emission,"target_rgb":b.rgb}),
    )?;
    let mut arms = Vec::new();
    for free in [true, false] {
        let name = if free { "free-logits" } else { "network" };
        let m = model(config, b.frame.low, free, rgb)?;
        let (inf, mut report) = optimize(
            &m,
            Arc::clone(&ctx),
            |s| b.feed(s, rgb),
            &o.out.join(name),
            o.seed,
            o.steps,
            if free { 0.03 } else { 0.001 },
        )?;
        let names = if free {
            vec!["fit.logits"]
        } else {
            vec!["head.lobe_candidates", "core.up0", "adapter.observation"]
        };
        let grad = gradients(&m, &inf, Arc::clone(&ctx), |s| b.feed(s, rgb), &names)?;
        let image = output(&inf, 1, 6 * n);
        let weights = output(&inf, 2, 2 * n * CANDIDATES);
        let z = output(&inf, 3, 2 * n * CANDIDATES);
        let mass: Vec<f64> = (0..2 * n)
            .map(|i| (0..CANDIDATES).map(|k| weights[k * 2 * n + i] as f64).sum())
            .collect();
        let min_mass = mass.iter().copied().fold(f64::INFINITY, f64::min);
        let max_mass = mass.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if mass
            .iter()
            .any(|v| !v.is_finite() || (v - 1.0).abs() > 2e-6)
        {
            return Err(
                format!("selector left the candidate hull: mass {min_mass}..{max_mass}").into(),
            );
        }
        report["normalization"] = json!({"minimum":min_mass,"maximum":max_mass});
        write(
            o.out.join(name).join("prediction.json"),
            &json!({"logits":z,"weights":weights,"lobes":image}),
        )?;
        let actual = b.pixels(&b.compose(&image));
        png(&o.out.join(format!("{name}.png")), &actual, extent)?;
        report["name"] = json!(name);
        report["gradients"] = grad;
        report["rgb"] = image_score(&actual, &b.target.rgb);
        report["lobe_mse"] = json!(mse(&image, &b.target.lobes));
        report["excess_over_conditional_bound"] =
            json!(report["final_loss"].as_f64().unwrap() - bound);
        report["logit_min"] = json!(z.iter().copied().fold(f32::INFINITY, f32::min));
        report["logit_max"] = json!(z.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        report["mean_weight_by_candidate"] = json!(
            (0..CANDIDATES)
                .map(|k| weights[k * 2 * n..(k + 1) * 2 * n]
                    .iter()
                    .map(|v| *v as f64)
                    .sum::<f64>()
                    / (2 * n) as f64)
                .collect::<Vec<_>>()
        );
        arms.push(report);
    }
    Ok(
        json!({"mixture":config.mixture,"config":config,"objective":if rgb{"linear remodulated RGB"}else{"linear illumination lobes"},"history":"frozen native zero-head recurrence; not updated during fitting","analytic_check":analytic,"native_replay_mse":parity,"conditional_bound":bound,"max_dual_gap":gap,"arms":arms}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rgb_product_hull_has_an_optimality_certificate() {
        let pts: Vec<_> = (0..6)
            .flat_map(|a| (0..6).map(move |b| [a as f32, b as f32, (a + b) as f32]))
            .collect();
        let p = project_many(&pts, [2.0, 3.0, 5.0]).unwrap();
        assert!(p.squared_error < 1e-10 && p.dual_gap < 1e-6);
        assert!((p.weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let p = project_many(&pts, [-1.0, -1.0, -2.0]).unwrap();
        assert_eq!(p.point, [0.0; 3]);
    }
    #[test]
    fn diagnostic_does_not_change_network_parameter_schema() {
        let c = Config::default();
        let original = graph::build(c, [8, 8], 0).unwrap();
        let probe = model(c, [8, 8], false, true).unwrap();
        assert_eq!(
            original
                .params
                .iter()
                .map(|p| (&p.name, p.len))
                .collect::<Vec<_>>(),
            probe
                .params
                .iter()
                .map(|p| (&p.name, p.len))
                .collect::<Vec<_>>()
        );
        let free = model(c, [8, 8], true, true).unwrap();
        assert_eq!(free.params.len(), 1);
        assert_eq!(free.params[0].name, "fit.logits");
    }
}
