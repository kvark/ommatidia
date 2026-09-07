//! Fixed-budget rendering diagnostics. Labels score predictions; never prepare them.
use super::*;

pub struct Image {
    pub rgb: Vec<f32>,
    pub termination: Vec<Vec<f32>>, // pixel-major, with final escape mass
}
pub fn render(
    session: &mut meganeura::Session,
    c: &Config,
    shape: RenderShape,
    obs: &Observations,
    view: &View,
) -> Result<Image> {
    let [w, h] = c.extent;
    let n = (w * h) as usize;
    let mut result = Image {
        rgb: Vec::new(),
        termination: Vec::new(),
    };
    for start in (0..n).step_by(shape.rays) {
        let rays: Vec<_> = (0..shape.rays)
            .map(|i| {
                let p = (start + i).min(n - 1);
                view.camera
                    .ray([(p % w as usize) as f32, (p / w as usize) as f32], c.extent)
            })
            .collect();
        let (mut queries, deltas) = data::ray_queries(obs.bounds, &rays, shape.steps)?;
        queries.extend((0..shape.probes).map(|_| field::Query {
            position: obs.bounds.center,
            direction: [0.0, 0.0, 1.0],
        }));
        Prepared::new(obs, c, &queries)?.feed(session);
        session.set_input("ray.deltas", &deltas);
        session.step();
        session.wait();
        let rgb = session.read_output(shape.rays * 3);
        let mut mass = vec![0.0; (shape.steps + 1) * shape.rays];
        session.read_output_by_index(1, &mut mass);
        if rgb
            .iter()
            .chain(&mass)
            .any(|v| !v.is_finite() || *v < -1e-6)
        {
            return Err("invalid field colour or termination mass".into());
        }
        let len = (n - start).min(shape.rays);
        result.rgb.extend_from_slice(&rgb[..3 * len]);
        for r in 0..len {
            let weights: Vec<_> = (0..=shape.steps)
                .map(|s| mass[s * shape.rays + r])
                .collect();
            if (weights.iter().sum::<f32>() - 1.0).abs() > 1e-4 {
                return Err("termination masses do not sum to one".into());
            }
            result.termination.push(weights);
        }
    }
    Ok(result)
}

pub fn geometry(
    image: &Image,
    obs: &Observations,
    view: &TargetView,
    c: &Config,
    steps: usize,
) -> serde_json::Value {
    let Some(labels) = &view.surface else {
        return serde_json::Value::Null;
    };
    let mut valid = 0usize;
    let mut hits = 0usize;
    let mut sources = 0usize;
    let mut nll = 0.0f64;
    let mut correct = 0usize;
    let mut error = 0.0f64;
    let mut early = 0.0f64;
    let mut source_mass = 0.0f64;
    for (p, weights) in image.termination.iter().enumerate() {
        let ray = view.camera.ray(
            [
                (p % c.extent[0] as usize) as f32,
                (p / c.extent[0] as usize) as f32,
            ],
            c.extent,
        );
        let Some(bin) =
            surface::class(obs.bounds, ray, labels.distance[p], labels.ray_limit, steps)
        else {
            continue;
        };
        valid += 1;
        nll -= (weights[bin] as f64 + 1e-8).ln();
        let opacity = 1.0 - weights[steps];
        correct += usize::from((opacity >= 0.5) == labels.distance[p].is_some());
        if let Some(depth) = labels.distance[p] {
            hits += 1;
            let (near, far) = data::ray_interval(obs.bounds, ray);
            let expected = weights[..steps]
                .iter()
                .enumerate()
                .map(|(s, w)| w * (near + (s as f32 + 0.5) * (far - near) / steps as f32))
                .sum::<f32>()
                / opacity.max(1e-8);
            error += (expected - depth).abs() as f64;
            early += weights[..bin].iter().sum::<f32>() as f64;
            if labels.emission[p].iter().any(|v| *v > 0.0) {
                sources += 1;
                source_mass += weights[bin] as f64;
            }
        }
    }
    serde_json::json!({"valid_rays":valid,"excluded_rays":image.termination.len()-valid,
      "termination_nll":nll/valid.max(1) as f64,"hit_miss_accuracy":correct as f64/valid.max(1) as f64,
      "hit_depth_mae_world":(hits>0).then_some(error/hits.max(1) as f64),"hit_rays":hits,
      "early_termination_mass":(hits>0).then_some(early/hits.max(1) as f64),
      "emitter_rays":sources,"emitter_surface_mass":(sources>0).then_some(source_mass/sources.max(1) as f64)})
}

/// Same radiometric transform in both maps; depth divided by declared radius.
pub fn save_geometry(
    path: &Path,
    image: &Image,
    obs: &Observations,
    view: &TargetView,
    c: &Config,
    steps: usize,
) -> Result<()> {
    let Some(labels) = &view.surface else {
        return Ok(());
    };
    let mut predicted = Vec::new();
    let mut reference = Vec::new();
    let mut opacity = Vec::new();
    for (p, w) in image.termination.iter().enumerate() {
        let ray = view.camera.ray(
            [
                (p % c.extent[0] as usize) as f32,
                (p / c.extent[0] as usize) as f32,
            ],
            c.extent,
        );
        let (near, far) = data::ray_interval(obs.bounds, ray);
        let a = 1.0 - w[steps];
        let depth = w[..steps]
            .iter()
            .enumerate()
            .map(|(s, w)| w * (near + (s as f32 + 0.5) * (far - near) / steps as f32))
            .sum::<f32>()
            / a.max(1e-8);
        predicted.extend([depth / obs.bounds.radius; 3]);
        reference.extend([labels.distance[p].unwrap_or(0.0) / obs.bounds.radius; 3]);
        opacity.extend([a; 3]);
    }
    super::png(&path.with_extension("depth.png"), &predicted, c.extent)?;
    super::png(
        &path.with_extension("depth-reference.png"),
        &reference,
        c.extent,
    )?;
    super::png(&path.with_extension("opacity.png"), &opacity, c.extent)
}

pub fn ablated(obs: &Observations, zero: bool) -> Observations {
    let mut out = obs.clone();
    for (i, v) in out.views.iter_mut().enumerate() {
        if zero {
            v.rgb.fill(0.0)
        } else if obs.views.len() > 1 {
            v.rgb = obs.views[(i + 1) % obs.views.len()].rgb.clone()
        } else {
            let n = v.rgb.len() / 3;
            v.rgb.rotate_left(3 * (n / 3));
        }
    }
    out
}
