//! CPU camera projection and batch assembly. No depth/normal/velocity lookup.
use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct SourceEvidence {
    pub rgb: Vec<f32>,
    pub direction: Vec<f32>,
    pub valid: Vec<f32>,
    pub survival: Option<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Prepared {
    pub images: Vec<Vec<f32>>,
    pub sources: Vec<SourceEvidence>,
    pub indices: Vec<[Vec<u32>; 4]>,
    pub weights: Vec<[Vec<f32>; 4]>,
    pub positions: Vec<f32>,
    pub directions: Vec<f32>,
    pub inverse_count: Vec<f32>,
    pub coverage: Vec<f32>,
}
impl Prepared {
    pub fn new(
        observations: &Observations,
        config: &Config,
        queries: &[Query],
    ) -> Result<Self, String> {
        observations.validate(config)?;
        if queries.is_empty() || queries.len() > 1_048_576 {
            return Err("invalid query count".into());
        }
        if queries.iter().any(|q| {
            !q.position.iter().chain(&q.direction).all(|v| v.is_finite())
                || (dot(q.direction, q.direction) - 1.0).abs() > 1e-3
        }) {
            return Err("queries need finite position and unit direction".into());
        }
        let [w, h] = config.extent.map(|n| n as usize);
        let n = w * h;
        let q = queries.len();
        let mut result = Self {
            images: Vec::new(),
            sources: Vec::new(),
            indices: Vec::new(),
            weights: Vec::new(),
            positions: Vec::new(),
            directions: Vec::new(),
            inverse_count: vec![0.0; q],
            coverage: vec![0.0; q],
        };
        for query in queries {
            let x = observations.bounds.normalize(query.position);
            if x.iter().any(|v| !v.is_finite() || v.abs() > 1e10) {
                return Err("normalized query is out of range".into());
            }
            result.positions.extend(x);
            for f in 0..config.position_frequencies {
                let k = std::f32::consts::PI * (1u32 << f) as f32;
                result.positions.extend(x.map(|v| (k * v).sin()));
                result.positions.extend(x.map(|v| (k * v).cos()));
            }
            result.directions.extend(query.direction);
        }
        for view in &observations.views {
            let mut image = vec![0.0; 9 * n];
            let origin = observations.bounds.normalize(view.camera.origin);
            if origin.iter().any(|v| !v.is_finite()) {
                return Err("normalized camera overflows".into());
            }
            for i in 0..n {
                let ray = view
                    .camera
                    .ray([(i % w) as f32, (i / w) as f32], config.extent);
                for c in 0..3 {
                    let v = view.rgb[3 * i + c] * config.exposure;
                    if !v.is_finite() {
                        return Err("source exposure overflows".into());
                    }
                    image[c * n + i] = v / (1.0 + v);
                    image[(3 + c) * n + i] = origin[c];
                    image[(6 + c) * n + i] = ray.direction[c];
                }
            }
            let mut source = SourceEvidence {
                rgb: if config.view_fusion.uses_rgb() {
                    view.rgb.clone()
                } else {
                    Vec::new()
                },
                direction: vec![0.0; 3 * q],
                valid: vec![0.0; q],
                survival: (config.view_fusion == ViewFusion::VisibleRgb)
                    .then(|| vec![0.0; q * (visibility::BINS + 1)]),
            };
            let mut indices: [Vec<u32>; 4] = std::array::from_fn(|_| vec![0; q]);
            let mut weights: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0; q]);
            for (i, query) in queries.iter().enumerate() {
                let Some([x, y]) = view.camera.project(query.position, config.extent) else {
                    continue;
                };
                if !x.is_finite()
                    || !y.is_finite()
                    || x < 0.0
                    || y < 0.0
                    || x > (w - 1) as f32
                    || y > (h - 1) as f32
                {
                    continue;
                }
                let x0 = x.floor() as usize;
                let y0 = y.floor() as usize;
                let fx = x - x0 as f32;
                let fy = y - y0 as f32;
                for k in 0..4 {
                    let dx = k % 2;
                    let dy = k / 2;
                    indices[k][i] = ((y0 + dy).min(h - 1) * w + (x0 + dx).min(w - 1)) as u32;
                    weights[k][i] = (if dx == 0 { 1.0 - fx } else { fx })
                        * (if dy == 0 { 1.0 - fy } else { fy });
                }
                result.coverage[i] += 1.0;
                source.valid[i] = 1.0;
                let direction = unit(std::array::from_fn(|c| {
                    query.position[c] - view.camera.origin[c]
                }));
                source.direction[3 * i..3 * i + 3].copy_from_slice(&direction);
                if let Some(coefficients) = &mut source.survival {
                    let row = visibility::survival(
                        observations.bounds,
                        view.camera.origin,
                        query.position,
                    );
                    coefficients[i * (visibility::BINS + 1)..(i + 1) * (visibility::BINS + 1)]
                        .copy_from_slice(&row);
                }
            }
            if config.view_fusion.uses_rgb() {
                result.sources.push(source);
            }
            result.images.push(image);
            result.indices.push(indices);
            result.weights.push(weights);
        }
        for (inverse, count) in result.inverse_count.iter_mut().zip(&mut result.coverage) {
            *inverse = 1.0 / count.max(1.0);
            *count /= config.views as f32;
        }
        Ok(result)
    }
    pub fn feed(&self, session: &mut meganeura::Session) {
        for v in 0..self.images.len() {
            session.set_input(&format!("view{v}.rgb_rays"), &self.images[v]);
            if let Some(source) = self.sources.get(v) {
                session.set_input(&format!("view{v}.linear_rgb"), &source.rgb);
                session.set_input(&format!("view{v}.source_direction"), &source.direction);
                if source.survival.is_none() {
                    session.set_input(&format!("view{v}.valid"), &source.valid);
                }
                if let Some(survival) = &source.survival {
                    session.set_input(&format!("view{v}.survival"), survival);
                }
            }
            for k in 0..4 {
                session.set_input_u32(&format!("view{v}.index{k}"), &self.indices[v][k]);
                session.set_input(&format!("view{v}.weight{k}"), &self.weights[v][k]);
            }
        }
        session.set_input("query.position", &self.positions);
        session.set_input("query.direction", &self.directions);
        if !self.sources.iter().any(|s| s.survival.is_some()) {
            session.set_input("query.inverse_count", &self.inverse_count);
            session.set_input("query.coverage", &self.coverage);
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct RenderShape {
    pub rays: usize,
    pub steps: usize,
    pub probes: usize,
}
impl RenderShape {
    pub fn queries(self) -> Result<usize, String> {
        if self.rays == 0
            || self.rays > 65536
            || !(2..=256).contains(&self.steps)
            || self.probes > 65536
        {
            return Err("invalid bounded ray/sample/probe counts".into());
        }
        self.rays
            .checked_mul(self.steps)
            .and_then(|n| n.checked_add(self.probes))
            .filter(|n| *n <= 1_048_576)
            .ok_or("too many field samples".into())
    }
}
/// All ground truth is here, outside Prepared and Observations.
pub struct Targets {
    pub rgb: Vec<f32>,
    pub emission: Vec<f32>,
    pub emission_mask: Vec<f32>,
    pub environment: [f32; 3],
    pub environment_mask: [f32; 3],
}
impl Targets {
    pub fn feed(&self, session: &mut meganeura::Session) {
        session.set_input("target.rgb", &self.rgb);
        session.set_input("target.emission", &self.emission);
        session.set_input("target.emission_mask", &self.emission_mask);
        session.set_input("target.environment", &self.environment);
        session.set_input("target.environment_mask", &self.environment_mask);
    }
}
/// Uniform ray samples through the acquisition cube; no target depth sampling.
pub fn ray_queries(
    bounds: Bounds,
    rays: &[Ray],
    steps: usize,
) -> Result<(Vec<Query>, Vec<f32>), String> {
    sample_rays(bounds, rays, steps, None)
}

/// One random quadrature point per fixed ray interval. The interval widths and
/// surface-termination classes stay unchanged; only where the field is queried
/// changes. No target metadata is accepted. Evaluation keeps deterministic midpoints.
pub fn stratified_ray_queries(
    bounds: Bounds,
    rays: &[Ray],
    steps: usize,
    seed: u64,
) -> Result<(Vec<Query>, Vec<f32>), String> {
    sample_rays(bounds, rays, steps, Some(seed))
}

fn sample_rays(
    bounds: Bounds,
    rays: &[Ray],
    steps: usize,
    seed: Option<u64>,
) -> Result<(Vec<Query>, Vec<f32>), String> {
    bounds.validate()?;
    RenderShape {
        rays: rays.len(),
        steps,
        probes: 0,
    }
    .queries()?;
    if rays.iter().any(|r| {
        r.origin.iter().chain(&r.direction).any(|v| !v.is_finite())
            || (dot(r.direction, r.direction) - 1.0).abs() > 1e-3
    }) {
        return Err("rays require finite origins and unit directions".into());
    }
    let intervals: Vec<_> = rays.iter().map(|r| ray_interval(bounds, *r)).collect();
    let mut rng = seed.map(crate::rng::Rng::new);
    let mut queries = Vec::with_capacity(rays.len() * steps);
    let mut deltas = Vec::with_capacity(rays.len() * steps);
    for s in 0..steps {
        for (ray, &(near, far)) in rays.iter().zip(&intervals) {
            let dt = (far - near) / steps as f32;
            let u = rng.as_mut().map_or(0.5, |r| r.uniform());
            let t = near + (s as f32 + u) * dt;
            queries.push(Query {
                position: std::array::from_fn(|i| ray.origin[i] + t * ray.direction[i]),
                direction: ray.direction,
            });
            deltas.push(dt / bounds.radius);
        }
    }
    Ok((queries, deltas))
}
/// Append balanced emitter/non-emitter probes. Probe labels are never feature inputs.
pub fn append_probes(
    queries: &mut Vec<Query>,
    lighting: &Lighting,
    count: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let offset = queries.len();
    let q = offset + count;
    let mut labels = vec![0.0; 3 * q];
    let mut mask = vec![0.0; 3 * q];
    let positive: Vec<_> = lighting
        .probes
        .iter()
        .filter(|p| p.radiance.iter().any(|v| *v > 0.0))
        .collect();
    let negative: Vec<_> = lighting
        .probes
        .iter()
        .filter(|p| p.radiance.iter().all(|v| *v == 0.0))
        .collect();
    let mut rng = crate::rng::Rng::new(seed);
    let mut valid = 0;
    for i in 0..count {
        let pool = if i % 2 == 0 && !positive.is_empty() {
            &positive
        } else if !negative.is_empty() {
            &negative
        } else {
            &positive
        };
        let p = if pool.is_empty() {
            None
        } else {
            Some(pool[rng.below(pool.len() as u32) as usize])
        };
        queries.push(Query {
            position: p.map_or([0.0; 3], |p| p.position),
            direction: [0.0, 0.0, 1.0],
        });
        if let Some(p) = p {
            labels[3 * (offset + i)..3 * (offset + i + 1)].copy_from_slice(&p.radiance);
            mask[3 * (offset + i)..3 * (offset + i + 1)].fill(1.0);
            valid += 1;
        }
    }
    if valid > 0 {
        let scale = (q as f32 / valid as f32).sqrt();
        mask.iter_mut().for_each(|v| *v *= scale);
    }
    (labels, mask)
}

/// Ray/cube interval in world units. Depends only on acquisition inputs.
pub fn ray_interval(bounds: Bounds, r: Ray) -> (f32, f32) {
    let mut near = 0.0f32;
    let mut far = f32::INFINITY;
    for axis in 0..3 {
        let lo = bounds.center[axis] - bounds.radius;
        let hi = bounds.center[axis] + bounds.radius;
        if r.direction[axis].abs() < 1e-8 {
            if r.origin[axis] < lo || r.origin[axis] > hi {
                return (0.0, 0.0);
            }
        } else {
            let a = (lo - r.origin[axis]) / r.direction[axis];
            let b = (hi - r.origin[axis]) / r.direction[axis];
            near = near.max(a.min(b));
            far = far.min(a.max(b));
        }
    }
    if far <= near || !far.is_finite() {
        (0.0, 0.0)
    } else {
        (near, far)
    }
}
