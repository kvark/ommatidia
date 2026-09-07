from pathlib import Path

def edit(name, old, new):
    p=Path(name); s=p.read_text()
    assert s.count(old)==1, (name, old[:80], s.count(old))
    p.write_text(s.replace(old,new))

edit('ommatidia/src/field/data.rs', '''    bounds.validate()?;
    RenderShape {''', '''    sample_rays(bounds, rays, steps, None)
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
    RenderShape {''')
edit('ommatidia/src/field/data.rs', '    let mut queries = Vec::new();\n    let mut deltas = Vec::new();', '    let mut rng = seed.map(crate::rng::Rng::new);\n    let mut queries = Vec::with_capacity(rays.len() * steps);\n    let mut deltas = Vec::with_capacity(rays.len() * steps);')
edit('ommatidia/src/field/data.rs', '            let t = near + (s as f32 + 0.5) * dt;', '            let u = rng.as_mut().map_or(0.5, |r| r.uniform());\n            let t = near + (s as f32 + u) * dt;')

edit('ommatidia-train/src/bin/field.rs', '    let mut diagnostics = false;', '    let mut diagnostics = false;\n    let mut stratified = false;')
edit('ommatidia-train/src/bin/field.rs', '        if flag == "--diagnostics" {', '        if flag == "--stratified" {\n            stratified = true;\n            continue;\n        }\n        if flag == "--diagnostics" {')
edit('ommatidia-train/src/bin/field.rs', r'  --rays N --samples N --probes N --rate F\n', r'  --rays N --samples N --probes N --rate F --stratified\n')
edit('ommatidia-train/src/bin/field.rs', '''        let (mut queries, deltas) =
            data::ray_queries(example.observations.bounds, &rays, shape.steps)?;''', '''        // Independent stream: changing sample count/mode must not change
        // fitting camera pixels or emission/incident-probe choices.
        let sample_seed = seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let (mut queries, deltas) = if stratified {
            data::stratified_ray_queries(example.observations.bounds, &rays, shape.steps, sample_seed)?
        } else {
            data::ray_queries(example.observations.bounds, &rays, shape.steps)?
        };''')
edit('ommatidia-train/src/bin/field.rs', '"surface_weight":surface_weight,"emitter_fraction":emitter_fraction,"diagnostics":diagnostics,', '''"surface_weight":surface_weight,"emitter_fraction":emitter_fraction,"diagnostics":diagnostics,
        "sampling":if stratified {"stratified-fixed-intervals"} else {"midpoint"},
        "evaluation_sampling":"midpoint","parameter_count":model.params.iter().map(|p|p.len).sum::<usize>(),
        "image_rays_seen":steps as u64 * shape.rays as u64,
        "ray_queries_seen":steps as u64 * training_shape.rays as u64 * shape.steps as u64,''')

edit('ommatidia/src/transport/graph.rs', 'fn filled(g: &mut Graph, x: NodeId, value: f32) -> NodeId {', '''/// Training-only objective coefficients. Defaults preserve the prior objective.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct LossWeights {
    pub compressed: f32,
    pub physical: f32,
    pub low_frequency: f32,
    pub confidence: f32,
    pub temporal: f32,
}
impl Default for LossWeights {
    fn default() -> Self {
        Self { compressed: 1.0, physical: 0.1, low_frequency: 0.05, confidence: 0.01, temporal: 0.01 }
    }
}
impl LossWeights {
    fn values(self) -> [f32; 5] {
        [self.compressed, self.physical, self.low_frequency, self.confidence, self.temporal]
    }
    pub fn validate(self) -> Result<(), String> {
        let values = self.values();
        if values.iter().any(|v| !v.is_finite() || *v < 0.0) || values.iter().all(|v| *v == 0.0) {
            return Err("loss weights must be finite, nonnegative and not all zero".into());
        }
        Ok(())
    }
    /// Call after the last `feed` in an unroll to override the default loss.
    pub fn feed(self, session: &mut meganeura::Session) {
        session.set_input("loss.weights", &self.values());
    }
}

fn filled(g: &mut Graph, x: NodeId, value: f32) -> NodeId {''')
edit('ommatidia/src/transport/graph.rs', '    let mut b = Builder::new();', '''    let mut b = Builder::new();
    let weights: Option<[NodeId; 5]> = (unroll > 0).then(|| {
        let input = b.g.input("loss.weights", &[5]);
        split(&mut b.g, input, 5, 1, 1).try_into().unwrap()
    });''')
edit('ommatidia/src/transport/graph.rs', '''        let mut loss = b.g.mse_loss(encoded, encoded_target);
        let physical = scaled_mse(&mut b.g, image, target, scale);
        let weight = b.g.scalar(0.1);
        let physical = b.g.mul(physical, weight);''', '''        let [compressed_weight, physical_weight, low_frequency_weight, confidence_weight, temporal_weight] = weights.unwrap();
        let loss = b.g.mse_loss(encoded, encoded_target);
        let mut loss = b.g.mul(loss, compressed_weight);
        let physical = scaled_mse(&mut b.g, image, target, scale);
        let physical = b.g.mul(physical, physical_weight);''')
edit('ommatidia/src/transport/graph.rs', '        let weight = b.g.scalar(0.05);\n        let lf = b.g.mul(lf, weight);', '        let lf = b.g.mul(lf, low_frequency_weight);')
edit('ommatidia/src/transport/graph.rs', '        let weight = b.g.scalar(0.01);\n        let cl = b.g.mul(cl, weight);', '        let cl = b.g.mul(cl, confidence_weight);')
edit('ommatidia/src/transport/graph.rs', '            let weight = b.g.scalar(0.01);\n            let tl = b.g.mul(tl, weight);', '            let tl = b.g.mul(tl, temporal_weight);')
edit('ommatidia/src/transport/graph.rs', '    session.set_input(&format!("{tag}.features"), &p.features);', '    LossWeights::default().feed(session);\n    session.set_input(&format!("{tag}.features"), &p.features);')

edit('ommatidia-train/src/bin/transport.rs', '    let mut fixed_exposure_loss = false;', '    let mut fixed_exposure_loss = false;\n    let mut weights = graph::LossWeights::default();')
edit('ommatidia-train/src/bin/transport.rs', r'[--fixed-exposure-loss]\nCaptures', r'[--fixed-exposure-loss]\n  --compressed-weight F [1] --physical-weight F [0.1] --low-frequency-weight F [0.05]\n  --confidence-weight F [0.01] --temporal-weight F [0.01]\nCaptures')
edit('ommatidia-train/src/bin/transport.rs', '            "--lr" => rate = v.parse()?,', '''            "--lr" => rate = v.parse()?,
            "--compressed-weight" => weights.compressed = v.parse()?,
            "--physical-weight" => weights.physical = v.parse()?,
            "--low-frequency-weight" => weights.low_frequency = v.parse()?,
            "--confidence-weight" => weights.confidence = v.parse()?,
            "--temporal-weight" => weights.temporal = v.parse()?,''')
edit('ommatidia-train/src/bin/transport.rs', '    if !(1..=8).contains(&unroll)', '    weights.validate()?;\n    if !(1..=8).contains(&unroll)')
edit('ommatidia-train/src/bin/transport.rs', '            let fraction = update as f32 / steps.max(1) as f32;', '            weights.feed(&mut session);\n            let fraction = update as f32 / steps.max(1) as f32;')
edit('ommatidia-train/src/bin/transport.rs', '"fixed_exposure_loss":fixed_exposure_loss,', '"fixed_exposure_loss":fixed_exposure_loss,"loss_weights":weights,')
