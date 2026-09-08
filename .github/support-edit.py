from pathlib import Path

def edit(name, old, new):
    p=Path(name);s=p.read_text();assert s.count(old)==1,(name,old[:100],s.count(old));p.write_text(s.replace(old,new))

edit('ommatidia/src/field/mod.rs', 'pub mod surface;', 'pub mod surface;\npub mod visibility;')
edit('ommatidia/src/field/mod.rs', '    LateRgb,\n}', '    LateRgb,\n    /// RGB-predicted source termination distributions gate geometry and appearance.\n    VisibleRgb,\n}\nimpl ViewFusion {\n    pub fn uses_rgb(self) -> bool { self != Self::Moments }\n}')
edit('ommatidia/src/field/data.rs','    pub valid: Vec<f32>,','    pub valid: Vec<f32>,\n    pub survival: Option<Vec<f32>>,')
p=Path('ommatidia/src/field/data.rs');s=p.read_text().replace('config.view_fusion == ViewFusion::LateRgb','config.view_fusion.uses_rgb()');p.write_text(s)
edit('ommatidia/src/field/data.rs','                valid: vec![0.0; q],','                valid: vec![0.0; q],\n                survival: (config.view_fusion == ViewFusion::VisibleRgb).then(|| vec![0.0; q * (visibility::BINS + 1)]),')
edit('ommatidia/src/field/data.rs','                source.direction[3 * i..3 * i + 3].copy_from_slice(&direction);','''                source.direction[3 * i..3 * i + 3].copy_from_slice(&direction);
                if let Some(coefficients) = &mut source.survival {
                    let row = visibility::survival(observations.bounds, view.camera.origin, query.position);
                    coefficients[i * (visibility::BINS + 1)..(i + 1) * (visibility::BINS + 1)].copy_from_slice(&row);
                }''')
edit('ommatidia/src/field/data.rs','                session.set_input(&format!("view{v}.valid"), &source.valid);','''                session.set_input(&format!("view{v}.valid"), &source.valid);
                if let Some(survival) = &source.survival {
                    session.set_input(&format!("view{v}.survival"), survival);
                }''')
edit('ommatidia/src/field/graph.rs','    let support = b.g.div(weights, den);','''    let support = if c.view_fusion == super::ViewFusion::VisibleRgb {
        // Do not renormalize weak absolute visibility back to full copying.
        let total = views.iter().fold(None, |a, v| add(&mut b.g, a, v.valid)).unwrap();
        scaled(&mut b.g, total, 1.0 / c.views as f32)
    } else { b.g.div(weights, den) };''')
edit('ommatidia/src/field/graph.rs','    environment: NodeId,\n}', '    environment: NodeId,\n    visibility: Vec<NodeId>,\n}')
edit('ommatidia/src/field/graph.rs','    let mut views = Vec::new();','    let mut views = Vec::new();\n    let mut visibility = Vec::new();\n    let mut visible_sum = None;')
edit('ommatidia/src/field/graph.rs','        let table = b.g.transpose(matrix);','''        let table = b.g.transpose(matrix);
        let probabilities = if c.view_fusion == super::ViewFusion::VisibleRgb {
            let logits = b.linear(table, "field.visibility", c.channels, (super::visibility::BINS + 1) as u32);
            for p in &mut b.params {
                if p.name == "field.visibility.weight" { p.kind = crate::model::InitKind::Zeros; }
            }
            let probability = b.g.softmax(logits);
            visibility.push(probability);
            Some(probability)
        } else { None };
        let mut projected_probability = None;''')
p=Path('ommatidia/src/field/graph.rs');s=p.read_text().replace('c.view_fusion == super::ViewFusion::LateRgb','c.view_fusion.uses_rgb()');p.write_text(s)
edit('ommatidia/src/field/graph.rs','            let scalar_weight = weight;','''            let scalar_weight = weight;
            if let Some(table) = probabilities {
                let tap = b.g.embedding(indices, table);
                let weight = b.g.broadcast_inner(scalar_weight, super::visibility::BINS + 1);
                let tap = b.g.mul(tap, weight);
                projected_probability = add(&mut b.g, projected_probability, tap);
            }''')
edit('ommatidia/src/field/graph.rs','''        let projected = projected.unwrap();
        if let Some(rgb) = color {''','''        let projected = projected.unwrap();
        let visible = projected_probability.map(|probability| {
            let survival = b.g.input(&format!("view{v}.survival"), &[q, super::visibility::BINS + 1]);
            let mass = b.g.mul(probability, survival);
            let mass = b.g.sum_inner(mass);
            b.g.reshape(mass, &[q, 1])
        });
        if let Some(weight) = visible { visible_sum = add(&mut b.g, visible_sum, weight); }
        if let Some(rgb) = color {''')
edit('ommatidia/src/field/graph.rs','                valid: b.g.input(&format!("view{v}.valid"), &[q, 1]),','''                valid: visible.unwrap_or_else(|| b.g.input(&format!("view{v}.valid"), &[q, 1])),''')
edit('ommatidia/src/field/graph.rs','''        let sq = b.g.mul(projected, projected);
        sum = add(&mut b.g, sum, projected);
        squared = add(&mut b.g, squared, sq);
    }
    let inv = b.g.input("query.inverse_count", &[q, 1]);''','''        let sq = b.g.mul(projected, projected);
        let (projected, sq) = if let Some(weight) = visible {
            let weight = b.g.broadcast_inner(weight, ch);
            (b.g.mul(projected, weight), b.g.mul(sq, weight))
        } else { (projected, sq) };
        sum = add(&mut b.g, sum, projected);
        squared = add(&mut b.g, squared, sq);
    }
    let inv = if let Some(total) = visible_sum {
        let eps = constant(&mut b.g, total, 1e-6);
        let den = b.g.add(total, eps);
        let one = constant(&mut b.g, total, 1.0);
        b.g.div(one, den)
    } else { b.g.input("query.inverse_count", &[q, 1]) };''')
edit('ommatidia/src/field/graph.rs','    let coverage = b.g.input("query.coverage", &[q, 1]);','''    let coverage = if let Some(total) = visible_sum {
        scaled(&mut b.g, total, 1.0 / c.views as f32)
    } else { b.g.input("query.coverage", &[q, 1]) };''')
edit('ommatidia/src/field/graph.rs','''        environment,
    }
}''','''        environment,
        visibility,
    }
}''')
edit('ommatidia/src/field/graph.rs','    b.g.set_outputs(vec![f.density, f.radiance, f.emission, f.environment]);','''    let mut outputs = vec![f.density, f.radiance, f.emission, f.environment];
    outputs.extend(f.visibility); // VisibleRgb: source pixel-major probabilities.
    b.g.set_outputs(outputs);''')
edit('ommatidia/src/field/graph.rs','        if surface {','''        for (v, probability) in f.visibility.iter().enumerate() {
            let labels = b.g.input(&format!("target.view{v}.termination"), &[(c.extent[0] * c.extent[1]) as usize, super::visibility::BINS + 1]);
            let eps = constant(&mut b.g, *probability, 1e-8);
            let p = b.g.add(*probability, eps);
            let log = b.g.log(p);
            let weighted = b.g.mul(labels, log);
            let ce = b.g.sum_all(weighted);
            let ce = b.g.neg(ce);
            loss = b.g.add(loss, ce);
        }
        if surface {''')
edit('ommatidia-train/src/bin/field.rs','    record: field::SceneRecord,','    record: field::SceneRecord,\n    context_surfaces: Vec<Option<surface::Capture>>,')
edit('ommatidia-train/src/bin/field.rs','        let train: Vec<_> = views','''        let context_surfaces = context_indices.iter().map(|i| views[*i].surface.clone()).collect();
        let train: Vec<_> = views''')
edit('ommatidia-train/src/bin/field.rs','''            record,
        });''','''            record,
            context_surfaces,
        });''')
edit('ommatidia-train/src/bin/field.rs','    let mut surface_weight = None::<f32>;','    let mut surface_weight = None::<f32>;\n    let mut visibility_weight = 0.05f32;')
p=Path('ommatidia-train/src/bin/field.rs');s=p.read_text().replace('moments|late-rgb --eval-checkpoint','moments|late-rgb|visible-rgb --visibility-weight F [0.05] --eval-checkpoint').replace('                    "late-rgb" => field::ViewFusion::LateRgb,','                    "late-rgb" => field::ViewFusion::LateRgb,\n                    "visible-rgb" => field::ViewFusion::VisibleRgb,').replace('view fusion must be moments or late-rgb','view fusion must be moments, late-rgb or visible-rgb');p.write_text(s)
edit('ommatidia-train/src/bin/field.rs','            "--view-fusion" => {','            "--visibility-weight" => visibility_weight = v.parse()?,\n            "--view-fusion" => {')
edit('ommatidia-train/src/bin/field.rs','    let held = eval.as_ref().map(|p| load(p, &c)).transpose()?;','''    if !visibility_weight.is_finite() || visibility_weight < 0.0 {
        return Err("visibility weight must be finite and nonnegative".into());
    }
    let visibility_targets = if c.view_fusion == field::ViewFusion::VisibleRgb && eval_checkpoint.is_none() {
        train.iter().map(|e| field::visibility::Targets::new(&e.observations, &c, &e.context_surfaces, visibility_weight)).collect::<std::result::Result<Vec<_>, _>>()?
    } else { Vec::new() };
    let held = eval.as_ref().map(|p| load(p, &c)).transpose()?;''')
edit('ommatidia-train/src/bin/field.rs','        targets.feed(&mut session);','''        targets.feed(&mut session);
        if !visibility_targets.is_empty() {
            visibility_targets[step % train.len()].feed(&mut session);
        }''')
edit('ommatidia-train/src/bin/field.rs','"surface_weight":surface_weight,"emitter_fraction":emitter_fraction,','"surface_weight":surface_weight,"visibility_weight":visibility_weight,"emitter_fraction":emitter_fraction,')
edit('ommatidia/src/transport/graph.rs','''pub fn build(config: Config, low: [u32; 2], unroll: usize) -> Result<Network, String> {
    config.validate(low)?;''','''pub fn build(config: Config, low: [u32; 2], unroll: usize) -> Result<Network, String> {
    build_projected(config, low, unroll, false)
}

/// Target-only projected-colour supervision, without new inference parameters.
/// A zero coefficient retains the paired training graph.
pub fn build_projected(config: Config, low: [u32; 2], unroll: usize, projected: bool) -> Result<Network, String> {
    if projected && unroll == 0 { return Err("projected supervision is training-only".into()); }
    config.validate(low)?;''')
edit('ommatidia/src/transport/graph.rs','    let mut previous = None;','''    let projected_weight = projected.then(|| b.g.input("loss.projected_weight", &[1]));
    let mut previous = None;''')
edit('ommatidia/src/transport/graph.rs','''        previous_target = Some(target);
        total_loss''','''        if let Some(weight) = projected_weight {
            let teacher = b.g.input(&format!("{tag}.projected"), &[6 * n]);
            let fixed_scale = filled(&mut b.g, image, config.exposure);
            let auxiliary = scaled_mse(&mut b.g, image, teacher, fixed_scale);
            let auxiliary = b.g.mul(auxiliary, weight);
            loss = b.g.add(loss, auxiliary);
        }
        previous_target = Some(target);
        total_loss''')
edit('ommatidia-train/src/bin/transport.rs','    let mut candidate_oracle = false;','    let mut candidate_oracle = false;\n    let mut projected_weight = None::<f32>;')
edit('ommatidia-train/src/bin/transport.rs','            "--lr" => rate = v.parse()?,','            "--lr" => rate = v.parse()?,\n            "--projected-weight" => projected_weight = Some(v.parse()?),')
edit('ommatidia-train/src/bin/transport.rs','    weights.validate()?;','''    weights.validate()?;
    if projected_weight.is_some_and(|w| !w.is_finite() || w < 0.0) {
        return Err("projected weight must be finite and nonnegative".into());
    }
    if projected_weight.is_some() && eval_only { return Err("projected targets are training-only".into()); }''')
edit('ommatidia-train/src/bin/transport.rs','        let network = graph::build(config, low, unroll)?;','        let network = graph::build_projected(config, low, unroll, projected_weight.is_some())?;')
edit('ommatidia-train/src/bin/transport.rs','''                graph::feed(&mut session, &format!("f{slot}"), &prepared, target, slot);
                if fixed_exposure_loss''','''                graph::feed(&mut session, &format!("f{slot}"), &prepared, target, slot);
                if let Some(weight) = projected_weight {
                    let candidates = ommatidia::transport::oracle::Candidates {
                        spatial: prepared.candidates.clone(), history: prepared.history.clone(),
                        prior: prepared.prior.clone(), selected: Vec::new(),
                    };
                    let projection = ommatidia::transport::oracle::reconstruct(&candidates, &target.lobes)?;
                    session.set_input(&format!("f{slot}.projected"), &projection.lobes);
                    session.set_input("loss.projected_weight", &[weight]);
                }
                if fixed_exposure_loss''')
edit('ommatidia-train/src/bin/transport.rs','"loss_weights":weights,','"loss_weights":weights,"projected_weight":projected_weight,')
p=Path('ommatidia-train/src/bin/transport.rs');s=p.read_text().replace(r'[--fixed-exposure-loss]\n',r'[--fixed-exposure-loss] [--projected-weight F]\n');p.write_text(s)
edit('ommatidia/src/field/data.rs','                session.set_input(&format!("view{v}.valid"), &source.valid);','''                if source.survival.is_none() { session.set_input(&format!("view{v}.valid"), &source.valid); }''')
edit('ommatidia/src/field/data.rs','''        session.set_input("query.inverse_count", &self.inverse_count);
        session.set_input("query.coverage", &self.coverage);''','''        if !self.sources.iter().any(|s| s.survival.is_some()) {
            session.set_input("query.inverse_count", &self.inverse_count);
            session.set_input("query.coverage", &self.coverage);
        }''')
