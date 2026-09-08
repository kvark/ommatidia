//! Disentangle source depth, volume termination and known-surface appearance.
//! True positions are used ONLY by the explicitly privileged appearance diagnostic.
use super::*;
use ommatidia::{
    dataset::{Plane, Reader},
    field::{
        self, Config, Manifest, Observations, Query, View, ViewFusion, ViewRecord,
        data::{Prepared, RenderShape},
        graph, surface, visibility,
    },
};

fn view(reader: &mut Reader, record: &ViewRecord) -> Result<View> {
    let layout = *reader.layout();
    let sample = reader.sample(record.sample)?;
    let n = layout.hr_texels();
    let mut rgb = vec![0.0; 3 * n];
    for c in 0..3 {
        let values = sample
            .hr_channel(&layout, Plane::Color, c)
            .ok_or("missing reference RGB")?;
        for i in 0..n {
            rgb[3 * i + c] = values[i].to_f32();
        }
    }
    Ok(View {
        camera: record.camera,
        rgb,
    })
}
fn ce(g: &mut meganeura::Graph, p: meganeura::NodeId, name: &str) -> meganeura::NodeId {
    let shape = g.node(p).ty.shape.clone();
    let target = g.input(name, &shape);
    let eps = g.constant(vec![1e-8; shape.iter().product()], &shape);
    let p = g.add(p, eps);
    let lp = g.log(p);
    let term = g.mul(target, lp);
    let sum = g.sum_all(term);
    g.neg(sum)
}
fn model(c: &Config, task: &str, shape: RenderShape) -> Result<Network> {
    let mut m = if task == "volume-depth" {
        graph::build_diagnostics(c, shape)?
    } else {
        graph::build_points(
            c,
            if task == "source-depth" {
                1
            } else {
                shape.rays
            },
        )?
    };
    let outputs = m.graph.outputs().to_vec();
    let loss = match task {
        "source-depth" => {
            let mut total = None;
            for v in 0..c.views {
                let term = ce(
                    &mut m.graph,
                    outputs[4 + v],
                    &format!("target.view{v}.termination"),
                );
                total = Some(total.map_or(term, |prev| m.graph.add(prev, term)));
            }
            total.unwrap()
        }
        "volume-depth" => ce(&mut m.graph, outputs[1], "target.termination"),
        "surface-appearance" => {
            let target = m.graph.input("fit.rgb", &[shape.rays, 3]);
            let mask = m.graph.input("fit.mask", &[shape.rays, 3]);
            let one = m
                .graph
                .constant(vec![1.0; 3 * shape.rays], &[shape.rays, 3]);
            let a = m.graph.add(outputs[1], one);
            let a = m.graph.log(a);
            let a = m.graph.mul(a, mask);
            let b = m.graph.add(target, one);
            let b = m.graph.log(b);
            let b = m.graph.mul(b, mask);
            m.graph.mse_loss(a, b)
        }
        _ => return Err("invalid field component".into()),
    };
    m.graph
        .set_outputs(std::iter::once(loss).chain(outputs).collect());
    Ok(m)
}
fn classes(prob: &[f32], labels: &[f32], bins: usize) -> Value {
    let mut correct = 0;
    let mut valid = 0;
    let mut hit = 0;
    let mut certainty = 0.0;
    for (p, t) in prob.chunks_exact(bins).zip(labels.chunks_exact(bins)) {
        if !t.iter().any(|v| *v > 0.0) {
            continue;
        }
        valid += 1;
        let arg = |v: &[f32]| (0..v.len()).max_by(|a, b| v[*a].total_cmp(&v[*b])).unwrap();
        let predicted = arg(p);
        let truth = arg(t);
        correct += usize::from(predicted == truth);
        hit += usize::from((p[bins - 1] < 0.5) == (truth != bins - 1));
        certainty += p[truth] as f64;
    }
    json!({"valid":valid,"bin_accuracy":correct as f64/valid.max(1) as f64,"hit_miss_accuracy":hit as f64/valid.max(1) as f64,"mean_target_probability":certainty/valid.max(1) as f64})
}
pub(super) fn run(o: &Options, ctx: Arc<blade_graphics::Context>) -> Result<Value> {
    let mut reader = Reader::open(&o.data)?;
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(o.data.with_extension("scene.json"))?)?;
    manifest.validate(reader.len())?;
    let records: Vec<_> = manifest.records.iter().filter(|r| r.scene == 0).collect();
    if records.len() < 5 {
        return Err("need context and fitting cameras".into());
    }
    let c = Config {
        extent: manifest.extent,
        views: 3,
        channels: 8,
        hidden: 32,
        view_fusion: ViewFusion::VisibleRgb,
        ..Default::default()
    };
    c.validate()?;
    let context_ids: Vec<_> = (0..c.views)
        .map(|v| v * (records.len() - 1) / c.views)
        .collect();
    let mut source = Vec::new();
    let mut labels = Vec::new();
    for i in &context_ids {
        source.push(view(&mut reader, records[*i])?);
        labels.push(records[*i].surface.clone());
    }
    let obs = Observations {
        bounds: manifest.scenes[0].bounds,
        views: source,
    };
    obs.validate(&c)?;
    let fit_id = (0..records.len() - 1)
        .find(|i| !context_ids.contains(i))
        .ok_or("no fitting camera")?;
    let fitting = view(&mut reader, records[fit_id])?;
    let surfaces = records[fit_id]
        .surface
        .as_ref()
        .ok_or("missing fitting surface truth")?;
    surfaces.validate(c.extent)?;
    let [w, h] = c.extent;
    let side = 8usize;
    let pixels: Vec<_> = (0..side * side)
        .map(|i| {
            ((i / side * h as usize / side + h as usize / (2 * side)).min(h as usize - 1))
                * w as usize
                + (i % side * w as usize / side + w as usize / (2 * side)).min(w as usize - 1)
        })
        .collect();
    if pixels.windows(2).any(|w| w[0] == w[1]) {
        return Err("diagnostic needs an image at least 8x8".into());
    }
    let rays: Vec<_> = pixels
        .iter()
        .map(|p| {
            fitting
                .camera
                .ray([(p % w as usize) as f32, (p / w as usize) as f32], c.extent)
        })
        .collect();
    let shape = RenderShape {
        rays: rays.len(),
        steps: 64,
        probes: 0,
    };
    let source_targets = visibility::Targets::new(&obs, &c, &labels, 1.0)?;
    if source_targets.valid == 0 {
        return Err("no valid source labels".into());
    }
    let volume_labels: Vec<_> = pixels
        .iter()
        .map(|p| Some((surfaces.distance[*p], surfaces.ray_limit)))
        .collect();
    let volume_targets =
        surface::Targets::new(obs.bounds, &rays, shape.steps, &volume_labels, 1.0)?;
    if volume_targets.valid == 0 {
        return Err("no valid volume labels".into());
    }
    let mut target_rgb = vec![0.0; 3 * shape.rays];
    let mut mask = vec![0.0; 3 * shape.rays];
    let mut valid = 0usize;
    let mut points = Vec::new();
    for (r, (&p, ray)) in pixels.iter().zip(&rays).enumerate() {
        let inside = surfaces.distance[p].filter(|t| {
            let (near, far) = field::data::ray_interval(obs.bounds, *ray);
            *t >= near && *t < far
        });
        points.push(Query {
            position: inside.map_or(obs.bounds.center, |t| {
                std::array::from_fn(|i| ray.origin[i] + ray.direction[i] * t)
            }),
            direction: ray.direction,
        });
        if inside.is_some() {
            target_rgb[3 * r..3 * r + 3].copy_from_slice(&fitting.rgb[3 * p..3 * p + 3]);
            mask[3 * r..3 * r + 3].fill(1.0);
            valid += 1;
        }
    }
    if valid == 0 {
        return Err("no valid known-surface RGB".into());
    }
    let scale = (shape.rays as f32 / valid as f32).sqrt();
    mask.iter_mut().for_each(|v| *v *= scale);
    let (queries, deltas) = match o.task.as_str() {
        "source-depth" => (
            vec![Query {
                position: obs.bounds.center,
                direction: [0.0, 0.0, -1.0],
            }],
            Vec::new(),
        ),
        "volume-depth" => field::data::ray_queries(obs.bounds, &rays, shape.steps)?,
        "surface-appearance" => (points, Vec::new()),
        _ => unreachable!(),
    };
    let prepared = Prepared::new(&obs, &c, &queries)?;
    let m = model(&c, &o.task, shape)?;
    let feed = |s: &mut meganeura::Session| {
        prepared.feed(s);
        match o.task.as_str() {
            "source-depth" => source_targets.feed(s),
            "volume-depth" => {
                s.set_input("ray.deltas", &deltas);
                volume_targets.feed(s);
            }
            "surface-appearance" => {
                s.set_input("fit.rgb", &target_rgb);
                s.set_input("fit.mask", &mask);
            }
            _ => unreachable!(),
        }
    };
    write(
        o.out.join("observations.json"),
        &serde_json::to_value(&obs)?,
    )?;
    write(
        o.out.join("fitting-labels.json"),
        &json!({"role":"training-only ground truth; NOT runtime observations","fitting_record":records[fit_id].sample,"pixels":pixels,"source_mass":source_targets.mass,"volume_mass":volume_targets.mass,"rgb":target_rgb,"mask":mask,"appearance_uses_truth_positions":o.task=="surface-appearance"}),
    )?;
    let (inf, mut report) = optimize(
        &m,
        Arc::clone(&ctx),
        feed,
        &o.out.join("network"),
        o.seed,
        o.steps,
        0.001,
    )?;
    let names = match o.task.as_str() {
        "source-depth" => vec!["field.visibility.weight", "adapter.rgb_rays"],
        "volume-depth" => vec![
            "field.density.weight",
            "field.geometry.in.weight",
            "adapter.rgb_rays",
        ],
        _ => vec![
            "field.appearance.out.weight",
            "field.geometry.in.weight",
            "adapter.rgb_rays",
        ],
    };
    report["gradients"] = gradients(&m, &inf, ctx, feed, &names)?;
    match o.task.as_str() {
        "source-depth" => {
            let mut p = Vec::new();
            for v in 0..c.views {
                p.extend(output(
                    &inf,
                    5 + v,
                    (w * h) as usize * (visibility::BINS + 1),
                ));
            }
            let t: Vec<_> = source_targets.mass.iter().flatten().copied().collect();
            report["classification"] = classes(&p, &t, visibility::BINS + 1);
            write(o.out.join("prediction.json"), &json!({"probability":p}))?;
        }
        "volume-depth" => {
            let p = output(&inf, 2, shape.rays * (shape.steps + 1));
            let transpose = |v: &[f32]| {
                (0..shape.rays)
                    .flat_map(|r| (0..=shape.steps).map(move |s| v[s * shape.rays + r]))
                    .collect::<Vec<_>>()
            };
            report["classification"] = classes(
                &transpose(&p),
                &transpose(&volume_targets.mass),
                shape.steps + 1,
            );
            write(
                o.out.join("prediction.json"),
                &json!({"probability_step_major":p}),
            )?;
        }
        _ => {
            let p = output(&inf, 2, 3 * shape.rays);
            let ids: Vec<_> = (0..p.len()).filter(|i| mask[*i] > 0.0).collect();
            let a: Vec<_> = ids.iter().map(|i| p[*i]).collect();
            let t: Vec<_> = ids.iter().map(|i| target_rgb[*i]).collect();
            report["valid_hits"] = json!(valid);
            report["appearance"] = image_score(&a, &t);
            write(o.out.join("prediction.json"), &json!({"linear_rgb":p}))?;
            png(
                &o.out.join("sampled-appearance.png"),
                &p,
                [side as u32, side as u32],
            )?;
            png(
                &o.out.join("sampled-reference.png"),
                &target_rgb,
                [side as u32, side as u32],
            )?;
        }
    }
    report["context_records"] = json!(context_ids);
    report["fitting_record"] = json!(records[fit_id].sample);
    report["sampled_rays"] = json!(shape.rays);
    report["config"] = serde_json::to_value(&c)?;
    report["privileged_position_diagnostic"] = json!(o.task == "surface-appearance");
    Ok(report)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn isolated_losses_keep_parameters_but_do_not_change_inference() {
        let c = Config {
            extent: [8, 8],
            views: 2,
            channels: 4,
            hidden: 16,
            view_fusion: ViewFusion::VisibleRgb,
            ..Default::default()
        };
        let shape = RenderShape {
            rays: 4,
            steps: 16,
            probes: 0,
        };
        let original = graph::build_points(&c, 4).unwrap();
        for task in ["source-depth", "volume-depth", "surface-appearance"] {
            let m = model(&c, task, shape).unwrap();
            assert_eq!(
                m.params
                    .iter()
                    .map(|p| (&p.name, p.len))
                    .collect::<Vec<_>>(),
                original
                    .params
                    .iter()
                    .map(|p| (&p.name, p.len))
                    .collect::<Vec<_>>()
            );
            assert_eq!(m.graph.node(m.graph.outputs()[0]).ty.num_elements(), 1);
        }
    }
}
