//! Shared image pyramid, pixel-aligned multiview pooling, implicit field, volume rendering.
use super::{Config, data::RenderShape};
use crate::neural::Builder;
pub use crate::neural::Network;
use meganeura::{Graph, NodeId};

fn constant(g: &mut Graph, like: NodeId, value: f32) -> NodeId {
    let shape = g.node(like).ty.shape.clone();
    g.constant(vec![value; shape.iter().product()], &shape)
}
fn scaled(g: &mut Graph, x: NodeId, value: f32) -> NodeId {
    let s = constant(g, x, value);
    g.mul(x, s)
}
fn add(g: &mut Graph, a: Option<NodeId>, b: NodeId) -> Option<NodeId> {
    Some(a.map_or(b, |a| g.add(a, b)))
}
fn columns(g: &mut Graph, a: NodeId, b: NodeId, rows: usize, ca: usize, cb: usize) -> NodeId {
    // Channel operators use flat NCHW storage; retain explicit matrix boundaries
    // so their flat backward scatter reshapes before joining other gradients.
    let a = g.reshape(a, &[rows * ca]);
    let b = g.reshape(b, &[rows * cb]);
    let h = g.concat(a, b, rows as u32, ca as u32, cb as u32, 1);
    g.reshape(h, &[rows, ca + cb])
}
fn rows(g: &mut Graph, x: NodeId, start: usize, count: usize, channels: usize) -> NodeId {
    let total = g.node(x).ty.num_elements() / channels;
    let x = g.reshape(x, &[total * channels]);
    let x = if start == 0 {
        x
    } else {
        g.split_b(
            x,
            1,
            (start * channels) as u32,
            ((total - start) * channels) as u32,
            1,
        )
    };
    let x = if start + count == total {
        x
    } else {
        g.split_a(
            x,
            1,
            (count * channels) as u32,
            ((total - start - count) * channels) as u32,
            1,
        )
    };
    g.reshape(x, &[count, channels])
}
fn bias(b: &mut Builder, name: &str, value: f32) {
    let p = b
        .params
        .iter_mut()
        .find(|p| p.name == format!("{name}.bias"))
        .unwrap();
    p.kind = crate::model::InitKind::Values(vec![value; p.len]);
}
struct ViewEvidence {
    features: NodeId,
    rgb: NodeId,
    direction: NodeId,
    valid: NodeId,
}

/// Late appearance fusion, not an occlusion oracle. Keep each source until a
/// shared scoring MLP chooses its contribution. No target-direction-dependent
/// geometry, and missing source coverage falls back exactly.
fn source_fusion(
    b: &mut Builder,
    c: &Config,
    q: usize,
    views: &[ViewEvidence],
    inputs: [NodeId; 6],
) -> NodeId {
    let [mean, var, latent, direction, emission, fallback] = inputs;
    let ch = c.channels as usize;
    let mut rgb_sum = None;
    let mut weight_sum = None;
    for view in views {
        let mut feature = columns(&mut b.g, view.features, mean, q, ch, ch);
        feature = columns(&mut b.g, feature, var, q, 2 * ch, ch);
        feature = columns(&mut b.g, feature, latent, q, 3 * ch, c.hidden as usize);
        let exposed = scaled(&mut b.g, view.rgb, c.exposure);
        let one = constant(&mut b.g, exposed, 1.0);
        let den = b.g.add(exposed, one);
        let encoded = b.g.div(exposed, den);
        let dim = 3 * ch + c.hidden as usize;
        feature = columns(&mut b.g, feature, encoded, q, dim, 3);
        feature = columns(&mut b.g, feature, view.direction, q, dim + 3, 3);
        feature = columns(&mut b.g, feature, direction, q, dim + 6, 3);
        let score = b.linear(feature, "field.source.score.in", (dim + 9) as u32, c.hidden);
        let score = b.g.silu(score);
        let score = b.linear(score, "field.source.score.out", c.hidden, 1);
        let score = b.g.softplus(score, 1.0);
        let weight = b.g.mul(score, view.valid);
        weight_sum = add(&mut b.g, weight_sum, weight);
        let weight = b.g.broadcast_inner(weight, 3);
        let part = b.g.mul(view.rgb, weight);
        rgb_sum = add(&mut b.g, rgb_sum, part);
    }
    for p in &mut b.params {
        if p.name == "field.source.score.out.weight" {
            p.kind = crate::model::InitKind::Zeros;
        }
    }
    let weights = weight_sum.unwrap();
    let eps = constant(&mut b.g, weights, 1e-8);
    let den = b.g.add(weights, eps);
    let support = b.g.div(weights, den);
    let den = b.g.broadcast_inner(den, 3);
    let color = b.g.div(rgb_sum.unwrap(), den);
    // Source RGB already contains emission. Subtract before mixing scattered
    // light, since volume rendering adds the supervised emission exactly once.
    let negative = b.g.neg(emission);
    let color = b.g.add(color, negative);
    let color = b.g.relu(color);
    let gate_input = columns(&mut b.g, latent, direction, q, c.hidden as usize, 3);
    let gate = b.linear(gate_input, "field.source.gate", c.hidden + 3, 1);
    bias(b, "field.source.gate", -2.0);
    for p in &mut b.params {
        if p.name == "field.source.gate.weight" {
            p.kind = crate::model::InitKind::Zeros;
        }
    }
    let gate = b.g.sigmoid(gate);
    let gate = b.g.mul(gate, support);
    let gate = b.g.broadcast_inner(gate, 3);
    let one = constant(&mut b.g, gate, 1.0);
    let negative = b.g.neg(gate);
    let rest = b.g.add(one, negative);
    let fallback = b.g.mul(fallback, rest);
    let source = b.g.mul(color, gate);
    b.g.add(fallback, source)
}

struct Field {
    density: NodeId,
    radiance: NodeId,
    scattered: NodeId,
    emission: NodeId,
    environment: NodeId,
}
fn field(b: &mut Builder, c: &Config, q: usize) -> Field {
    let [w, h] = c.extent;
    let n = (w * h) as usize;
    let ch = c.channels as usize;
    let mut sum = None;
    let mut squared = None;
    let mut global = None;
    let mut views = Vec::new();
    for v in 0..c.views {
        let image = b.g.input(&format!("view{v}.rgb_rays"), &[9 * n]);
        let features = b.encode(image, "adapter.rgb_rays", 9, c.extent, c.channels);
        let matrix = b.g.reshape(features, &[ch, n]);
        let ones = b.g.constant(vec![1.0 / (n * c.views) as f32; n], &[n, 1]);
        let pooled = b.g.matmul(matrix, ones);
        global = add(&mut b.g, global, pooled);
        let table = b.g.transpose(matrix);
        let mut projected = None;
        let mut color = None;
        let rgb = (c.view_fusion == super::ViewFusion::LateRgb)
            .then(|| b.g.input(&format!("view{v}.linear_rgb"), &[n, 3]));
        for k in 0..4 {
            let indices = b.g.input_u32(&format!("view{v}.index{k}"), &[q]);
            let weight = b.g.input(&format!("view{v}.weight{k}"), &[q, 1]);
            let scalar_weight = weight;
            let weight = b.g.broadcast_inner(weight, ch);
            let feature = b.g.embedding(indices, table);
            let feature = b.g.mul(feature, weight);
            projected = add(&mut b.g, projected, feature);
            if let Some(table) = rgb {
                let tap = b.g.embedding(indices, table);
                let weight = b.g.broadcast_inner(scalar_weight, 3);
                let tap = b.g.mul(tap, weight);
                color = add(&mut b.g, color, tap);
            }
        }
        let projected = projected.unwrap();
        if let Some(rgb) = color {
            views.push(ViewEvidence {
                features: projected,
                rgb,
                direction: b.g.input(&format!("view{v}.source_direction"), &[q, 3]),
                valid: b.g.input(&format!("view{v}.valid"), &[q, 1]),
            });
        }
        let sq = b.g.mul(projected, projected);
        sum = add(&mut b.g, sum, projected);
        squared = add(&mut b.g, squared, sq);
    }
    let inv = b.g.input("query.inverse_count", &[q, 1]);
    let inv = b.g.broadcast_inner(inv, ch);
    let mean = b.g.mul(sum.unwrap(), inv);
    let second = b.g.mul(squared.unwrap(), inv);
    let square = b.g.mul(mean, mean);
    let neg = b.g.neg(square);
    let var = b.g.add(second, neg);
    let var = b.g.relu(var);
    let combined = columns(&mut b.g, mean, var, q, ch, ch);
    let position = b.g.input("query.position", &[q, c.position_channels()]);
    let combined = columns(
        &mut b.g,
        combined,
        position,
        q,
        2 * ch,
        c.position_channels(),
    );
    let coverage = b.g.input("query.coverage", &[q, 1]);
    let dim = 2 * ch + c.position_channels();
    let combined = columns(&mut b.g, combined, coverage, q, dim, 1);
    let mut latent = b.linear(combined, "field.geometry.in", (dim + 1) as u32, c.hidden);
    latent = b.g.silu(latent);
    for i in 0..3 {
        let a = b.linear(latent, &format!("field.geometry.{i}"), c.hidden, c.hidden);
        let a = b.g.silu(a);
        let a = scaled(&mut b.g, a, 0.1);
        latent = b.g.add(latent, a);
    }
    // This entire branch is independent of query viewing direction.
    let density = b.linear(latent, "field.density", c.hidden, 1);
    bias(b, "field.density", -2.0);
    let density = b.g.softplus(density, 1.0);
    let emission = b.linear(latent, "field.emission", c.hidden, 3);
    bias(b, "field.emission", -4.0);
    let emission = b.g.softplus(emission, 1.0);
    let direction = b.g.input("query.direction", &[q, 3]);
    let directional = columns(&mut b.g, latent, direction, q, c.hidden as usize, 3);
    let directional = b.linear(directional, "field.appearance.in", c.hidden + 3, c.hidden);
    let directional = b.g.silu(directional);
    let radiance = b.linear(directional, "field.appearance.out", c.hidden, 3);
    bias(b, "field.appearance.out", -1.0);
    let radiance = b.g.softplus(radiance, 1.0);
    let scattered = radiance;
    let mut radiance = b.g.add(scattered, emission);
    let pooled = b.g.reshape(global.unwrap(), &[1, ch]);
    let environment = b.linear(pooled, "field.environment", c.channels, 3);
    let environment = b.g.softplus(environment, 1.0);
    // Append parameters after all historical parameters so paired common
    // encoder/geometry/fallback initializations stay identical at the same seed.
    let scattered = if views.is_empty() {
        scattered
    } else {
        source_fusion(
            b,
            c,
            q,
            &views,
            [mean, var, latent, direction, emission, scattered],
        )
    };
    if !views.is_empty() {
        radiance = b.g.add(scattered, emission);
    }
    Field {
        density,
        radiance,
        scattered,
        emission,
        environment,
    }
}
/// Query output order: density [Q,1], radiance [Q,3], emission [Q,3], environment [1,3].
/// Density is per normalized scene unit; radiance is scene-linear. No baked scene table.
pub fn build_points(c: &Config, queries: usize) -> Result<Network, String> {
    c.validate()?;
    if queries == 0 || queries > 1_048_576 {
        return Err("invalid field query count".into());
    }
    let mut b = Builder::new();
    let f = field(&mut b, c, queries);
    b.g.set_outputs(vec![f.density, f.radiance, f.emission, f.environment]);
    Ok(Network {
        graph: b.g,
        params: b.params,
    })
}
struct Rendered {
    termination: NodeId,
    total: NodeId,
    direct: NodeId,
    indirect: NodeId,
}
fn render(g: &mut Graph, f: &Field, shape: RenderShape) -> Rendered {
    let n = shape.rays;
    let mut trans = g.constant(vec![1.0; n], &[n, 1]);
    let mut direct = g.constant(vec![0.0; 3 * n], &[n, 3]);
    let mut indirect = g.constant(vec![0.0; 3 * n], &[n, 3]);
    let deltas = g.input("ray.deltas", &[shape.steps * n, 1]);
    let mut termination = None;
    for s in 0..shape.steps {
        let sigma = rows(g, f.density, s * n, n, 1);
        let dt = rows(g, deltas, s * n, n, 1);
        let tau = g.mul(sigma, dt);
        let neg = g.neg(tau);
        let keep = g.exp(neg);
        let neg = g.neg(keep);
        let one = constant(g, keep, 1.0);
        let alpha = g.add(one, neg);
        let weight = g.mul(trans, alpha);
        let flat = g.reshape(weight, &[n]);
        termination = Some(match termination {
            None => flat,
            Some(prior) => g.concat(prior, flat, 1, (s * n) as u32, n as u32, 1),
        });
        let weight = g.broadcast_inner(weight, 3);
        let emission = rows(g, f.emission, s * n, n, 3);
        let part = g.mul(weight, emission);
        direct = g.add(direct, part);
        let scattered = rows(g, f.scattered, s * n, n, 3);
        let part = g.mul(weight, scattered);
        indirect = g.add(indirect, part);
        trans = g.mul(trans, keep);
    }
    let one = g.constant(vec![1.0; n], &[n, 1]);
    let background = g.matmul(one, f.environment);
    let escape = g.reshape(trans, &[n]);
    let termination = g.concat(
        termination.unwrap(),
        escape,
        1,
        (shape.steps * n) as u32,
        n as u32,
        1,
    );
    let trans = g.broadcast_inner(trans, 3);
    let background = g.mul(trans, background);
    let direct = g.add(direct, background);
    Rendered {
        termination,
        total: g.add(direct, indirect),
        direct,
        indirect,
    }
}
/// Angular incident radiance along arbitrary rays. Outputs are total, direct,
/// indirect [R,3]. Same field/weights as image rendering; no independent head
/// that can explain labels without learning occlusion or scene appearance.
pub fn build_incident(c: &Config, shape: RenderShape) -> Result<Network, String> {
    c.validate()?;
    let q = shape.queries()?;
    if shape.probes != 0 {
        return Err("incident inference does not use surface probes".into());
    }
    let mut b = Builder::new();
    let f = field(&mut b, c, q);
    let image = render(&mut b.g, &f, shape);
    b.g.set_outputs(vec![image.total, image.direct, image.indirect]);
    Ok(Network {
        graph: b.g,
        params: b.params,
    })
}
fn log_radiance(g: &mut Graph, x: NodeId, exposure: f32) -> NodeId {
    let x = scaled(g, x, exposure);
    let one = constant(g, x, 1.0);
    let x = g.add(x, one);
    g.log(x)
}
fn masked_loss(g: &mut Graph, a: NodeId, b: NodeId, mask: NodeId, e: f32) -> NodeId {
    let a = log_radiance(g, a, e);
    let b = log_radiance(g, b, e);
    let a = g.mul(a, mask);
    let b = g.mul(b, mask);
    g.mse_loss(a, b)
}
/// Training uses volume-rendered RGB plus separately masked source-emission and environment losses.
/// Inference output is [rendered RGB, density, radiance, emission, environment].
pub fn build_render(c: &Config, shape: RenderShape, training: bool) -> Result<Network, String> {
    build(c, shape, training, 0, false)
}
/// Last `incident_rays` in each ray-sample block supervise incident light;
/// preceding rays supervise camera RGB. They share density, visibility,
/// emission and scattered radiance. The loss weight is a training parameter,
/// not a runtime config change. Scale target masks for loss-weight ablations.
pub fn build_training(
    c: &Config,
    shape: RenderShape,
    incident_rays: usize,
) -> Result<Network, String> {
    if incident_rays >= shape.rays {
        return Err("incident training must retain at least one image ray".into());
    }
    build(c, shape, true, incident_rays, false)
}
/// Training-only termination NLL; weight-zero targets preserve the entire graph.
pub fn build_surface_training(
    c: &Config,
    shape: RenderShape,
    incident_rays: usize,
) -> Result<Network, String> {
    if incident_rays >= shape.rays {
        return Err("surface training needs image rays".into());
    }
    build(c, shape, true, incident_rays, true)
}
/// Inference diagnostics: RGB and step-major termination masses, including escape.
pub fn build_diagnostics(c: &Config, shape: RenderShape) -> Result<Network, String> {
    c.validate()?;
    let mut b = Builder::new();
    let f = field(&mut b, c, shape.queries()?);
    let r = render(&mut b.g, &f, shape);
    b.g.set_outputs(vec![r.total, r.termination]);
    Ok(Network {
        graph: b.g,
        params: b.params,
    })
}
fn build(
    c: &Config,
    shape: RenderShape,
    training: bool,
    incident_rays: usize,
    surface: bool,
) -> Result<Network, String> {
    c.validate()?;
    let q = shape.queries()?;
    let mut b = Builder::new();
    let f = field(&mut b, c, q);
    let rendered = render(&mut b.g, &f, shape);
    if training {
        let image_rays = shape.rays - incident_rays;
        let rgb = rows(&mut b.g, rendered.total, 0, image_rays, 3);
        let target = b.g.input("target.rgb", &[image_rays, 3]);
        let a = log_radiance(&mut b.g, rgb, c.exposure);
        let target = log_radiance(&mut b.g, target, c.exposure);
        let mut loss = b.g.mse_loss(a, target);
        if surface {
            let target =
                b.g.input("target.termination", &[(shape.steps + 1) * shape.rays]);
            let eps = constant(&mut b.g, rendered.termination, 1e-8);
            let mass = b.g.add(rendered.termination, eps);
            let log_mass = b.g.log(mass);
            let error = b.g.mul(target, log_mass);
            let error = b.g.sum_all(error);
            let error = b.g.neg(error);
            loss = b.g.add(loss, error);
        }
        let target = b.g.input("target.emission", &[q, 3]);
        let mask = b.g.input("target.emission_mask", &[q, 3]);
        let auxiliary = masked_loss(&mut b.g, f.emission, target, mask, c.exposure);
        let auxiliary = scaled(&mut b.g, auxiliary, 0.05);
        loss = b.g.add(loss, auxiliary);
        let target = b.g.input("target.environment", &[1, 3]);
        let mask = b.g.input("target.environment_mask", &[1, 3]);
        let auxiliary = masked_loss(&mut b.g, f.environment, target, mask, c.exposure);
        let auxiliary = scaled(&mut b.g, auxiliary, 0.05);
        loss = b.g.add(loss, auxiliary);
        if incident_rays != 0 {
            for (name, prediction) in [("direct", rendered.direct), ("indirect", rendered.indirect)]
            {
                let prediction = rows(&mut b.g, prediction, image_rays, incident_rays, 3);
                let target =
                    b.g.input(&format!("target.incident_{name}"), &[incident_rays, 3]);
                let mask =
                    b.g.input(&format!("target.incident_{name}_mask"), &[incident_rays, 3]);
                let auxiliary = masked_loss(&mut b.g, prediction, target, mask, c.exposure);
                let auxiliary = scaled(&mut b.g, auxiliary, 0.5);
                loss = b.g.add(loss, auxiliary);
            }
        }
        b.g.set_outputs(vec![loss]);
    } else {
        b.g.set_outputs(vec![
            rendered.total,
            f.density,
            f.radiance,
            f.emission,
            f.environment,
        ]);
    }
    Ok(Network {
        graph: b.g,
        params: b.params,
    })
}
