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
struct Field {
    density: NodeId,
    radiance: NodeId,
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
    for v in 0..c.views {
        let image = b.g.input(&format!("view{v}.rgb_rays"), &[9 * n]);
        let features = b.encode(image, "adapter.rgb_rays", 9, c.extent, c.channels);
        let matrix = b.g.reshape(features, &[ch, n]);
        let ones = b.g.constant(vec![1.0 / (n * c.views) as f32; n], &[n, 1]);
        let pooled = b.g.matmul(matrix, ones);
        global = add(&mut b.g, global, pooled);
        let table = b.g.transpose(matrix);
        let mut projected = None;
        for k in 0..4 {
            let indices = b.g.input_u32(&format!("view{v}.index{k}"), &[q]);
            let weight = b.g.input(&format!("view{v}.weight{k}"), &[q, 1]);
            let weight = b.g.broadcast_inner(weight, ch);
            let feature = b.g.embedding(indices, table);
            let feature = b.g.mul(feature, weight);
            projected = add(&mut b.g, projected, feature);
        }
        let projected = projected.unwrap();
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
    let radiance = b.g.add(radiance, emission);
    let pooled = b.g.reshape(global.unwrap(), &[1, ch]);
    let environment = b.linear(pooled, "field.environment", c.channels, 3);
    let environment = b.g.softplus(environment, 1.0);
    Field {
        density,
        radiance,
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
fn render(g: &mut Graph, f: &Field, shape: RenderShape) -> NodeId {
    let n = shape.rays;
    let mut trans = g.constant(vec![1.0; n], &[n, 1]);
    let mut result = g.constant(vec![0.0; 3 * n], &[n, 3]);
    let deltas = g.input("ray.deltas", &[shape.steps * n, 1]);
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
        let weight = g.broadcast_inner(weight, 3);
        let radiance = rows(g, f.radiance, s * n, n, 3);
        let part = g.mul(weight, radiance);
        result = g.add(result, part);
        trans = g.mul(trans, keep);
    }
    let one = g.constant(vec![1.0; n], &[n, 1]);
    let background = g.matmul(one, f.environment);
    let trans = g.broadcast_inner(trans, 3);
    let background = g.mul(trans, background);
    g.add(result, background)
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
    c.validate()?;
    let q = shape.queries()?;
    let mut b = Builder::new();
    let f = field(&mut b, c, q);
    let rgb = render(&mut b.g, &f, shape);
    if training {
        let target = b.g.input("target.rgb", &[shape.rays, 3]);
        let a = log_radiance(&mut b.g, rgb, c.exposure);
        let target = log_radiance(&mut b.g, target, c.exposure);
        let mut loss = b.g.mse_loss(a, target);
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
        b.g.set_outputs(vec![loss]);
    } else {
        b.g.set_outputs(vec![rgb, f.density, f.radiance, f.emission, f.environment]);
    }
    Ok(Network {
        graph: b.g,
        params: b.params,
    })
}
