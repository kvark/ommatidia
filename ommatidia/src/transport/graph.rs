//! Shared local spatial core with lobe-specific reconstruction heads.
//! Short unrolls differentiate through radiance reprojection, not camera geometry.
use super::*;
use meganeura::{Graph, NodeId};
use std::collections::BTreeMap;

pub struct Network {
    pub graph: Graph,
    pub params: Vec<crate::model::ParamInit>,
}
impl Network {
    pub fn initialize(&self, session: &mut meganeura::Session, seed: u64) {
        let mut rng = crate::rng::Rng::new(seed);
        for p in &self.params {
            use crate::model::InitKind;
            let values: Vec<_> = match &p.kind {
                InitKind::Zeros => vec![0.0; p.len],
                InitKind::Ones => vec![1.0; p.len],
                InitKind::Values(v) => v.clone(),
                InitKind::Kaiming { fan_in } => (0..p.len)
                    .map(|_| rng.normal() * (2.0 / *fan_in as f32).sqrt())
                    .collect(),
            };
            session.set_parameter(&p.name, &values);
        }
    }
}
struct Builder {
    g: Graph,
    params: Vec<crate::model::ParamInit>,
    shared: BTreeMap<String, NodeId>,
}
impl Builder {
    fn parameter(&mut self, name: &str, out: u32, input: u32, k: u32, zero: bool) -> NodeId {
        if let Some(&id) = self.shared.get(name) {
            return id;
        }
        let len = (out * input * k * k) as usize;
        let id = self.g.parameter(name, &[len]);
        self.params.push(crate::model::ParamInit {
            name: name.into(),
            len,
            kind: if zero {
                crate::model::InitKind::Zeros
            } else {
                crate::model::InitKind::Kaiming {
                    fan_in: (input * k * k) as usize,
                }
            },
        });
        self.shared.insert(name.into(), id);
        id
    }
    fn conv(
        &mut self,
        x: NodeId,
        name: &str,
        shape: [u32; 3],
        out: u32,
        stride: u32,
        zero: bool,
    ) -> NodeId {
        let [input, w, h] = shape;
        let kernel = if zero { 1 } else { 3 };
        let weight = self.parameter(name, out, input, kernel, zero);
        self.g.conv2d(
            x,
            weight,
            1,
            input,
            h,
            w,
            out,
            kernel,
            kernel,
            stride,
            kernel / 2,
        )
    }
    fn block(&mut self, x: NodeId, name: &str, shape: [u32; 3]) -> NodeId {
        let a = self.g.silu(x);
        let a = self.conv(a, &format!("{name}.a"), shape, shape[0], 1, false);
        let a = self.g.silu(a);
        let a = self.conv(a, &format!("{name}.b"), shape, shape[0], 1, false);
        let scale = filled(&mut self.g, a, 0.1);
        let a = self.g.mul(a, scale);
        self.g.add(x, a)
    }
    fn core(&mut self, input: NodeId, low: [u32; 2], config: Config) -> NodeId {
        let [w, h] = low;
        let c = config.channels;
        let slots = config.scale.pow(2);
        let stem = self.conv(
            input,
            "adapter.observation",
            [(FEATURES as u32 + 6) * slots, w, h],
            c,
            1,
            false,
        );
        let a = self.block(stem, "core.level0", [c, w, h]);
        let b = self.conv(a, "core.down1", [c, w, h], 2 * c, 2, false);
        let b = self.block(b, "core.level1", [2 * c, w / 2, h / 2]);
        let d = self.conv(b, "core.down2", [2 * c, w / 2, h / 2], 4 * c, 2, false);
        let d = self.block(d, "core.level2", [4 * c, w / 4, h / 4]);
        let up = self.g.upsample_2x(d, 1, 4 * c, h / 4, w / 4);
        let up = self.g.concat(up, b, 1, 4 * c, 2 * c, w * h / 4);
        let up = self.conv(up, "core.up1", [6 * c, w / 2, h / 2], 2 * c, 1, false);
        let up = self.g.upsample_2x(up, 1, 2 * c, h / 2, w / 2);
        let up = self.g.concat(up, a, 1, 2 * c, c, w * h);
        let up = self.conv(up, "core.up0", [3 * c, w, h], c, 1, false);
        let up = self.g.silu(up);
        let logits = self.conv(
            up,
            "head.lobe_candidates",
            [c, w, h],
            2 * CANDIDATES as u32 * slots,
            1,
            true,
        );
        self.g.softplus(logits, 1.0)
    }
}
fn filled(g: &mut Graph, x: NodeId, value: f32) -> NodeId {
    let shape = g.node(x).ty.shape.clone();
    g.constant(vec![value; shape.iter().product()], &shape)
}
fn compress(g: &mut Graph, x: NodeId, e: f32) -> NodeId {
    let x = g.relu(x);
    let scale = filled(g, x, e);
    let v = g.mul(x, scale);
    let one = filled(g, v, 1.0);
    let den = g.add(v, one);
    g.div(v, den)
}
fn split(g: &mut Graph, mut x: NodeId, groups: u32, channels: u32, spatial: u32) -> Vec<NodeId> {
    let mut result = Vec::new();
    for i in 0..groups {
        if i + 1 == groups {
            result.push(x);
        } else {
            result.push(g.split_a(x, 1, channels, (groups - i - 1) * channels, spatial));
            x = g.split_b(x, 1, channels, (groups - i - 1) * channels, spatial);
        }
    }
    result
}
fn rgb_weights(g: &mut Graph, weight: NodeId, slots: u32, spatial: u32) -> NodeId {
    let w = split(g, weight, 2, slots, spatial);
    let d = g.concat(w[0], w[0], 1, slots, slots, spatial);
    let d = g.concat(d, w[0], 1, 2 * slots, slots, spatial);
    let s = g.concat(w[1], w[1], 1, slots, slots, spatial);
    let s = g.concat(s, w[1], 1, 2 * slots, slots, spatial);
    g.concat(d, s, 1, 3 * slots, 3 * slots, spatial)
}
fn warp(g: &mut Graph, image: NodeId, maps: &[(NodeId, NodeId); 4], n: usize) -> NodeId {
    let table = g.reshape(image, &[6 * n, 1]);
    let mut sum = None;
    for &(indices, coefficients) in maps {
        let tap = g.embedding(indices, table);
        let tap = g.reshape(tap, &[6 * n]);
        let tap = g.mul(tap, coefficients);
        sum = Some(match sum {
            None => tap,
            Some(s) => g.add(s, tap),
        });
    }
    sum.unwrap()
}
fn scaled_mse(g: &mut Graph, a: NodeId, b: NodeId, scale: NodeId) -> NodeId {
    let a = g.mul(a, scale);
    let b = g.mul(b, scale);
    g.mse_loss(a, b)
}

/// `unroll == 0` builds inference; positive values build a tied-weight training
/// graph. Geometry, rejection maps and moments are detached, radiance is not.
pub fn build(config: Config, low: [u32; 2], unroll: usize) -> Result<Network, String> {
    config.validate(low)?;
    if unroll > 8 {
        return Err("unroll must be at most eight".into());
    }
    let slots = config.scale.pow(2);
    let spatial = low[0] * low[1];
    let n = (slots * spatial) as usize;
    let mut b = Builder {
        g: Graph::new(),
        params: Vec::new(),
        shared: BTreeMap::new(),
    };
    let mut previous = None;
    let mut previous_target = None;
    let mut total_loss = None;
    for frame in 0..unroll.max(1) {
        let tag = format!("f{frame}");
        let features = b.g.input(&format!("{tag}.features"), &[FEATURES * n]);
        let maps = std::array::from_fn(|k| {
            if frame == 0 {
                (0, 0)
            } else {
                (
                    b.g.input_u32(&format!("{tag}.warp{k}"), &[6 * n]),
                    b.g.input(&format!("{tag}.coeff{k}"), &[6 * n]),
                )
            }
        });
        let history = match previous {
            Some(image) => warp(&mut b.g, image, &maps, n),
            None => b.g.input(&format!("{tag}.history"), &[6 * n]),
        };
        let hc = compress(&mut b.g, history, config.exposure);
        let input =
            b.g.concat(features, hc, 1, FEATURES as u32 * slots, 6 * slots, spatial);
        let multiplier = b.core(input, low, config);
        let prior = b.g.input(&format!("{tag}.prior"), &[CANDIDATES * 2 * n]);
        let weights = b.g.mul(prior, multiplier);
        let ws = split(&mut b.g, weights, CANDIDATES as u32, 2 * slots, spatial);
        let mut sum = ws[0];
        for &w in &ws[1..] {
            sum = b.g.add(sum, w);
        }
        let eps = filled(&mut b.g, sum, 1e-12);
        sum = b.g.add(sum, eps);
        let candidates = b.g.input(&format!("{tag}.candidates"), &[SCALES * 6 * n]);
        let mut candidates = split(&mut b.g, candidates, SCALES as u32, 6 * slots, spatial);
        candidates.push(history);
        let mut image = None;
        let mut normalized = None;
        let mut history_share = None;
        for k in 0..CANDIDATES {
            let w = b.g.div(ws[k], sum);
            if k == SCALES {
                history_share = Some(w);
            }
            normalized = Some(match normalized {
                None => w,
                Some(a) => {
                    b.g.concat(a, w, 1, 2 * k as u32 * slots, 2 * slots, spatial)
                }
            });
            let wrgb = rgb_weights(&mut b.g, w, slots, spatial);
            let part = b.g.mul(wrgb, candidates[k]);
            image = Some(match image {
                None => part,
                Some(a) => b.g.add(a, part),
            });
        }
        let image = image.unwrap();
        previous = Some(image);
        if unroll == 0 {
            b.g.set_outputs(vec![image, normalized.unwrap()]);
            break;
        }
        let target = b.g.input(&format!("{tag}.target"), &[6 * n]);
        let scale = b.g.input(&format!("{tag}.loss_scale"), &[6 * n]);
        let encoded = compress(&mut b.g, image, config.exposure);
        let encoded_target = compress(&mut b.g, target, config.exposure);
        let mut loss = b.g.mse_loss(encoded, encoded_target);
        let physical = scaled_mse(&mut b.g, image, target, scale);
        let weight = b.g.scalar(0.1);
        let physical = b.g.mul(physical, weight);
        loss = b.g.add(loss, physical);
        // Low-frequency physical lobe error, no loss-time change to the estimator.
        let error = b.g.neg(target);
        let error = b.g.add(image, error);
        let error = b.g.mul(error, scale);
        let block = 4;
        let channels = 6 * slots;
        let mut kernel = vec![0.0; (channels * channels * block * block) as usize];
        for c in 0..channels as usize {
            let start = (c * channels as usize + c) * (block * block) as usize;
            kernel[start..start + (block * block) as usize].fill(1.0 / (block * block) as f32);
        }
        let k =
            b.g.constant(kernel, &[(channels * channels * block * block) as usize]);
        let avg = b.g.conv2d(
            error, k, 1, channels, low[1], low[0], channels, block, block, block, 0,
        );
        let zero = b.g.constant(
            vec![0.0; (channels * spatial / (block * block)) as usize],
            &[(channels * spatial / (block * block)) as usize],
        );
        let lf = b.g.mse_loss(avg, zero);
        let weight = b.g.scalar(0.05);
        let lf = b.g.mul(lf, weight);
        loss = b.g.add(loss, lf);
        let confidence = b.g.input(&format!("{tag}.confidence"), &[2 * n]);
        let mask = b.g.input(&format!("{tag}.confidence_mask"), &[2 * n]);
        let cl = scaled_mse(&mut b.g, history_share.unwrap(), confidence, mask);
        let weight = b.g.scalar(0.01);
        let cl = b.g.mul(cl, weight);
        loss = b.g.add(loss, cl);
        if let Some(old_target) = previous_target {
            let reference_history = warp(&mut b.g, old_target, &maps, n);
            let neg = b.g.neg(history);
            let change = b.g.add(image, neg);
            let neg = b.g.neg(reference_history);
            let expected = b.g.add(target, neg);
            let valid = b.g.input(&format!("{tag}.temporal_mask"), &[6 * n]);
            let masked = b.g.mul(valid, scale);
            let tl = scaled_mse(&mut b.g, change, expected, masked);
            let weight = b.g.scalar(0.01);
            let tl = b.g.mul(tl, weight);
            loss = b.g.add(loss, tl);
        }
        previous_target = Some(target);
        total_loss = Some(match total_loss {
            None => loss,
            Some(s) => b.g.add(s, loss),
        });
    }
    if let Some(loss) = total_loss {
        let divisor = b.g.scalar(unroll as f32);
        let loss = b.g.div(loss, divisor);
        b.g.set_outputs(vec![loss]);
    }
    Ok(Network {
        graph: b.g,
        params: b.params,
    })
}

/// Upload one training slot. Targets never enter the inference feature tensor.
pub fn feed(
    session: &mut meganeura::Session,
    tag: &str,
    p: &super::cpu::Prepared,
    target: &Target,
    frame: usize,
) {
    session.set_input(&format!("{tag}.features"), &p.features);
    session.set_input(&format!("{tag}.candidates"), &p.candidates);
    session.set_input(&format!("{tag}.prior"), &p.prior);
    if frame == 0 {
        session.set_input(&format!("{tag}.history"), &p.history);
    } else {
        for k in 0..4 {
            session.set_input_u32(&format!("{tag}.warp{k}"), &p.indices[k]);
            session.set_input(&format!("{tag}.coeff{k}"), &p.coefficients[k]);
        }
    }
    session.set_input(&format!("{tag}.target"), &target.lobes);
    session.set_input(
        &format!("{tag}.loss_scale"),
        &target
            .lobes
            .iter()
            .map(|v| 1.0 / (0.1 + v))
            .collect::<Vec<_>>(),
    );
    let (labels, mask) = cpu::confidence(p, target);
    session.set_input(&format!("{tag}.confidence"), &labels);
    session.set_input(&format!("{tag}.confidence_mask"), &mask);
    if frame != 0 {
        let n = p.history.len() / 6;
        let mut mask = vec![0.0; 6 * n];
        for c in 0..6 {
            mask[c * n..(c + 1) * n].copy_from_slice(&p.validity[c / 3 * n..(c / 3 + 1) * n]);
        }
        session.set_input(&format!("{tag}.temporal_mask"), &mask);
    }
}
