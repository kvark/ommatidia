//! Shared local spatial core with lobe-specific reconstruction heads.
//! Short unrolls differentiate through radiance reprojection, not camera geometry.
use super::*;
use meganeura::{Graph, NodeId};

use crate::neural::Builder;
pub use crate::neural::Network;

/// Training-only objective coefficients. Defaults preserve the prior objective.
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
        Self {
            compressed: 1.0,
            physical: 0.1,
            low_frequency: 0.05,
            confidence: 0.01,
            temporal: 0.01,
        }
    }
}
impl LossWeights {
    fn values(self) -> [f32; 5] {
        [
            self.compressed,
            self.physical,
            self.low_frequency,
            self.confidence,
            self.temporal,
        ]
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
    build_projected(config, low, unroll, false)
}

/// Target-only projected-colour supervision, without new inference parameters.
/// A zero coefficient retains the paired training graph.
pub fn build_projected(
    config: Config,
    low: [u32; 2],
    unroll: usize,
    projected: bool,
) -> Result<Network, String> {
    if projected && unroll == 0 {
        return Err("projected supervision is training-only".into());
    }
    config.validate(low)?;
    if unroll > 8 {
        return Err("unroll must be at most eight".into());
    }
    let slots = config.scale.pow(2);
    let spatial = low[0] * low[1];
    let n = (slots * spatial) as usize;
    let mut b = Builder::new();
    let objective_weights: Option<[NodeId; 5]> = (unroll > 0).then(|| {
        let input = b.g.input("loss.weights", &[5]);
        split(&mut b.g, input, 5, 1, 1).try_into().unwrap()
    });
    let projected_weight = projected.then(|| b.g.input("loss.projected_weight", &[1]));
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
        let features = b.encode(
            input,
            "adapter.observation",
            (FEATURES as u32 + 6) * slots,
            low,
            config.channels,
        );
        let logits = b.conv(
            features,
            "head.lobe_candidates",
            [config.channels, low[0], low[1]],
            2 * CANDIDATES as u32 * slots,
            1,
            true,
        );
        let multiplier = b.g.softplus(logits, 1.0);
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
        let [
            compressed_weight,
            physical_weight,
            low_frequency_weight,
            confidence_weight,
            temporal_weight,
        ] = objective_weights.unwrap();
        let loss = b.g.mse_loss(encoded, encoded_target);
        let mut loss = b.g.mul(loss, compressed_weight);
        let physical = scaled_mse(&mut b.g, image, target, scale);
        let physical = b.g.mul(physical, physical_weight);
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
        let lf = b.g.mul(lf, low_frequency_weight);
        loss = b.g.add(loss, lf);
        let confidence = b.g.input(&format!("{tag}.confidence"), &[2 * n]);
        let mask = b.g.input(&format!("{tag}.confidence_mask"), &[2 * n]);
        let cl = scaled_mse(&mut b.g, history_share.unwrap(), confidence, mask);
        let cl = b.g.mul(cl, confidence_weight);
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
            let tl = b.g.mul(tl, temporal_weight);
            loss = b.g.add(loss, tl);
        }
        if let Some(weight) = projected_weight {
            let teacher = b.g.input(&format!("{tag}.projected"), &[6 * n]);
            let fixed_scale = filled(&mut b.g, image, config.exposure);
            let auxiliary = scaled_mse(&mut b.g, image, teacher, fixed_scale);
            let auxiliary = b.g.mul(auxiliary, weight);
            loss = b.g.add(loss, auxiliary);
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
    LossWeights::default().feed(session);
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
