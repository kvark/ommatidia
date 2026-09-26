//! Shared local spatial core with lobe-specific reconstruction heads.
//! Short unrolls differentiate through radiance reprojection, not camera geometry.
use super::*;
use meganeura::{Graph, NodeId};

use crate::neural::Builder;
pub use crate::neural::Network;

/// One objective: RGB, identifiable radiance lobes, energy, structure, and time.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct LossWeights {
    pub compressed: f32,
    pub physical: f32,
    pub low_frequency: f32,
    pub temporal: f32,
    pub lobes: f32,
}
impl Default for LossWeights {
    fn default() -> Self {
        Self {
            compressed: 1.0,
            physical: 0.005,
            low_frequency: 0.01,
            temporal: 0.02,
            lobes: 0.5,
        }
    }
}
impl LossWeights {
    pub fn values(self) -> [f32; 5] {
        [
            self.compressed,
            self.physical,
            self.low_frequency,
            self.temporal,
            self.lobes,
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

fn decode(
    g: &mut Graph,
    spatial: NodeId,
    history: NodeId,
    history_weight: NodeId,
    residual: NodeId,
    exposure: f32,
) -> NodeId {
    // `spatial` already contains its (1-h) weight. Correct only this incoming
    // observation, then accumulate it; a post-blend residual is added again
    // every frame and amplifies a stationary bias by the history length.
    let one = filled(g, history_weight, 1.0);
    let negative = g.neg(history_weight);
    let incoming_weight = g.add(one, negative);
    let floor = filled(g, spatial, 1.0 / exposure);
    let floor = g.mul(incoming_weight, floor);
    let amplitude = g.add(spatial, floor);
    let scale = filled(g, spatial, 0.1);
    let amplitude = g.mul(amplitude, scale);
    let correction = g.mul(residual, amplitude);
    let corrected = g.add(spatial, correction);
    let corrected = g.relu(corrected);
    let retained = g.mul(history_weight, history);
    g.add(corrected, retained)
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
    let mut b = Builder::new();
    let objective_weights: Option<[NodeId; 5]> = (unroll > 0).then(|| {
        let input = b.g.input("loss.weights", &[5]);
        split(&mut b.g, input, 5, 1, 1).try_into().unwrap()
    });
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
        let residual = b.conv(
            features,
            "head.radiance",
            [config.channels, low[0], low[1]],
            6 * slots,
            1,
            true,
        );
        let prior = b.g.input(&format!("{tag}.prior"), &[CANDIDATES * 2 * n]);
        let ws = split(&mut b.g, prior, CANDIDATES as u32, 2 * slots, spatial);
        let candidates = b.g.input(&format!("{tag}.candidates"), &[SCALES * 6 * n]);
        let candidates = split(&mut b.g, candidates, SCALES as u32, 6 * slots, spatial);
        let mut image = None;
        for k in 0..SCALES {
            let w = ws[k];
            let wrgb = rgb_weights(&mut b.g, w, slots, spatial);
            let part = b.g.mul(wrgb, candidates[k]);
            image = Some(match image {
                None => part,
                Some(a) => b.g.add(a, part),
            });
        }
        let history_weight = rgb_weights(&mut b.g, ws[SCALES], slots, spatial);
        let image = decode(
            &mut b.g,
            image.unwrap(),
            history,
            history_weight,
            residual,
            config.exposure,
        );
        previous = Some(image);
        if unroll == 0 {
            b.g.set_outputs(vec![image]);
            break;
        }
        let target = b.g.input(&format!("{tag}.target"), &[6 * n]);
        let material = b.g.input(&format!("{tag}.rgb.albedo"), &[3 * n]);
        let emission = b.g.input(&format!("{tag}.rgb.emission"), &[3 * n]);
        let lobes = split(&mut b.g, image, 2, 3 * slots, spatial);
        let diffuse = b.g.mul(lobes[0], material);
        let rgb = b.g.add(diffuse, lobes[1]);
        let spatial_image = b.g.add(rgb, emission);
        let spatial_target = b.g.input(&format!("{tag}.rgb.target"), &[3 * n]);
        let spatial_scale = filled(&mut b.g, spatial_image, config.exposure);
        let color_channels = 3;
        let encoded = compress(&mut b.g, spatial_image, config.exposure);
        let encoded_target = compress(&mut b.g, spatial_target, config.exposure);
        let [
            compressed_weight,
            physical_weight,
            low_frequency_weight,
            temporal_weight,
            lobe_weight,
        ] = objective_weights.unwrap();
        let loss = b.g.mse_loss(encoded, encoded_target);
        let mut loss = b.g.mul(loss, compressed_weight);
        // RGB alone cannot identify diffuse/specular energy: opposite lobe
        // errors can cancel after material composition but pollute recurrence.
        let encoded_lobes = compress(&mut b.g, image, config.exposure);
        let encoded_target_lobes = compress(&mut b.g, target, config.exposure);
        let lobe_loss = b.g.mse_loss(encoded_lobes, encoded_target_lobes);
        let lobe_loss = b.g.mul(lobe_loss, lobe_weight);
        loss = b.g.add(loss, lobe_loss);
        let physical = scaled_mse(&mut b.g, spatial_image, spatial_target, spatial_scale);
        let physical = b.g.mul(physical, physical_weight);
        loss = b.g.add(loss, physical);
        // Coarse linear error catches broad energy drift.
        let error = b.g.neg(spatial_target);
        let error = b.g.add(spatial_image, error);
        let error = b.g.mul(error, spatial_scale);
        let block = 4;
        let channels = color_channels * slots;
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
        if let Some(old_target) = previous_target {
            let reference_history = warp(&mut b.g, old_target, &maps, n);
            let current = compress(&mut b.g, image, config.exposure);
            let old = compress(&mut b.g, history, config.exposure);
            let truth = compress(&mut b.g, target, config.exposure);
            let old_truth = compress(&mut b.g, reference_history, config.exposure);
            let neg = b.g.neg(old);
            let change = b.g.add(current, neg);
            let neg = b.g.neg(old_truth);
            let expected = b.g.add(truth, neg);
            let valid = b.g.input(&format!("{tag}.temporal_mask"), &[6 * n]);
            let tl = scaled_mse(&mut b.g, change, expected, valid);
            let tl = b.g.mul(tl, temporal_weight);
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
    if frame != 0 {
        let n = p.history.len() / 6;
        let mut mask = vec![0.0; 6 * n];
        for c in 0..6 {
            mask[c * n..(c + 1) * n].copy_from_slice(&p.validity[c / 3 * n..(c / 3 + 1) * n]);
        }
        session.set_input(&format!("{tag}.temporal_mask"), &mask);
    }
}

/// Material inputs are exact observations already available to the runtime.
/// Only target RGB is privileged. All buffers use the native subpixel packing.
pub fn feed_rgb(
    session: &mut meganeura::Session,
    tag: &str,
    frame: &Frame,
    target: &Target,
    config: Config,
) {
    let n = frame.surfaces.len();
    assert_eq!(target.rgb.len(), 3 * n);
    let width = (frame.low[0] * config.scale) as usize;
    let mut albedo = vec![0.0; 3 * n];
    let mut emission = vec![0.0; 3 * n];
    let mut rgb = vec![0.0; 3 * n];
    for (p, s) in frame.surfaces.iter().enumerate() {
        for c in 0..3 {
            let index = config.index(frame.low, c, p % width, p / width);
            albedo[index] = s.albedo_roughness[c];
            emission[index] = s.emission[c];
            rgb[index] = target.rgb[p * 3 + c];
        }
    }
    session.set_input(&format!("{tag}.rgb.albedo"), &albedo);
    session.set_input(&format!("{tag}.rgb.emission"), &emission);
    session.set_input(&format!("{tag}.rgb.target"), &rgb);
}

#[cfg(test)]
mod tests {
    use super::*;
    use meganeura::reference::{Feeds, evaluate_outputs};

    fn decoder() -> Graph {
        let mut g = Graph::new();
        let spatial = g.input("spatial", &[1]);
        let history = g.input("history", &[1]);
        let h = g.input("h", &[1]);
        let residual = g.input("residual", &[1]);
        let out = decode(&mut g, spatial, history, h, residual, 1.0);
        g.set_outputs(vec![out]);
        g
    }

    #[test]
    fn residual_is_accumulated_once_in_a_long_stationary_history() {
        let g = decoder();
        for cap in [1.0_f32, 8.0, 32.0] {
            let mut previous = 0.0_f32;
            let mut feeds = Feeds::new();
            for step in 1..=256 {
                let h = 1.0 - 1.0 / (step as f32).min(cap);
                feeds.set("spatial", &[1.0 - h]);
                feeds.set("history", &[previous]);
                feeds.set("h", &[h]);
                feeds.set("residual", &[1.0]);
                previous = evaluate_outputs(&g, &feeds).unwrap()[0].data[0] as f32;
                assert!(
                    (previous - 1.2).abs() < 1e-5,
                    "cap {cap}, step {step}: {previous}"
                );
            }
        }
    }

    #[test]
    fn clamping_an_incoming_estimate_does_not_erase_retained_history() {
        let g = decoder();
        let mut feeds = Feeds::new();
        feeds.set("spatial", &[0.25]);
        feeds.set("history", &[2.0]);
        feeds.set("h", &[0.75]);
        feeds.set("residual", &[-100.0]);
        assert_eq!(evaluate_outputs(&g, &feeds).unwrap()[0].data[0], 1.5);
        feeds.set("residual", &[0.0]);
        assert_eq!(evaluate_outputs(&g, &feeds).unwrap()[0].data[0], 1.75);
    }
}
