//! One biased U-Net predicts spatial radiance, history gates and recurrent latent.
//! Reprojection of both radiance and latent is differentiated through the unroll.
use super::*;
use meganeura::{Graph, NodeId};

use crate::neural::Builder;
pub use crate::neural::Network;

/// One objective: RGB, identifiable radiance lobes, energy, structure, and time.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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
fn broadcast(g: &mut Graph, scalar: NodeId, len: usize) -> NodeId {
    let x = g.reshape(scalar, &[1, 1]);
    let x = g.broadcast_inner(x, len);
    g.reshape(x, &[len])
}
fn compress(g: &mut Graph, x: NodeId, e: NodeId) -> NodeId {
    let x = g.relu(x);
    let scale = broadcast(g, e, g.node(x).ty.num_elements());
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
fn warp(g: &mut Graph, image: NodeId, maps: &[(NodeId, NodeId); 4], len: usize) -> NodeId {
    let table = g.reshape(image, &[len, 1]);
    let mut sum = None;
    for &(indices, coefficients) in maps {
        let tap = g.embedding(indices, table);
        let tap = g.reshape(tap, &[len]);
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
    z: NodeId,
    history: NodeId,
    alpha_rgb: NodeId,
    exposure: NodeId,
) -> NodeId {
    // Pinned Meganeura's Clamp is inference-only. Explicit piecewise selection
    // has the same value and derivative (away from the two nondifferentiable
    // endpoints). Greater has zero derivative. Unlike subtracting nested ReLUs,
    // this preserves logits inside the interval bit-for-bit in f32.
    let low = filled(g, z, -16.0);
    let high = filled(g, z, 11.0);
    let below = g.greater(low, z);
    let above = g.greater(z, high);
    let outside = g.add(below, above);
    let neg = g.neg(outside);
    let one = filled(g, z, 1.0);
    let inside = g.add(one, neg);
    let interior = g.mul(inside, z);
    let bottom = g.mul(below, low);
    let top = g.mul(above, high);
    let boundary = g.add(bottom, top);
    let z = g.add(interior, boundary);
    let spatial = g.exp(z);
    let scale = broadcast(g, exposure, g.node(spatial).ty.num_elements());
    let spatial = g.div(spatial, scale);
    let one = filled(g, alpha_rgb, 1.0);
    let negative = g.neg(alpha_rgb);
    let incoming_weight = g.add(one, negative);
    let incoming = g.mul(incoming_weight, spatial);
    let retained = g.mul(alpha_rgb, history);
    g.add(incoming, retained)
}

/// `unroll == 0` builds inference; positive values build a tied-weight training
/// graph. Geometry is detached, radiance and latent are not. Outputs are
/// `[lobes, latent, state]` for inference and `[loss, lobes, latent, state]`
/// for training. Meganeura differentiates only output zero (the scalar loss).
pub fn build(config: Config, low: [u32; 2], unroll: usize) -> Result<Network, String> {
    build_with_debug(config, low, unroll, false)
}

/// Append the final frame's two alpha planes for diagnostics, never as an input.
pub fn build_with_debug(
    config: Config,
    low: [u32; 2],
    unroll: usize,
    debug_alpha: bool,
) -> Result<Network, String> {
    build_impl(config, low, unroll, debug_alpha, 0)
}

/// Same model with a loss-only exclusion margin measured in HR pixels.
pub fn build_training(
    config: Config,
    low: [u32; 2],
    unroll: usize,
    margin: u32,
) -> Result<Network, String> {
    if unroll == 0 {
        return Err("training requires a positive unroll".into());
    }
    build_impl(config, low, unroll, false, margin)
}

/// Binary packed mask, shared by the objective and its independent tests.
pub fn loss_mask(
    config: Config,
    low: [u32; 2],
    channels: usize,
    margin: u32,
) -> Result<Vec<f32>, String> {
    let high = low.map(|v| v * config.scale);
    if high.iter().any(|v| margin >= v.div_ceil(2)) {
        return Err("loss margin removes the whole crop".into());
    }
    let mut mask = vec![0.0; (high[0] * high[1]) as usize * channels];
    for c in 0..channels {
        for y in margin..high[1] - margin {
            for x in margin..high[0] - margin {
                mask[config.index(low, c, x as usize, y as usize)] = 1.0;
            }
        }
    }
    Ok(mask)
}

fn build_impl(
    config: Config,
    low: [u32; 2],
    unroll: usize,
    debug_alpha: bool,
    margin: u32,
) -> Result<Network, String> {
    config.validate(low)?;
    if unroll > 8 {
        return Err("unroll must be at most eight".into());
    }
    let slots = config.scale.pow(2);
    let spatial = low[0] * low[1];
    let n = (slots * spatial) as usize;
    let state_channels = config.state_channels() as u32;
    let state_len = config.state_channels() * n;
    let mut b = Builder::new();
    let masks = if unroll > 0 && margin > 0 {
        let rgb = loss_mask(config, low, 3, margin)?;
        let norm = (rgb.len() as f32 / rgb.iter().sum::<f32>()).sqrt();
        let rgb_normalized =
            b.g.constant(rgb.iter().map(|v| v * norm).collect(), &[3 * n]);
        let lobe = loss_mask(config, low, 6, margin)?;
        let lobe_normalized =
            b.g.constant(lobe.iter().map(|v| v * norm).collect(), &[6 * n]);
        let mut coarse = Vec::new();
        for c in 0..3 * slots as usize {
            for y in 0..low[1] as usize / 4 {
                for x in 0..low[0] as usize / 4 {
                    let mut count = 0.0;
                    for dy in 0..4 {
                        for dx in 0..4 {
                            count += rgb
                                [(c * low[1] as usize + 4 * y + dy) * low[0] as usize + 4 * x + dx];
                        }
                    }
                    coarse.push(if count > 0.0 { 16.0 / count } else { 0.0 });
                }
            }
        }
        let norm =
            (coarse.len() as f32 / coarse.iter().filter(|v| **v > 0.0).count() as f32).sqrt();
        let coarse_len = coarse.len();
        let coarse = b.g.constant(
            coarse.into_iter().map(|v| v * norm).collect(),
            &[coarse_len],
        );
        let binary = b.g.constant(rgb, &[3 * n]);
        Some((rgb_normalized, lobe_normalized, binary, coarse))
    } else {
        None
    };
    let objective_weights: Option<[NodeId; 5]> = (unroll > 0).then(|| {
        let input = b.g.input("loss.weights", &[5]);
        split(&mut b.g, input, 5, 1, 1).try_into().unwrap()
    });
    let mut previous = None;
    let mut previous_target = None;
    let mut total_loss = None;
    let mut final_outputs = Vec::new();
    for frame in 0..unroll.max(1) {
        let tag = format!("f{frame}");
        let features = b.g.input(
            &format!("{tag}.features"),
            &[config.observation_channels() * spatial as usize],
        );
        let exposure = b.g.input(&format!("{tag}.exposure"), &[1]);
        let valid = b.g.input(&format!("{tag}.valid"), &[n]);
        let metadata = b.g.input(&format!("{tag}.metadata"), &[7 * n]);
        let maps = std::array::from_fn(|k| {
            (
                b.g.input_u32(&format!("{tag}.warp{k}"), &[state_len]),
                b.g.input(&format!("{tag}.coeff{k}"), &[state_len]),
            )
        });
        let previous_state = match previous {
            Some(state) => state,
            None => b.g.input(&format!("{tag}.history"), &[state_len]),
        };
        let warped = warp(&mut b.g, previous_state, &maps, state_len);
        let history =
            b.g.split_a(warped, 1, 6 * slots, (state_channels - 6) * slots, spatial);
        let rest =
            b.g.split_b(warped, 1, 6 * slots, (state_channels - 6) * slots, spatial);
        let state_features = b.g.split_a(
            rest,
            1,
            (config.latent_channels + 4) * slots,
            3 * slots,
            spatial,
        );
        let hc = compress(&mut b.g, history, exposure);
        let recurrent = b.g.concat(
            hc,
            state_features,
            1,
            6 * slots,
            (config.latent_channels + 4) * slots,
            spatial,
        );
        let recurrent = b.g.concat(
            recurrent,
            valid,
            1,
            (config.latent_channels + 10) * slots,
            slots,
            spatial,
        );
        let input = b.g.concat(
            features,
            recurrent,
            1,
            config.observation_channels() as u32,
            (config.latent_channels + 11) * slots,
            spatial,
        );
        let features = b.encode(
            input,
            "adapter.observation",
            config.input_channels() as u32,
            low,
            config.channels,
            config.levels,
        );
        let shape = [config.channels, low[0], low[1]];
        let z = b.head(features, "head.radiance", shape, 6 * slots, 0.0);
        let a = b.head(features, "head.alpha", shape, 2 * slots, 4.0_f32.ln());
        let s = b.head(
            features,
            "head.latent",
            shape,
            config.latent_channels * slots,
            0.0,
        );
        let latent = b.g.tanh(s);
        let alpha = b.g.sigmoid(a);
        let valid2 = b.g.concat(valid, valid, 1, slots, slots, spatial);
        let alpha = b.g.mul(alpha, valid2);
        let alpha_rgb = rgb_weights(&mut b.g, alpha, slots, spatial);
        let image = decode(&mut b.g, z, history, alpha_rgb, exposure);
        let state = b.g.concat(
            image,
            latent,
            1,
            6 * slots,
            config.latent_channels * slots,
            spatial,
        );
        let state = b.g.concat(
            state,
            metadata,
            1,
            (6 + config.latent_channels) * slots,
            7 * slots,
            spatial,
        );
        previous = Some(state);
        final_outputs = vec![image, latent, state];
        if debug_alpha {
            final_outputs.push(alpha);
        }
        if unroll == 0 {
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
        let spatial_scale = broadcast(&mut b.g, exposure, 3 * n);
        let color_channels = 3;
        let encoded = compress(&mut b.g, spatial_image, exposure);
        let encoded_target = compress(&mut b.g, spatial_target, exposure);
        let [
            compressed_weight,
            physical_weight,
            low_frequency_weight,
            temporal_weight,
            lobe_weight,
        ] = objective_weights.unwrap();
        let loss = match masks {
            Some((rgb, ..)) => scaled_mse(&mut b.g, encoded, encoded_target, rgb),
            None => b.g.mse_loss(encoded, encoded_target),
        };
        let mut loss = b.g.mul(loss, compressed_weight);
        // RGB alone cannot identify diffuse/specular energy: opposite lobe
        // errors can cancel after material composition but pollute recurrence.
        let encoded_lobes = compress(&mut b.g, image, exposure);
        let encoded_target_lobes = compress(&mut b.g, target, exposure);
        let lobe_loss = match masks {
            Some((_, lobe, ..)) => scaled_mse(&mut b.g, encoded_lobes, encoded_target_lobes, lobe),
            None => b.g.mse_loss(encoded_lobes, encoded_target_lobes),
        };
        let lobe_loss = b.g.mul(lobe_loss, lobe_weight);
        loss = b.g.add(loss, lobe_loss);
        let physical_scale = match masks {
            Some((rgb, ..)) => b.g.mul(spatial_scale, rgb),
            None => spatial_scale,
        };
        let physical = scaled_mse(&mut b.g, spatial_image, spatial_target, physical_scale);
        let physical = b.g.mul(physical, physical_weight);
        loss = b.g.add(loss, physical);
        // Coarse linear error catches broad energy drift.
        let error = b.g.neg(spatial_target);
        let error = b.g.add(spatial_image, error);
        let error = b.g.mul(error, spatial_scale);
        let error = match masks {
            Some((_, _, mask, _)) => b.g.mul(error, mask),
            None => error,
        };
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
        let lf = match masks {
            Some((_, _, _, coarse)) => scaled_mse(&mut b.g, avg, zero, coarse),
            None => b.g.mse_loss(avg, zero),
        };
        let lf = b.g.mul(lf, low_frequency_weight);
        loss = b.g.add(loss, lf);
        if let Some(old_target) = previous_target {
            let reference_history = warp(&mut b.g, old_target, &maps, state_len);
            let reference_history = b.g.split_a(
                reference_history,
                1,
                6 * slots,
                (state_channels - 6) * slots,
                spatial,
            );
            let current = compress(&mut b.g, image, exposure);
            let old = compress(&mut b.g, history, exposure);
            let truth = compress(&mut b.g, target, exposure);
            let old_truth = compress(&mut b.g, reference_history, exposure);
            let neg = b.g.neg(old);
            let change = b.g.add(current, neg);
            let neg = b.g.neg(old_truth);
            let expected = b.g.add(truth, neg);
            let valid_rgb = rgb_weights(&mut b.g, valid2, slots, spatial);
            let valid_rgb = match masks {
                Some((_, lobe, ..)) => b.g.mul(valid_rgb, lobe),
                None => valid_rgb,
            };
            let tl = scaled_mse(&mut b.g, change, expected, valid_rgb);
            let tl = b.g.mul(tl, temporal_weight);
            loss = b.g.add(loss, tl);
        }
        let zeros =
            b.g.constant(vec![0.0; state_len - 6 * n], &[state_len - 6 * n]);
        previous_target = Some(b.g.concat(
            target,
            zeros,
            1,
            6 * slots,
            (state_channels - 6) * slots,
            spatial,
        ));
        total_loss = Some(match total_loss {
            None => loss,
            Some(s) => b.g.add(s, loss),
        });
    }
    if let Some(loss) = total_loss {
        let divisor = b.g.scalar(unroll as f32);
        let loss = b.g.div(loss, divisor);
        final_outputs.insert(0, loss);
    }
    b.g.set_outputs(final_outputs);
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
    session.set_input(&format!("{tag}.exposure"), &[p.exposure]);
    session.set_input(&format!("{tag}.valid"), &p.validity);
    session.set_input(&format!("{tag}.metadata"), &p.metadata);
    if frame == 0 {
        session.set_input(&format!("{tag}.history"), &p.history);
    }
    for k in 0..4 {
        session.set_input_u32(&format!("{tag}.warp{k}"), &p.indices[k]);
        session.set_input(&format!("{tag}.coeff{k}"), &p.coefficients[k]);
    }
    session.set_input(&format!("{tag}.target"), &target.lobes);
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
        let z = g.input("z", &[1]);
        let history = g.input("history", &[1]);
        let alpha = g.input("alpha", &[1]);
        let exposure = g.input("exposure", &[1]);
        let out = decode(&mut g, z, history, alpha, exposure);
        g.set_outputs(vec![out]);
        g
    }

    #[test]
    fn direct_decoder_matches_scalar_and_exposure_units() {
        let g = decoder();
        for alpha in [0.0_f32, 0.8, 1.0] {
            for k in [1.0_f32, 8.0] {
                let mut f = Feeds::new();
                f.set("z", &[0.25]);
                f.set("history", &[3.0 * k]);
                f.set("alpha", &[alpha]);
                f.set("exposure", &[2.0 / k]);
                let actual = evaluate_outputs(&g, &f).unwrap()[0].data[0];
                let expected =
                    k as f64 * ((1.0 - alpha as f64) * 0.25_f64.exp() / 2.0 + alpha as f64 * 3.0);
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn log_clamp_stays_bounded_and_has_the_piecewise_derivative() {
        let mut g = Graph::new();
        let z = g.parameter("z", &[1]);
        let zero = g.constant(vec![0.0], &[1]);
        let one = g.constant(vec![1.0], &[1]);
        let out = decode(&mut g, z, zero, zero, one);
        g.set_outputs(vec![out]);
        let backward = meganeura::autodiff::differentiate(&g);
        for value in [-1e20_f32, -17.0, -15.0, 0.25, 10.0, 12.0, 1e20] {
            let mut feeds = Feeds::new();
            feeds.set("z", &[value]);
            let got = evaluate_outputs(&backward, &feeds).unwrap();
            let want = f64::from(value).clamp(-16.0, 11.0).exp();
            assert!((got[0].data[0] - want).abs() <= 1e-12 * want);
            let grad = if (-16.0..11.0).contains(&value) {
                want
            } else {
                0.0
            };
            assert!((got[1].data[0] - grad).abs() <= 1e-12 * want);
        }
    }
}
