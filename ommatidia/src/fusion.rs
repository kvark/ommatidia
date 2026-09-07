//! Versioned candidate-aware temporal fusion.
//!
//! Compression is a feature/loss encoding, not an averaging space. Legacy
//! checkpoints keep their exact contract; new modes interpolate and blend
//! physical (possibly demodulated) radiance. Candidate-aware gates are small
//! dynamic affine classifiers, evaluated after the actual candidates exist.

use meganeura::{Graph, NodeId};
use serde::{Deserialize, Serialize};

use crate::transform;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum Mode {
    /// Historical compressed-space interpolation and positive-odds gates.
    #[default]
    Legacy = 0,
    /// Physical interpolation/blending with the historical positive-odds gates.
    Linear = 1,
    /// Physical fusion with signed, candidate-conditioned gate coefficients.
    CandidateAware = 2,
}

impl Mode {
    pub fn gate_parameters(self) -> u32 {
        match self {
            Self::CandidateAware => 4,
            _ => 1,
        }
    }

    pub fn is_linear(self) -> bool {
        self != Self::Legacy
    }
}

/// Features use compressed candidates for bounded dynamic range. Missing
/// history has no disagreement feature: the stored zero is not black evidence.
pub fn features(current: [f32; 3], guide: [f32; 3], history: [f32; 3], valid: f32) -> [f32; 4] {
    let distance = |a: [f32; 3], b: [f32; 3]| {
        a.into_iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            / 3.0
    };
    [
        1.0,
        distance(current, guide),
        valid * distance(current, history),
        valid * distance(guide, history),
    ]
}

pub fn candidate_gate(coefficients: [f32; 4], features: [f32; 4]) -> f32 {
    let score: f32 = coefficients
        .into_iter()
        .zip(features)
        .map(|(a, b)| a * b)
        .sum();
    0.5 * (0.5 * score).tanh() + 0.5
}

/// Scale gate odds without mistaking signed coefficients for probabilities.
/// Layout is feature-major, then subpixel-major, then spatial position.
pub fn calibrate(values: &mut [f32], mode: Mode, slots: usize, spatial: usize, scale: f32) {
    assert!(scale.is_finite() && scale >= 0.0);
    assert_eq!(
        values.len(),
        mode.gate_parameters() as usize * slots * spatial
    );
    if mode == Mode::CandidateAware {
        if scale == 0.0 {
            values.fill(0.0);
            values[..slots * spatial].fill(-80.0);
        } else {
            for bias in &mut values[..slots * spatial] {
                *bias += scale.ln();
            }
        }
    } else {
        values.iter_mut().for_each(|value| *value *= scale);
    }
}

/// Identifiable least-squares history target for a detached candidate pair.
/// This is a target-generation primitive, not a trained confidence model.
#[derive(Clone, Copy, Debug)]
pub struct ConfidenceTarget {
    pub history_share: f32,
    /// Squared linear RGB separation; zero separation supplies no supervision.
    pub separation: f32,
}

/// Minimize ||(1-h) current + h history - reference||^2 for h in [0, 1].
/// Invalid and indistinguishable candidates are excluded, not labelled as
/// trustworthy/untrustworthy history. Inputs must be finite linear radiance.
pub fn confidence_target(
    current: [f32; 3],
    history: [f32; 3],
    reference: [f32; 3],
    valid: bool,
) -> Option<ConfidenceTarget> {
    if !valid
        || !current
            .into_iter()
            .chain(history)
            .chain(reference)
            .all(f32::is_finite)
    {
        return None;
    }
    // f64 avoids overflow/cancellation while constructing offline HDR labels.
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for c in 0..3 {
        let difference = history[c] as f64 - current[c] as f64;
        numerator += (reference[c] as f64 - current[c] as f64) * difference;
        denominator += difference * difference;
    }
    if denominator <= 1.0e-12 {
        return None;
    }
    Some(ConfidenceTarget {
        history_share: (numerator / denominator).clamp(0.0, 1.0) as f32,
        separation: denominator.min(f32::MAX as f64) as f32,
    })
}

/// CPU fusion contract. Inputs are compressed; only the final result is
/// compressed again. This also preserves the established HDR decode ceiling.
pub fn blend(
    current: [f32; 3],
    guide: [f32; 3],
    history: [f32; 3],
    gather_share: f32,
    history_share: f32,
) -> [f32; 3] {
    std::array::from_fn(|c| {
        let spatial = (1.0 - gather_share) * transform::decompress(guide[c])
            + gather_share * transform::decompress(current[c]);
        transform::compress(
            (1.0 - history_share) * spatial + history_share * transform::decompress(history[c]),
        )
    })
}

fn constant(graph: &mut Graph, len: usize, value: f32) -> NodeId {
    graph.constant(vec![value; len], &[len])
}

fn compress(graph: &mut Graph, value: NodeId, len: usize) -> NodeId {
    let positive = graph.relu(value);
    let one = constant(graph, len, 1.0);
    let denominator = graph.add(one, positive);
    graph.div(positive, denominator)
}

fn decompress(graph: &mut Graph, value: NodeId, len: usize) -> NodeId {
    let positive = graph.relu(value);
    let minus_ceiling = constant(graph, len, -(1.0 - 1.0 / 4096.0));
    let excess = graph.add(positive, minus_ceiling);
    let excess = graph.relu(excess);
    let minus_excess = graph.neg(excess);
    let clipped = graph.add(positive, minus_excess);
    let negative = graph.neg(clipped);
    let one = constant(graph, len, 1.0);
    let denominator = graph.add(one, negative);
    graph.div(clipped, denominator)
}

fn rgb_gate(graph: &mut Graph, gate: NodeId, shape: [u32; 3]) -> NodeId {
    let [batch, slots, spatial] = shape;
    let twice = graph.concat(gate, gate, batch, slots, slots, spatial);
    graph.concat(twice, gate, batch, 2 * slots, slots, spatial)
}

fn mix(graph: &mut Graph, first: NodeId, second: NodeId, gate: NodeId, shape: [u32; 3]) -> NodeId {
    let gates = rgb_gate(graph, gate, shape);
    let negative = graph.neg(gates);
    let one = constant(graph, (shape[0] * 3 * shape[1] * shape[2]) as usize, 1.0);
    let keep = graph.add(one, negative);
    let a = graph.mul(first, keep);
    let b = graph.mul(second, gates);
    graph.add(a, b)
}

fn distance(graph: &mut Graph, a: NodeId, b: NodeId, shape: [u32; 3]) -> NodeId {
    let [batch, slots, spatial] = shape;
    let neg = graph.neg(b);
    let delta = graph.add(a, neg);
    let squared = graph.mul(delta, delta);
    let r = graph.split_a(squared, batch, slots, 2 * slots, spatial);
    let gb = graph.split_b(squared, batch, slots, 2 * slots, spatial);
    let g = graph.split_a(gb, batch, slots, slots, spatial);
    let b = graph.split_b(gb, batch, slots, slots, spatial);
    let rg = graph.add(r, g);
    let rgb = graph.add(rg, b);
    let third = constant(graph, (batch * slots * spatial) as usize, 1.0 / 3.0);
    graph.mul(rgb, third)
}

fn gate(
    graph: &mut Graph,
    coefficients: NodeId,
    features: [NodeId; 4],
    mode: Mode,
    shape: [u32; 3],
) -> NodeId {
    let [batch, slots, spatial] = shape;
    if mode != Mode::CandidateAware {
        let denominator = graph.add(coefficients, features[0]);
        return graph.div(coefficients, denominator);
    }
    let mut rest = coefficients;
    let mut score = None;
    for (index, feature) in features.into_iter().enumerate() {
        let remaining = (3 - index) as u32 * slots;
        let part = if remaining == 0 {
            rest
        } else {
            graph.split_a(rest, batch, slots, remaining, spatial)
        };
        if remaining != 0 {
            rest = graph.split_b(rest, batch, slots, remaining, spatial);
        }
        let term = graph.mul(part, feature);
        score = Some(match score {
            None => term,
            Some(previous) => graph.add(previous, term),
        });
    }
    let half = constant(graph, (batch * slots * spatial) as usize, 0.5);
    let score = graph.mul(score.unwrap(), half);
    let t = graph.tanh(score);
    let t = graph.mul(t, half);
    graph.add(t, half)
}

/// Training/image graph counterpart of the CPU and unpack shader contract.
/// Candidate-aware modes consume actual candidates *after* spatial gathering,
/// so a dynamic gate is no longer blind to the radiance it is selecting.
pub(crate) fn build(
    graph: &mut Graph,
    candidates: [NodeId; 3],
    validity: NodeId,
    gates: [NodeId; 2],
    mode: Mode,
    shape: [u32; 3],
) -> NodeId {
    let [current, guide, history] = candidates;
    let [batch, slots, spatial] = shape;
    let len = (batch * 3 * slots * spatial) as usize;
    let one = constant(graph, (batch * slots * spatial) as usize, 1.0);
    let cg = distance(graph, current, guide, shape);
    let ch = distance(graph, current, history, shape);
    let gh = distance(graph, guide, history, shape);
    let ch = graph.mul(ch, validity);
    let gh = graph.mul(gh, validity);
    let features = [one, cg, ch, gh];
    let gather_share = gate(graph, gates[0], features, mode, shape);
    let history_share = gate(graph, gates[1], features, mode, shape);
    let history_share = graph.mul(history_share, validity);
    let current = decompress(graph, current, len);
    let guide = decompress(graph, guide, len);
    let history = decompress(graph, history, len);
    let spatial = mix(graph, guide, current, gather_share, shape);
    let result = mix(graph, spatial, history, history_share, shape);
    compress(graph, result, len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_mean_is_not_mean_compressed_radiance() {
        let high = [transform::compress(4.0); 3];
        let result = blend([0.0; 3], high, [0.0; 3], 0.5, 0.0);
        assert!((transform::decompress(result[0]) - 2.0).abs() < 1.0e-6);
        assert!((transform::decompress(high[0] * 0.5) - 2.0 / 3.0).abs() < 1.0e-6);
    }

    #[test]
    fn gate_observes_candidates_and_masks_missing_history() {
        let coefficients = [2.0, 0.0, -20.0, 0.0];
        let matching = features([0.2; 3], [0.2; 3], [0.2; 3], 1.0);
        let changed = features([0.2; 3], [0.2; 3], [0.9; 3], 1.0);
        assert!(candidate_gate(coefficients, matching) > 0.8);
        assert!(candidate_gate(coefficients, changed) < 0.01);
        assert_eq!(
            features([0.2; 3], [0.3; 3], [0.0; 3], 0.0),
            features([0.2; 3], [0.3; 3], [1.0; 3], 0.0)
        );
    }

    #[test]
    fn confidence_targets_distinguish_invalid_from_identical() {
        let target = confidence_target([0.0; 3], [4.0; 3], [2.0; 3], true).unwrap();
        assert_eq!(target.history_share, 0.5);
        assert_eq!(target.separation, 48.0);
        assert!(confidence_target([1.0; 3], [1.0; 3], [2.0; 3], true).is_none());
        assert!(confidence_target([0.0; 3], [4.0; 3], [2.0; 3], false).is_none());
        assert!(confidence_target([f32::NAN; 3], [4.0; 3], [2.0; 3], true).is_none());
        assert_eq!(
            confidence_target([0.0; 3], [4.0; 3], [8.0; 3], true)
                .unwrap()
                .history_share,
            1.0
        );
    }

    #[test]
    fn calibration_changes_bias_not_candidate_sensitivity() {
        let mut values = [0.0, 0.0, -2.0, -3.0, 4.0, 5.0, 6.0, 7.0];
        calibrate(&mut values, Mode::CandidateAware, 2, 1, 2.0);
        assert_eq!(values[0], 2.0f32.ln());
        assert_eq!(&values[2..], &[-2.0, -3.0, 4.0, 5.0, 6.0, 7.0]);
        calibrate(&mut values, Mode::CandidateAware, 2, 1, 0.0);
        assert_eq!(
            candidate_gate([values[0], values[2], values[4], values[6]], [1.0; 4]),
            0.0
        );
    }
}
