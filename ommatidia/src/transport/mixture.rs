//! Explicit checkpoint contract for candidate-weight parameterization.
use super::CANDIDATES;
use meganeura::{Graph, NodeId};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Softplus,
    MaskedSoftmax,
}
impl Mode {
    /// v2 prevents older runtimes from silently interpreting softmax weights as softplus.
    pub fn version(self) -> u32 {
        match self {
            Self::Softplus => 1,
            Self::MaskedSoftmax => 2,
        }
    }
}
impl std::str::FromStr for Mode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "softplus" => Ok(Self::Softplus),
            "masked-softmax" => Ok(Self::MaskedSoftmax),
            _ => Err("mixture must be softplus or masked-softmax".into()),
        }
    }
}
/// Match d log(softplus(z))/dz at zero: both modes start at the same prior
/// and have the same first-order sensitivity to the common initialized head.
pub const SOFTMAX_GAIN: f32 = std::f32::consts::LOG2_E * 0.5;

fn fill(g: &mut Graph, x: NodeId, v: f32) -> NodeId {
    let shape = g.node(x).ty.shape.clone();
    g.constant(vec![v; shape.iter().product()], &shape)
}
fn choose(g: &mut Graph, yes: NodeId, no: NodeId, mask: NodeId) -> NodeId {
    let one = fill(g, mask, 1.0);
    let neg = g.neg(mask);
    let inverse = g.add(one, neg);
    let yes = g.mul(yes, mask);
    let no = g.mul(no, inverse);
    g.add(yes, no)
}
fn split(g: &mut Graph, mut x: NodeId, channels: u32, spatial: u32) -> Vec<NodeId> {
    let mut rows = Vec::with_capacity(CANDIDATES);
    for k in 0..CANDIDATES - 1 {
        let rest = (CANDIDATES - k - 1) as u32 * channels;
        rows.push(g.split_a(x, 1, channels, rest, spatial));
        x = g.split_b(x, 1, channels, rest, spatial);
    }
    rows.push(x);
    rows
}
/// Same candidate-major layout as native logits. Priors are observed constants:
/// finite, nonnegative, and at least one positive prior per lobe/output pixel.
/// Native preparation guarantees this through its five spatial candidates.
///
/// Center *before* adding log-priors so large common logit shifts do not erase
/// prior differences. The second centering/normalization is Meganeura softmax.
/// The centering value is detached; softmax's shift invariance makes its derivative
/// cancel analytically. Invalid candidates never participate in centering.
pub fn build(g: &mut Graph, logits: NodeId, prior: NodeId, channels: u32, spatial: u32) -> NodeId {
    let z = split(g, logits, channels, spatial);
    let p = split(g, prior, channels, spatial);
    let legal: Vec<_> = p
        .iter()
        .map(|&v| {
            let zero = fill(g, v, 0.0);
            g.greater(v, zero)
        })
        .collect();
    let sentinel = fill(g, z[0], -f32::MAX);
    let mut center = choose(g, z[0], sentinel, legal[0]);
    for k in 1..CANDIDATES {
        let candidate = choose(g, z[k], sentinel, legal[k]);
        let greater = g.greater(candidate, center);
        center = choose(g, candidate, center, greater);
    }
    let center = g.stop_gradient(center);
    let negative_center = g.neg(center);
    let gain = fill(g, z[0], SOFTMAX_GAIN);
    let mut packed = None;
    for k in 0..CANDIDATES {
        // Replace the invalid value before subtraction: no 0 * infinity in masking.
        let safe_z = choose(g, z[k], center, legal[k]);
        let shifted = g.add(safe_z, negative_center);
        let shifted = g.mul(shifted, gain);
        let one = fill(g, p[k], 1.0);
        let safe_prior = choose(g, p[k], one, legal[k]);
        let log_prior = g.log(safe_prior);
        let score = g.add(shifted, log_prior);
        let score = choose(g, score, sentinel, legal[k]);
        packed = Some(match packed {
            None => score,
            Some(old) => g.concat(old, score, 1, k as u32 * channels, channels, spatial),
        });
    }
    let matrix = g.reshape(
        packed.unwrap(),
        &[CANDIDATES, (channels * spatial) as usize],
    );
    let rows = g.transpose(matrix);
    let normalized = g.softmax(rows);
    let columns = g.transpose(normalized);
    g.reshape(columns, &[CANDIDATES * (channels * spatial) as usize])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sidecars_do_not_reinterpret_older_weights() {
        let old =
            "(version:1,scale:2,channels:8,exposure:1.0,diffuse_frames:32.0,specular_frames:8.0)";
        let c: super::super::Config = ron::from_str(old).unwrap();
        assert_eq!(c.mixture, Mode::Softplus);
        c.validate([8, 8]).unwrap();
        let c = super::super::Config {
            mixture: Mode::MaskedSoftmax,
            ..c
        };
        assert!(c.validate([8, 8]).is_err());
        let c = super::super::Config { version: 2, ..c };
        c.validate([8, 8]).unwrap();
        let decoded: super::super::Config = ron::from_str(&ron::to_string(&c).unwrap()).unwrap();
        assert_eq!(decoded.mixture, Mode::MaskedSoftmax);
        assert_eq!(decoded.version, 2);
    }
}
