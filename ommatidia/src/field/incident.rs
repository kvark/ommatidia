//! Target-only directional radiance, measured before scattering at the receiver.
//!
//! Directions point from the probe toward the scene, not along photon travel.
//! `direct` is visible emission/sky; `indirect` contains at least one scattering
//! event after leaving the probe. Neither contains receiver albedo or a cosine.
use super::{Ray, dot};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Proposal {
    UniformHemisphere,
    EmitterAimed,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Estimate {
    pub mean: [f32; 3],
    /// Variance of the stored Monte Carlo mean, not variance of individual paths.
    pub variance_of_mean: [f32; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Probe {
    /// Actual offset ray origin in world coordinates.
    pub origin: [f32; 3],
    pub direction: [f32; 3],
    /// Sampling strata for pointwise supervision, NOT a quadrature rule.
    pub proposal: Proposal,
    pub direct: Estimate,
    pub indirect: Estimate,
    /// Measured on paired sums, preserving direct/indirect covariance.
    pub total_variance_of_mean: [f32; 3],
}
impl Probe {
    pub fn ray(&self) -> Ray {
        Ray {
            origin: self.origin,
            direction: self.direction,
        }
    }
    pub fn total(&self) -> [f32; 3] {
        std::array::from_fn(|c| self.direct.mean[c] + self.indirect.mean[c])
    }
    pub fn validate(&self) -> Result<(), String> {
        if self
            .origin
            .iter()
            .chain(&self.direction)
            .any(|v| !v.is_finite())
            || (dot(self.direction, self.direction) - 1.0).abs() > 1e-3
            || self
                .direct
                .mean
                .iter()
                .chain(&self.indirect.mean)
                .chain(&self.direct.variance_of_mean)
                .chain(&self.indirect.variance_of_mean)
                .chain(&self.total_variance_of_mean)
                .chain(&self.total())
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err("invalid incident-radiance probe".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capture {
    pub version: u32,
    pub integrator: String,
    pub max_bounces: u32,
    pub batches: u32,
    pub paths_per_batch: u32,
    pub ray_epsilon: f32,
    pub ray_distance: f32,
    pub radiance_ceiling: f32,
    pub probes: Vec<Probe>,
}
impl Capture {
    pub fn validate(&self) -> Result<(), String> {
        if self.version != 1
            || self.integrator != "blade-canonical-point-ray-v1"
            || !(2..=4096).contains(&self.batches)
            || self.paths_per_batch != 4
            || self.max_bounces > 64
            || self.probes.is_empty()
            || self.probes.len() > 65536
            || !self.ray_epsilon.is_finite()
            || self.ray_epsilon <= 0.0
            || !self.ray_distance.is_finite()
            || self.ray_distance <= self.ray_epsilon
            || !self.radiance_ceiling.is_finite()
            || self.radiance_ceiling <= 0.0
        {
            return Err("invalid incident capture provenance or sampling budget".into());
        }
        for p in &self.probes {
            p.validate()?;
        }
        Ok(())
    }
}

/// Welford moments of independent batch means, accumulated in f64.
#[derive(Default)]
pub struct Moments {
    count: usize,
    mean: [f64; 3],
    m2: [f64; 3],
}
impl Moments {
    pub fn push(&mut self, rgb: [f32; 3]) -> Result<(), String> {
        if rgb.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("non-finite/negative radiance batch; refusing to label it".into());
        }
        self.count += 1;
        for (c, &v) in rgb.iter().enumerate() {
            let delta = v as f64 - self.mean[c];
            self.mean[c] += delta / self.count as f64;
            self.m2[c] += delta * (v as f64 - self.mean[c]);
        }
        Ok(())
    }
    pub fn finish(&self) -> Result<Estimate, String> {
        if self.count < 2 {
            return Err("variance requires at least two independent batches".into());
        }
        let out = Estimate {
            mean: self.mean.map(|v| v as f32),
            variance_of_mean: self
                .m2
                .map(|v| (v.max(0.0) / (self.count * (self.count - 1)) as f64) as f32),
        };
        if out
            .mean
            .iter()
            .chain(&out.variance_of_mean)
            .any(|v| !v.is_finite())
        {
            return Err("radiance moments overflow".into());
        }
        Ok(out)
    }
}

/// Labels remain separate from the RGB observations and projected query features.
pub struct Targets {
    pub direct: Vec<f32>,
    pub indirect: Vec<f32>,
    pub direct_mask: Vec<f32>,
    pub indirect_mask: Vec<f32>,
}
impl Targets {
    pub fn new(probes: &[&Probe], exposure: f32) -> Result<Self, String> {
        if probes.is_empty() || !exposure.is_finite() || exposure <= 0.0 {
            return Err("incident targets require probes and positive exposure".into());
        }
        for p in probes {
            p.validate()?;
        }
        // Bounded inverse uncertainty in log1p space. This is a delta-method
        // approximation, not a likelihood model. Normalize per component count;
        // a noiseless label cannot acquire infinite precision.
        let collect = |direct: bool| -> Result<(Vec<f32>, Vec<f32>), String> {
            let mut rgb = Vec::with_capacity(3 * probes.len());
            let mut weights = Vec::with_capacity(3 * probes.len());
            for p in probes {
                let e = if direct { p.direct } else { p.indirect };
                for c in 0..3 {
                    if !(e.mean[c] * exposure).is_finite() {
                        return Err("incident exposure overflows".into());
                    }
                    rgb.push(e.mean[c]);
                    let derivative = exposure as f64 / (1.0 + exposure as f64 * e.mean[c] as f64);
                    let variance = e.variance_of_mean[c] as f64 * derivative * derivative;
                    weights.push((1.0 / (1.0 + variance / 0.01)) as f32);
                }
            }
            let sum: f64 = weights.iter().map(|v| *v as f64).sum();
            if sum <= 0.0 {
                return Err("incident supervision has no finite weight".into());
            }
            let scale = weights.len() as f64 / sum;
            Ok((
                rgb,
                weights
                    .into_iter()
                    .map(|w| (w as f64 * scale).sqrt() as f32)
                    .collect(),
            ))
        };
        let (direct, direct_mask) = collect(true)?;
        let (indirect, indirect_mask) = collect(false)?;
        Ok(Self {
            direct,
            indirect,
            direct_mask,
            indirect_mask,
        })
    }
    /// Scaling the masks (rather than specializing the graph) keeps the
    /// weight-zero control's inputs, query layout and initialization identical.
    pub fn weighted(mut self, weight: f32) -> Result<Self, String> {
        if !weight.is_finite() || weight < 0.0 {
            return Err("invalid incident loss weight".into());
        }
        let scale = weight.sqrt();
        for v in self.direct_mask.iter_mut().chain(&mut self.indirect_mask) {
            *v *= scale;
        }
        Ok(self)
    }
    pub fn feed(&self, session: &mut meganeura::Session) {
        session.set_input("target.incident_direct", &self.direct);
        session.set_input("target.incident_indirect", &self.indirect);
        session.set_input("target.incident_direct_mask", &self.direct_mask);
        session.set_input("target.incident_indirect_mask", &self.indirect_mask);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batch_variance_is_variance_of_mean_and_retains_covariance() {
        let mut a = Moments::default();
        assert!(a.finish().is_err());
        for v in [1.0, 3.0] {
            a.push([v; 3]).unwrap();
        }
        let e = a.finish().unwrap();
        assert_eq!(e.mean, [2.0; 3]);
        assert_eq!(e.variance_of_mean, [1.0; 3]);
        let mut sum = Moments::default();
        for (a, b) in [(1.0, 3.0), (3.0, 1.0)] {
            sum.push([a + b; 3]).unwrap();
        }
        assert_eq!(sum.finish().unwrap().variance_of_mean, [0.0; 3]);
        assert!(a.push([f32::NAN; 3]).is_err());
    }
    #[test]
    fn uncertainty_weights_are_bounded_and_normalized() {
        let mut p = Probe {
            origin: [0.0; 3],
            direction: [0.0, 1.0, 0.0],
            proposal: Proposal::UniformHemisphere,
            direct: Estimate {
                mean: [1.0; 3],
                variance_of_mean: [0.0; 3],
            },
            indirect: Estimate {
                mean: [1.0; 3],
                variance_of_mean: [0.0; 3],
            },
            total_variance_of_mean: [0.0; 3],
        };
        let q = p.clone();
        p.indirect.variance_of_mean = [100.0; 3];
        let targets = Targets::new(&[&p, &q], 1.0).unwrap();
        assert!(targets.indirect_mask[0] < targets.indirect_mask[3]);
        assert!((targets.indirect_mask.iter().map(|v| v * v).sum::<f32>() - 6.0).abs() < 1e-5);
        assert_eq!(targets.direct_mask, vec![1.0; 6]);
        p.direction = [0.0; 3];
        assert!(p.validate().is_err());
    }
}
