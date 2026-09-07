//! First opaque surface targets. Never consulted by sampling or inference.
use super::{Bounds, Ray, data};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capture {
    pub version: u32,
    /// Labels and image radiance use the same nonjittered pixel-centre ray.
    pub pixel_filter: String,
    /// Maximum certified ray distance, in world units.
    pub ray_limit: f32,
    /// Euclidean distance along a unit ray, not camera Z. None is a certified miss.
    pub distance: Vec<Option<f32>>,
    /// First-hit material emission; zero at nonemitters and misses.
    pub emission: Vec<[f32; 3]>,
}
impl Capture {
    pub fn validate(&self, extent: [u32; 2]) -> Result<(), String> {
        let count = extent[0] as usize * extent[1] as usize;
        if self.version != 1
            || self.pixel_filter != "center"
            || !self.ray_limit.is_finite()
            || self.ray_limit <= 0.0
            || self.distance.len() != count
            || self.emission.len() != count
        {
            return Err("surface targets require v1 centre-ray maps of the image extent".into());
        }
        for (d, e) in self.distance.iter().zip(&self.emission) {
            if d.is_some_and(|t| !t.is_finite() || t <= 0.0 || t >= self.ray_limit)
                || e.iter().any(|v| !v.is_finite() || *v < 0.0)
                || (d.is_none() && *e != [0.0; 3])
            {
                return Err("invalid first-surface distance or emission".into());
            }
        }
        Ok(())
    }
}

/// A categorical termination target: one of the sampled intervals, or escape.
/// Hits outside the acquisition cube and uncertified misses are masked, not sky.
pub fn class(
    bounds: Bounds,
    ray: Ray,
    distance: Option<f32>,
    limit: f32,
    steps: usize,
) -> Option<usize> {
    if steps == 0 {
        return None;
    }
    let (near, far) = data::ray_interval(bounds, ray);
    if far <= near {
        return None;
    }
    match distance {
        Some(t) if t >= near && t < far => {
            Some((((t - near) / (far - near) * steps as f32) as usize).min(steps - 1))
        }
        None if far <= limit => Some(steps),
        _ => None,
    }
}

pub struct Targets {
    /// Step-major [steps+1, rays], already normalized by valid camera rays.
    pub mass: Vec<f32>,
    pub valid: usize,
}
impl Targets {
    pub fn new(
        bounds: Bounds,
        rays: &[Ray],
        steps: usize,
        labels: &[Option<(Option<f32>, f32)>],
        weight: f32,
    ) -> Result<Self, String> {
        if labels.len() != rays.len() || !weight.is_finite() || weight < 0.0 {
            return Err("surface labels/rays mismatch or invalid loss weight".into());
        }
        // Validate exactly the same bounds/ray contract as the observation sampler.
        data::ray_queries(bounds, rays, steps)?;
        let mut mass = vec![0.0; (steps + 1) * rays.len()];
        let mut valid = 0;
        for (i, label) in labels.iter().enumerate() {
            let Some((depth, limit)) = label else {
                continue;
            };
            if !limit.is_finite()
                || *limit <= 0.0
                || depth.is_some_and(|d| !d.is_finite() || d <= 0.0 || d >= *limit)
            {
                return Err("invalid surface ray label".into());
            }
            if let Some(bin) = class(bounds, rays[i], *depth, *limit, steps) {
                mass[bin * rays.len() + i] = 1.0;
                valid += 1;
            }
        }
        let scale = weight / valid.max(1) as f32;
        mass.iter_mut().for_each(|v| *v *= scale);
        Ok(Self { mass, valid })
    }
    pub fn feed(&self, session: &mut meganeura::Session) {
        session.set_input("target.termination", &self.mass);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn termination_labels_do_not_clear_occluded_space_or_shift_samples() {
        let b = Bounds {
            center: [0.0; 3],
            radius: 1.0,
        };
        let r = Ray {
            origin: [0.0, 0.0, -3.0],
            direction: [0.0, 0.0, 1.0],
        };
        let before = data::ray_queries(b, &[r], 4).unwrap();
        let a = Targets::new(b, &[r], 4, &[Some((Some(2.7), 20.0))], 1.0).unwrap();
        assert_eq!(a.mass, [0.0, 1.0, 0.0, 0.0, 0.0]);
        let miss = Targets::new(b, &[r], 4, &[Some((None, 20.0))], 1.0).unwrap();
        assert_eq!(miss.mass, [0.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(class(b, r, Some(5.0), 20.0, 4), None);
        assert_eq!(class(b, r, None, 3.0, 4), None);
        let zero = Targets::new(b, &[r], 4, &[Some((Some(2.7), 20.0))], 0.0).unwrap();
        assert!(zero.mass.iter().all(|v| *v == 0.0));
        let after = data::ray_queries(b, &[r], 4).unwrap();
        assert_eq!(before.1, after.1);
        for (a, b) in before.0.iter().zip(after.0) {
            assert_eq!(a.position, b.position);
        }
        assert!(Targets::new(b, &[r], 4, &[Some((Some(f32::NAN), 20.0))], 1.0).is_err());
    }
}
