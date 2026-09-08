//! RGB-predicted source-ray termination. Targets are never inference inputs.
use super::{Bounds, Config, Observations, Ray, data, dot, surface, unit};

/// Architecture constant for VisibleRgb: sixteen finite intervals and escape.
pub const BINS: usize = 16;

/// Visibility to a query from a source camera. A half-bin tolerance avoids
/// rejecting the surface itself. Source predictions, not truth depth, are
/// integrated with these deterministic geometry-only coefficients.
pub fn survival(bounds: Bounds, origin: [f32; 3], position: [f32; 3]) -> [f32; BINS + 1] {
    let delta = std::array::from_fn(|i| position[i] - origin[i]);
    let distance = dot(delta, delta).sqrt();
    if !distance.is_finite() || distance <= 1e-6 {
        return [0.0; BINS + 1];
    }
    let ray = Ray {
        origin,
        direction: unit(delta),
    };
    let (near, far) = data::ray_interval(bounds, ray);
    if far <= near {
        return [0.0; BINS + 1];
    }
    let u = (distance - near) / (far - near) * BINS as f32 - 0.5;
    std::array::from_fn(|bin| {
        if bin == BINS {
            1.0
        } else {
            (bin as f32 + 1.0 - u).clamp(0.0, 1.0)
        }
    })
}

/// Pixel-major categorical labels; captures are kept out of Observations.
/// Hits outside the acquisition cube and uncertified misses remain masked.
pub struct Targets {
    pub mass: Vec<Vec<f32>>,
    pub valid: usize,
}
impl Targets {
    pub fn new(
        obs: &Observations,
        config: &Config,
        captures: &[Option<surface::Capture>],
        weight: f32,
    ) -> Result<Self, String> {
        obs.validate(config)?;
        if captures.len() != config.views || !weight.is_finite() || weight < 0.0 {
            return Err("invalid source visibility targets or weight".into());
        }
        let [w, h] = config.extent;
        let n = (w * h) as usize;
        let mut mass = Vec::new();
        let mut valid = 0;
        for (view, capture) in obs.views.iter().zip(captures) {
            let capture = capture.as_ref().ok_or("visible-rgb training requires source surface labels, including the zero-weight arm")?;
            capture.validate(config.extent)?;
            let mut labels = vec![0.0; n * (BINS + 1)];
            for p in 0..n {
                let ray = view.camera.ray(
                    [(p % w as usize) as f32, (p / w as usize) as f32],
                    config.extent,
                );
                if let Some(bin) = surface::class(
                    obs.bounds,
                    ray,
                    capture.distance[p],
                    capture.ray_limit,
                    BINS,
                ) {
                    labels[p * (BINS + 1) + bin] = 1.0;
                    valid += 1;
                }
            }
            mass.push(labels);
        }
        let scale = weight / valid.max(1) as f32;
        for row in &mut mass {
            row.iter_mut().for_each(|v| *v *= scale);
        }
        Ok(Self { mass, valid })
    }
    pub fn feed(&self, session: &mut meganeura::Session) {
        for (v, mass) in self.mass.iter().enumerate() {
            session.set_input(&format!("target.view{v}.termination"), mass);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn survival_is_monotone_and_escape_stays_visible() {
        let b = Bounds {
            center: [0.0; 3],
            radius: 1.0,
        };
        let origin = [0.0, 0.0, 3.0];
        let mut prior = [1.0; BINS + 1];
        for i in 0..100 {
            let row = survival(b, origin, [0.0, 0.0, 1.0 - i as f32 * 0.03]);
            for j in 0..=BINS {
                assert!(row[j] <= prior[j] && row[j] >= 0.0);
            }
            assert_eq!(row[BINS], 1.0);
            prior = row;
        }
        assert_eq!(survival(b, origin, [0.0, 0.0, 0.0])[0], 0.0);
        assert_eq!(survival(b, origin, origin), [0.0; BINS + 1]);
    }
}
