//! Cross-realization risk diagnostics. Models never receive reference RGB at inference.
use super::CANDIDATES;
pub const DIM: usize = 4 + CANDIDATES;

/// Only observed candidate colours; no pixel, scene, geometry or target identity.
pub fn features(candidates: &[[f32; 3]; CANDIDATES], k: usize) -> [f64; DIM] {
    let mut x = [0.0; DIM];
    x[0] = 1.0;
    for c in 0..3 {
        x[1 + c] = (candidates[k][c].max(0.0) as f64).ln_1p();
    }
    for j in 0..CANDIDATES {
        x[4 + j] = mse(candidates[k], candidates[j]).ln_1p();
    }
    x
}
pub fn mse(a: [f32; 3], b: [f32; 3]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (*a as f64 - b as f64).powi(2))
        .sum::<f64>()
        / 3.0
}
#[derive(Clone)]
pub struct Regression {
    count: usize,
    xx: [[f64; DIM]; DIM],
    xy: [f64; DIM],
}
impl Default for Regression {
    fn default() -> Self {
        Self {
            count: 0,
            xx: [[0.0; DIM]; DIM],
            xy: [0.0; DIM],
        }
    }
}
impl Regression {
    pub fn samples(&self) -> usize {
        self.count
    }
    pub fn add(&mut self, x: [f64; DIM], risk: f64) -> Result<(), String> {
        if x.iter().any(|v| !v.is_finite()) || !risk.is_finite() || risk < 0.0 {
            return Err("nonfinite risk regression row".into());
        }
        // Predict expected linear squared risk, not expected log-risk.
        let y = risk;
        for i in 0..DIM {
            self.xy[i] += x[i] * y;
            for j in 0..DIM {
                self.xx[i][j] += x[i] * x[j];
            }
        }
        self.count += 1;
        Ok(())
    }
    /// Fixed ridge, normalized by examples. No validation/held-noise tuning.
    #[allow(clippy::needless_range_loop)] // Small indexed normal-equation elimination.
    pub fn fit(&self, ridge: f64) -> Result<[f64; DIM], String> {
        if self.count == 0 || !ridge.is_finite() || ridge <= 0.0 {
            return Err("empty regression or invalid ridge".into());
        }
        let mut a = [[0.0; DIM + 1]; DIM];
        for i in 0..DIM {
            for j in 0..DIM {
                a[i][j] = self.xx[i][j] / self.count as f64;
            }
            a[i][i] += ridge;
            a[i][DIM] = self.xy[i] / self.count as f64;
        }
        for i in 0..DIM {
            let p = (i..DIM)
                .max_by(|a0, b| a[*a0][i].abs().total_cmp(&a[*b][i].abs()))
                .unwrap();
            a.swap(i, p);
            if a[i][i].abs() < 1e-12 {
                return Err("singular risk regression".into());
            }
            let d = a[i][i];
            for v in &mut a[i][i..=DIM] {
                *v /= d;
            }
            let row = a[i];
            for (j, r) in a.iter_mut().enumerate() {
                if j != i {
                    let f = r[i];
                    for k in i..=DIM {
                        r[k] -= f * row[k];
                    }
                }
            }
        }
        let result = std::array::from_fn(|i| a[i][DIM]);
        if result.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite regression solution".into());
        }
        Ok(result)
    }
}
pub fn predict(weights: &[f64; DIM], x: &[f64; DIM]) -> f64 {
    weights.iter().zip(x).map(|(a, b)| a * b).sum()
}

/// Select on fitting realizations, evaluate on held realizations. Reference-based
/// per-pixel risk is privileged and not a deployable selector or oracle bound.
pub fn transfer(
    candidates: &[[[f32; 3]; CANDIDATES]],
    available: [bool; CANDIDATES],
    target: [f32; 3],
    fit: usize,
) -> Result<(usize, f64), String> {
    if fit == 0 || fit >= candidates.len() || !available.iter().any(|v| *v) {
        return Err("invalid cross-noise split or no common candidates".into());
    }
    let k = (0..CANDIDATES)
        .filter(|k| available[*k])
        .min_by(|a, b| {
            let risk = |k| {
                candidates[..fit]
                    .iter()
                    .map(|p| mse(p[k], target))
                    .sum::<f64>()
            };
            risk(*a).total_cmp(&risk(*b))
        })
        .unwrap();
    let risk = candidates[fit..]
        .iter()
        .map(|p| mse(p[k], target))
        .sum::<f64>()
        / (candidates.len() - fit) as f64;
    Ok((k, risk))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn held_noise_cannot_change_the_fitted_choice() {
        let mut p = vec![[[10.0; 3]; CANDIDATES]; 4];
        p[0][0] = [1.0; 3];
        p[1][0] = [1.0; 3];
        p[2][1] = [1.0; 3];
        p[3][1] = [1.0; 3];
        let (k, e) = transfer(&p, [true; CANDIDATES], [1.0; 3], 2).unwrap();
        assert_eq!(k, 0);
        assert_eq!(e, 81.0);
        p[2][1] = [1000.0; 3];
        assert_eq!(transfer(&p, [true; CANDIDATES], [1.0; 3], 2).unwrap().0, 0);
    }
    #[test]
    fn observable_regression_recovers_simple_risk() {
        let mut r = Regression::default();
        for i in 0..100 {
            let mut x = [0.0; DIM];
            x[0] = 1.0;
            x[1] = i as f64 / 100.0;
            r.add(x, 0.3 + 2.0 * x[1]).unwrap();
        }
        let w = r.fit(1e-6).unwrap();
        let mut x = [0.0; DIM];
        x[0] = 1.0;
        x[1] = 0.35;
        assert!((predict(&w, &x) - 1.0).abs() < 1e-4);
        assert!(r.add(x, f64::NAN).is_err());
    }
}

#[cfg(test)]
mod linear_risk_tests {
    use super::*;
    #[test]
    fn rare_bright_errors_retain_their_linear_risk() {
        let mut r = Regression::default();
        let mut x = [0.0; DIM];
        x[0] = 1.0;
        for risk in [0.0, 0.0, 0.0, 100.0] {
            r.add(x, risk).unwrap();
        }
        assert_eq!(r.samples(), 4);
        assert!((predict(&r.fit(1e-8).unwrap(), &x) - 25.0).abs() < 1e-5);
    }
}
