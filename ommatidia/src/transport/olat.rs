//! Linear OLAT mixtures for transport pretraining, not a sensor-noise simulator.
//! Each basis image must share geometry, camera, exposure and calibrated units.
//! JPEG/display-space images must be linearized and unclipped data verified first.
use crate::rng::Rng;

/// Complete measured response and an unbiased finite-light Monte Carlo estimate.
/// This reproduces sampling over the measured light basis, not arbitrary path noise.
pub fn pair(
    basis: &[Vec<f32>],
    coefficients: &[f32],
    samples: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if basis.is_empty() || basis.len() != coefficients.len() || samples == 0 {
        return Err(
            "OLAT requires a nonempty basis, matching coefficients and positive samples".into(),
        );
    }
    let n = basis[0].len();
    if n == 0
        || basis
            .iter()
            .any(|b| b.len() != n || b.iter().any(|v| !v.is_finite() || *v < 0.0))
        || coefficients.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(
            "OLAT observations must be finite, nonnegative, equally sized linear radiance".into(),
        );
    }
    let total: f64 = coefficients.iter().map(|v| *v as f64).sum();
    let mut target = vec![0.0f32; n];
    let mut noisy = vec![0.0f32; n];
    for (image, &a) in basis.iter().zip(coefficients) {
        for (value, &b) in target.iter_mut().zip(image) {
            *value += a * b;
        }
    }
    if !total.is_finite() || target.iter().any(|v| !v.is_finite()) {
        return Err("OLAT mixture overflows".into());
    }
    if total == 0.0 {
        return Ok((noisy, target));
    }
    let mut rng = Rng::new(seed);
    // Independent light draws per pixel. q_j=a_j/sum(a) gives a_j/q_j=sum(a).
    // All RGB channels of a pixel share the same draw to preserve chromaticity.
    if !n.is_multiple_of(3) {
        return Err("OLAT images must be interleaved RGB".into());
    }
    for pixel in noisy.chunks_exact_mut(3).enumerate() {
        let (p, output) = pixel;
        for _ in 0..samples {
            let draw = rng.uniform() as f64 * total;
            let mut cumulative = 0.0;
            let mut selected = coefficients.len() - 1;
            for (j, &a) in coefficients.iter().enumerate() {
                cumulative += a as f64;
                if draw < cumulative {
                    selected = j;
                    break;
                }
            }
            for c in 0..3 {
                output[c] += (total / samples as f64 * basis[selected][3 * p + c] as f64) as f32;
            }
        }
    }
    if noisy.iter().any(|v| !v.is_finite()) {
        return Err("sampled OLAT estimate overflows".into());
    }
    Ok((noisy, target))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_estimates_converge_to_linear_superposition() {
        let basis = vec![vec![0.0, 2.0, 4.0], vec![4.0, 0.0, 2.0]];
        let a = [0.25, 0.75];
        let mut mean = [0.0; 3];
        for seed in 0..8192 {
            let (noisy, target) = pair(&basis, &a, 1, seed).unwrap();
            assert_eq!(target, [3.0, 0.5, 2.5]);
            for c in 0..3 {
                mean[c] += noisy[c] / 8192.0;
            }
        }
        for c in 0..3 {
            assert!((mean[c] - [3.0, 0.5, 2.5][c]).abs() < 0.06);
        }
        assert!(pair(&basis, &[-1.0, 1.0], 1, 0).is_err());
    }
}
