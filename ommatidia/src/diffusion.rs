//! Noise schedule, training corruption, and sampler.
//!
//! Everything here operates on the sub-pixel residual described in
//! `docs/design.md`: the quantity being diffused is the difference between the
//! reference frame and nearest-neighbour upsampling of the input, rearranged
//! so that a scale-`S` upscale becomes `3 * S^2` channels at input resolution.
//!
//! # Parameterization
//!
//! The network predicts `x0` — the clean residual — rather than the noise that
//! was added to it. This is not the textbook choice and it is worth saying why.
//!
//! Recovering the clean signal from an e-prediction means computing
//! `x0 = (x_t - sqrt(1 - a) * eps) / sqrt(a)`. At the end of a cosine
//! schedule `sqrt(a)` is around `1e-3`, so that division multiplies whatever
//! error the network made by a thousand, and the very first sampling step —
//! the one taken from pure noise — is exactly where the network knows least.
//! In practice the sampler diverges immediately and returns noise, no matter
//! how low the training loss went.
//!
//! Predicting `x0` never performs that division. It also makes the two
//! objectives the same network with the same target: [`Objective::Direct`] is
//! x0-prediction with the noise level pinned at zero.
//!
//! [`Objective::Direct`]: crate::model::Objective::Direct

use crate::rng::Rng;

/// Discrete variance schedule over `T` timesteps.
///
/// Holds the cumulative products the corruption and the sampler need, so
/// neither has to recompute them per step.
#[derive(Clone, Debug)]
pub struct Schedule {
    /// Per-step variance.
    betas: Vec<f32>,
    /// `alpha_bar[t]`: the product of `1 - beta` up to and including `t`.
    alphas_cumprod: Vec<f32>,
}

impl Schedule {
    /// The cosine schedule from *Improved Denoising Diffusion Probabilistic
    /// Models* (Nichol and Dhariwal).
    ///
    /// Linear schedules destroy the signal too early, which wastes most of the
    /// trajectory on timesteps that carry no information. The cosine one keeps
    /// useful signal for longer, and matters more here than usual: the
    /// residual is small and band-limited to begin with, so a schedule that
    /// noises aggressively would spend nearly the whole trajectory on pure
    /// noise.
    pub fn cosine(steps: usize) -> Self {
        assert!(steps > 0, "a schedule needs at least one step");
        const OFFSET: f32 = 0.008;
        // f(t) = cos((t/T + s) / (1 + s) * pi/2)^2, normalised so f(0) = 1.
        let f = |t: usize| -> f32 {
            let x = (t as f32 / steps as f32 + OFFSET) / (1.0 + OFFSET);
            (x * std::f32::consts::FRAC_PI_2).cos().powi(2)
        };
        let f0 = f(0);

        let mut alphas_cumprod = Vec::with_capacity(steps);
        let mut betas = Vec::with_capacity(steps);
        let mut previous = 1.0f32;
        for t in 0..steps {
            let bar = (f(t + 1) / f0).clamp(1e-6, 1.0);
            // Clamped as in the paper: a beta near 1 makes the reverse step
            // numerically hopeless.
            betas.push((1.0 - bar / previous).clamp(0.0, 0.999));
            alphas_cumprod.push(bar);
            previous = bar;
        }
        Self {
            betas,
            alphas_cumprod,
        }
    }

    /// Number of timesteps.
    pub fn len(&self) -> usize {
        self.betas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.betas.is_empty()
    }

    /// Per-step variance at `t`.
    pub fn beta(&self, t: usize) -> f32 {
        self.betas[t]
    }

    /// `alpha_bar[t]`, the signal retained after `t + 1` corruption steps.
    pub fn alpha_bar(&self, t: usize) -> f32 {
        self.alphas_cumprod[t]
    }

    /// `alpha_bar` for the step before `t`, or 1 at the start of the chain
    /// where nothing has been added yet.
    pub fn alpha_bar_prev(&self, t: usize) -> f32 {
        if t == 0 {
            1.0
        } else {
            self.alphas_cumprod[t - 1]
        }
    }

    /// Corrupt `x0` to timestep `t`, writing `x_t` into `out`.
    ///
    /// `x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * noise`.
    pub fn add_noise(&self, x0: &[f32], noise: &[f32], t: usize, out: &mut [f32]) {
        assert_eq!(x0.len(), noise.len());
        assert_eq!(x0.len(), out.len());
        let bar = self.alpha_bar(t);
        let signal = bar.sqrt();
        let sigma = (1.0 - bar).sqrt();
        for i in 0..out.len() {
            out[i] = signal * x0[i] + sigma * noise[i];
        }
    }

    /// Recover the noise implied by `x_t` and a predicted clean signal.
    ///
    /// `eps = (x_t - sqrt(alpha_bar) * x0) / sqrt(1 - alpha_bar)`.
    ///
    /// The mirror of the e-parameterization's `x0` recovery, and the reason
    /// this direction is preferred: the divisor is `sqrt(1 - alpha_bar)`,
    /// which approaches 1 exactly where the other approaches 0.
    pub fn predict_eps(&self, x_t: &[f32], x0: &[f32], t: usize, out: &mut [f32]) {
        assert_eq!(x_t.len(), x0.len());
        assert_eq!(x_t.len(), out.len());
        let bar = self.alpha_bar(t);
        let signal = bar.sqrt();
        let sigma = (1.0 - bar).sqrt().max(1e-6);
        for i in 0..out.len() {
            out[i] = (x_t[i] - signal * x0[i]) / sigma;
        }
    }

    /// One deterministic DDIM step from `t` to `t_prev`, given a predicted
    /// clean signal.
    ///
    /// Jumping straight to `t_prev` rather than stepping one timestep at a
    /// time is what lets a schedule trained with a thousand steps be sampled
    /// with twenty. `eta` is fixed at zero: a deterministic trajectory is what
    /// makes frame-to-frame stability achievable at all, and a stochastic one
    /// would shimmer even on a static camera.
    ///
    /// `bound` clamps the prediction, since the residual is known to live in
    /// `(-gain, gain)` — the network is not obliged to respect that, and an
    /// excursion early in the chain compounds through every later step.
    pub fn ddim_step(
        &self,
        x_t: &[f32],
        x0: &[f32],
        t: usize,
        t_prev: Option<usize>,
        bound: f32,
        out: &mut [f32],
    ) {
        assert_eq!(x_t.len(), x0.len());
        assert_eq!(x_t.len(), out.len());
        let bar = self.alpha_bar(t);
        let signal = bar.sqrt();
        let sigma = (1.0 - bar).sqrt().max(1e-6);
        // At the end of the chain the target is the clean signal itself.
        let bar_prev = t_prev.map_or(1.0, |p| self.alpha_bar(p));
        let signal_prev = bar_prev.sqrt();
        let sigma_prev = (1.0 - bar_prev).max(0.0).sqrt();
        for i in 0..out.len() {
            let clean = x0[i].clamp(-bound, bound);
            let eps = (x_t[i] - signal * clean) / sigma;
            out[i] = signal_prev * clean + sigma_prev * eps;
        }
    }

    /// Evenly spaced descending timesteps for a sampler budgeted at `steps`.
    ///
    /// Always ends at 0 so the last step lands on the clean signal.
    pub fn sampling_timesteps(&self, steps: usize) -> Vec<usize> {
        assert!(steps > 0, "a sampler needs at least one step");
        let total = self.len();
        if steps >= total {
            return (0..total).rev().collect();
        }
        let mut out: Vec<usize> = (0..steps)
            .map(|i| {
                let position = i as f32 / (steps - 1).max(1) as f32;
                ((1.0 - position) * (total - 1) as f32).round() as usize
            })
            .collect();
        out.dedup();
        // `round` can leave the tail short of zero for some step counts.
        if out.last() != Some(&0) {
            out.push(0);
        }
        out
    }
}

/// Sinusoidal timestep embedding, in the transformer convention.
///
/// The first half is sine, the second cosine, over geometrically spaced
/// frequencies. Computed on the host and fed in as a tensor, so the graph does
/// not need a transcendental op it would otherwise only use here.
pub fn timestep_embedding(t: usize, dim: usize, max_period: f32) -> Vec<f32> {
    assert!(dim.is_multiple_of(2), "embedding width must be even");
    let half = dim / 2;
    let mut out = vec![0.0; dim];
    for i in 0..half {
        // exp(-ln(max_period) * i / half) spaced from 1 down to 1/max_period.
        let freq = (-(max_period.ln()) * i as f32 / half as f32).exp();
        let angle = t as f32 * freq;
        out[i] = angle.sin();
        out[half + i] = angle.cos();
    }
    out
}

/// Fill `out` with standard normal noise.
pub fn fill_normal(rng: &mut Rng, out: &mut [f32]) {
    for value in out.iter_mut() {
        *value = rng.normal();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_schedule_decays_monotonically() {
        let s = Schedule::cosine(1000);
        assert_eq!(s.len(), 1000);
        assert!(s.alpha_bar(0) > 0.99, "barely any signal lost at the start");
        assert!(s.alpha_bar(999) < 0.01, "signal should be gone at the end");
        for t in 1..s.len() {
            assert!(
                s.alpha_bar(t) <= s.alpha_bar(t - 1),
                "alpha_bar rose at {t}"
            );
            assert!(
                (0.0..=0.999).contains(&s.beta(t)),
                "beta {} at {t}",
                s.beta(t)
            );
        }
    }

    #[test]
    fn alpha_bar_prev_starts_clean() {
        let s = Schedule::cosine(50);
        assert_eq!(s.alpha_bar_prev(0), 1.0);
        assert_eq!(s.alpha_bar_prev(7), s.alpha_bar(6));
    }

    #[test]
    fn noising_then_inverting_recovers_the_noise() {
        let s = Schedule::cosine(200);
        let mut rng = Rng::new(11);
        let x0: Vec<f32> = (0..64).map(|_| rng.normal()).collect();
        let mut noise = vec![0.0; 64];
        fill_normal(&mut rng, &mut noise);

        for &t in &[0usize, 50, 150, 199] {
            let mut x_t = vec![0.0; 64];
            s.add_noise(&x0, &noise, t, &mut x_t);
            // Handed the exact clean signal, predict_eps has to land back on
            // the noise.
            let mut recovered = vec![0.0; 64];
            s.predict_eps(&x_t, &x0, t, &mut recovered);
            for (a, b) in noise.iter().zip(recovered.iter()) {
                assert!((a - b).abs() < 1e-2, "t={t}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn a_perfect_denoiser_walks_the_chain_back() {
        // Handed the true clean signal at every step, DDIM must arrive at it —
        // including from the very last timestep, which is where the
        // e-parameterization falls apart.
        let s = Schedule::cosine(1000);
        let mut rng = Rng::new(5);
        let x0: Vec<f32> = (0..32).map(|_| rng.normal()).collect();
        let mut noise = vec![0.0; 32];
        fill_normal(&mut rng, &mut noise);

        let steps = s.sampling_timesteps(10);
        assert_eq!(steps[0], 999, "sampling should start at the terminal step");
        let mut x = vec![0.0; 32];
        s.add_noise(&x0, &noise, steps[0], &mut x);

        let mut next = vec![0.0; 32];
        for (i, &t) in steps.iter().enumerate() {
            s.ddim_step(&x, &x0, t, steps.get(i + 1).copied(), 10.0, &mut next);
            x.copy_from_slice(&next);
        }
        for (a, b) in x0.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-2, "{a} vs {b}");
        }
    }

    /// The failure that motivates the parameterization, pinned so it cannot
    /// quietly come back.
    #[test]
    fn terminal_signal_rate_would_wreck_an_eps_prediction() {
        let s = Schedule::cosine(1000);
        let terminal = s.alpha_bar(999).sqrt();
        assert!(
            terminal < 1e-2,
            "terminal signal rate is {terminal}, the schedule changed"
        );
        // Dividing a prediction error by that is what x0-prediction avoids.
        let amplification = 1.0 / terminal;
        assert!(amplification > 100.0);
    }

    #[test]
    fn the_bound_contains_a_wild_prediction() {
        let s = Schedule::cosine(100);
        let x_t = vec![0.0; 8];
        let absurd = vec![1e6; 8];
        let mut out = vec![0.0; 8];
        s.ddim_step(&x_t, &absurd, 50, Some(25), 2.0, &mut out);
        assert!(
            out.iter().all(|v| v.is_finite() && v.abs() < 100.0),
            "an unbounded prediction escaped: {out:?}"
        );
    }

    #[test]
    fn sampling_timesteps_descend_to_zero() {
        let s = Schedule::cosine(1000);
        for &n in &[1usize, 4, 20, 50] {
            let steps = s.sampling_timesteps(n);
            assert_eq!(*steps.last().unwrap(), 0, "n={n} did not reach zero");
            for w in steps.windows(2) {
                assert!(w[0] > w[1], "n={n} did not descend: {steps:?}");
            }
        }
        // Asking for more steps than the schedule has walks all of it.
        assert_eq!(s.sampling_timesteps(5000).len(), 1000);
    }

    #[test]
    fn timestep_embedding_is_bounded_and_varies() {
        let a = timestep_embedding(0, 64, 10_000.0);
        let b = timestep_embedding(500, 64, 10_000.0);
        assert_eq!(a.len(), 64);
        assert!(a.iter().all(|v| v.abs() <= 1.0));
        assert!(b.iter().all(|v| v.abs() <= 1.0));
        assert_ne!(a, b, "different timesteps must embed differently");
        // t=0 is all sin(0)=0 then cos(0)=1.
        assert!(a[..32].iter().all(|&v| v == 0.0));
        assert!(a[32..].iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }
}
