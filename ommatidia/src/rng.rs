//! A small deterministic PRNG.
//!
//! Training has to be reproducible, and both the corruption process and the
//! epoch shuffle need a stream that can be seeded and replayed. This is
//! PCG-XSH-RR: one 64-bit state, good enough statistically for sampling noise,
//! and short enough not to be worth a dependency.

/// A seeded random stream.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
    increment: u64,
}

const MULTIPLIER: u64 = 6364136223846793005;

impl Rng {
    /// Start a stream from `seed`.
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
            // Any odd constant works; this one is the PCG reference default.
            increment: 1442695040888963407,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Uniform over the whole `u32` range.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(MULTIPLIER).wrapping_add(self.increment);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in `[0, n)`.
    ///
    /// Uses Lemire's multiply-shift with rejection, so the result is unbiased
    /// rather than merely close to it.
    pub fn below(&mut self, n: u32) -> u32 {
        assert!(n > 0, "range must be non-empty");
        let mut product = self.next_u32() as u64 * n as u64;
        let mut low = product as u32;
        if low < n {
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                product = self.next_u32() as u64 * n as u64;
                low = product as u32;
            }
        }
        (product >> 32) as u32
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f32 {
        // 24 bits is the f32 mantissa, so every value is exactly representable
        // and the spacing is uniform.
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Standard normal, by Box-Muller.
    ///
    /// A diffusion model learns to predict exactly the noise it was trained
    /// against, so the tails have to be right — an Irwin-Hall approximation
    /// would quietly cap them.
    pub fn normal(&mut self) -> f32 {
        // Guard the log against a zero draw.
        let u1 = self.uniform().max(f32::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u32 + 1) as usize;
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_the_same_stream() {
        let a: Vec<u32> = (0..16).map(|_| Rng::new(7).next_u32()).collect();
        let mut rng = Rng::new(7);
        assert_eq!(a[0], rng.clone().next_u32());
        let b: Vec<u32> = (0..16).map(|_| rng.next_u32()).collect();
        assert_ne!(b[0], b[1], "the stream should advance");
        assert_eq!(Rng::new(7).next_u32(), a[0]);
        assert_ne!(Rng::new(8).next_u32(), a[0]);
    }

    #[test]
    fn below_stays_in_range_and_covers_it() {
        let mut rng = Rng::new(1);
        let mut seen = [false; 5];
        for _ in 0..500 {
            let v = rng.below(5);
            assert!(v < 5);
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "every value should come up");
    }

    #[test]
    fn normal_has_the_right_moments() {
        let mut rng = Rng::new(42);
        const N: usize = 100_000;
        let samples: Vec<f32> = (0..N).map(|_| rng.normal()).collect();
        let mean = samples.iter().sum::<f32>() / N as f32;
        let variance = samples.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / N as f32;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((variance - 1.0).abs() < 0.03, "variance {variance}");
        // Box-Muller should reach well past where Irwin-Hall would be capped.
        let extreme = samples.iter().filter(|v| v.abs() > 3.0).count();
        assert!(extreme > 0, "the tails are missing");
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut items: Vec<u32> = (0..64).collect();
        Rng::new(3).shuffle(&mut items);
        assert_ne!(items, (0..64).collect::<Vec<_>>());
        items.sort();
        assert_eq!(items, (0..64).collect::<Vec<_>>());
    }

    #[test]
    fn uniform_stays_in_the_unit_interval() {
        let mut rng = Rng::new(9);
        for _ in 0..10_000 {
            let v = rng.uniform();
            assert!((0.0..1.0).contains(&v), "{v}");
        }
    }
}
