//! Persistent crop cursors and bounded, deterministic one-batch prefetch.
use crate::{Result, corpus::Corpus};
use ommatidia::{
    rng::Rng,
    transport::{Config, Frame, Target},
};
use serde::{Deserialize, Serialize};
use std::{
    sync::{Arc, mpsc},
    thread,
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cursor {
    pub sequence: usize,
    pub origin: [u32; 2],
    pub frame: usize,
    pub remaining: usize,
    pub gain: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Window {
    pub sequence: usize,
    pub origin: [u32; 2],
    pub start: usize,
    pub gain: f32,
    pub reset: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sampler {
    rng: Rng,
    pub cursors: Vec<Option<Cursor>>,
    pub crop: [u32; 2],
    pub unroll: usize,
    low: [u32; 2],
    sequences: usize,
    length: usize,
    pub windows: u64,
    pub cold_windows: u64,
}

impl Sampler {
    pub fn new(
        seed: u64,
        batch: usize,
        crop: [u32; 2],
        unroll: usize,
        low: [u32; 2],
        sequences: usize,
        length: usize,
    ) -> Result<Self> {
        if batch == 0
            || unroll == 0
            || unroll > length
            || sequences == 0
            || sequences > u32::MAX as usize
            || length > u32::MAX as usize
            || crop.contains(&0)
            || crop[0] > low[0]
            || crop[1] > low[1]
        {
            return Err("invalid cursor dimensions/count/unroll".into());
        }
        Ok(Self {
            rng: Rng::new(seed),
            cursors: vec![None; batch],
            crop,
            unroll,
            low,
            sequences,
            length,
            windows: 0,
            cold_windows: 0,
        })
    }

    pub fn next_windows(&mut self) -> Vec<Window> {
        let mut windows = Vec::with_capacity(self.cursors.len());
        for cursor in &mut self.cursors {
            let birth = cursor
                .as_ref()
                .is_none_or(|c| c.remaining == 0 || c.frame + self.unroll > self.length);
            if birth {
                // Reserve 8–16 windows where available. Uniformly starting right
                // at the tail would otherwise inflate the natural cold fraction.
                let maximum = (self.length / self.unroll).min(16);
                let minimum = maximum.min(8);
                let remaining = minimum + self.rng.below((maximum - minimum + 1) as u32) as usize;
                *cursor = Some(Cursor {
                    sequence: self.rng.below(self.sequences as u32) as usize,
                    origin: std::array::from_fn(|i| self.rng.below(self.low[i] - self.crop[i] + 1)),
                    frame: self
                        .rng
                        .below((self.length - remaining * self.unroll + 1) as u32)
                        as usize,
                    remaining,
                    gain: (4.0 * self.rng.uniform() - 2.0).exp2(),
                });
            }
            let c = cursor.as_mut().unwrap();
            let reset = self.rng.uniform() < 0.1 || birth;
            windows.push(Window {
                sequence: c.sequence,
                origin: c.origin,
                start: c.frame,
                gain: c.gain,
                reset,
            });
            c.frame += self.unroll;
            c.remaining -= 1;
            self.windows += 1;
            self.cold_windows += u64::from(reset);
        }
        windows
    }
}

pub struct Batch {
    pub windows: Vec<Window>,
    pub frames: Vec<Vec<(Frame, Target)>>,
    /// Snapshot after THIS batch, not the worker's next prefetched batch.
    pub next_sampler: Sampler,
    pub decode_seconds: f64,
}

pub struct Prefetch {
    receiver: Option<mpsc::Receiver<std::result::Result<Batch, String>>>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Prefetch {
    pub fn new(corpus: Arc<Corpus>, config: Config, mut sampler: Sampler) -> Self {
        let (sender, receiver) = mpsc::sync_channel(1);
        let worker = thread::spawn(move || {
            loop {
                let start = Instant::now();
                let windows = sampler.next_windows();
                let frames = windows
                    .iter()
                    .map(|w| {
                        (0..sampler.unroll)
                            .map(|i| {
                                corpus
                                    .crop(
                                        w.sequence,
                                        w.start + i,
                                        w.origin,
                                        sampler.crop,
                                        w.gain,
                                        config,
                                    )
                                    .map_err(|e| e.to_string())
                            })
                            .collect()
                    })
                    .collect::<std::result::Result<Vec<Vec<_>>, String>>();
                let result = frames.map(|frames| Batch {
                    windows,
                    frames,
                    next_sampler: sampler.clone(),
                    decode_seconds: start.elapsed().as_secs_f64(),
                });
                let failed = result.is_err();
                if sender.send(result).is_err() || failed {
                    break;
                }
            }
        });
        Self {
            receiver: Some(receiver),
            worker: Some(worker),
        }
    }
    pub fn receive(&self) -> Result<Batch> {
        Ok(self
            .receiver
            .as_ref()
            .unwrap()
            .recv()
            .map_err(|_| "crop prefetch worker stopped")?
            .map_err(|s| format!("crop prefetch: {s}"))?)
    }
}
impl Drop for Prefetch {
    fn drop(&mut self) {
        // Unblock a full-channel send before joining, including on train errors.
        self.receiver.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Update numbers are one-based. Short correctness runs remain in warm-up.
pub fn learning_rate(peak: f32, step: usize, total: usize) -> f32 {
    assert!(step > 0 && step <= total);
    if step <= 500 {
        peak * step as f32 / 500.0
    } else {
        let fraction = (step - 500) as f32 / (total - 500) as f32;
        peak * (0.1 + 0.45 * (1.0 + (std::f32::consts::PI * fraction).cos()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cursor_continuity_cold_fraction_and_exact_resume() {
        let mut sampler = Sampler::new(7, 8, [64; 2], 4, [128; 2], 40, 64).unwrap();
        let mut previous: Option<Vec<Window>> = None;
        for _ in 0..10000 {
            let current = sampler.next_windows();
            for (i, w) in current.iter().enumerate() {
                assert!(w.start + 4 <= 64 && w.sequence < 40);
                assert!(w.origin.iter().all(|v| *v <= 64));
                assert!((0.25..4.0).contains(&w.gain));
                if !w.reset {
                    let p = &previous.as_ref().unwrap()[i];
                    assert_eq!(
                        (w.sequence, w.origin, w.gain),
                        (p.sequence, p.origin, p.gain)
                    );
                    assert_eq!(w.start, p.start + 4);
                }
            }
            previous = Some(current);
        }
        let fraction = sampler.cold_windows as f64 / sampler.windows as f64;
        assert!((0.1..=0.2).contains(&fraction), "{fraction}");
        let json = serde_json::to_vec(&sampler).unwrap();
        let mut resumed: Sampler = serde_json::from_slice(&json).unwrap();
        for _ in 0..100 {
            assert_eq!(sampler.next_windows(), resumed.next_windows());
        }
    }
    #[test]
    fn tiny_sequences_are_cold_and_bad_crops_rejected() {
        let mut sampler = Sampler::new(1, 2, [16; 2], 2, [16; 2], 1, 2).unwrap();
        for _ in 0..20 {
            assert!(sampler.next_windows().iter().all(|w| w.reset));
        }
        assert!(Sampler::new(1, 2, [64; 2], 2, [16; 2], 1, 2).is_err());
    }
    #[test]
    fn schedule_has_linear_warmup_and_ten_percent_endpoint() {
        assert_eq!(learning_rate(1.0, 1, 5000), 0.002);
        assert_eq!(learning_rate(1.0, 500, 5000), 1.0);
        assert!((learning_rate(1.0, 5000, 5000) - 0.1).abs() < 1e-7);
    }
}
