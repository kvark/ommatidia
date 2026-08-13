//! Assembling training batches out of a dataset.
//!
//! One batch is `batch` random crops from random samples. The conditioning and
//! the sub-pixel residual come from [`ommatidia::batch`]; what this adds is the
//! corruption process, which is what makes it a diffusion batch rather than a
//! regression one.

use ommatidia::batch::{self, Crop};
use ommatidia::dataset::{Layout, Reader};
use ommatidia::diffusion::{self, Schedule};
use ommatidia::model::{ModelConfig, Objective};
use ommatidia::rng::Rng;

/// The tensors one training step needs.
pub struct Batch {
    /// `[batch, cond_channels, tile, tile]`.
    pub cond: Vec<f32>,
    /// `[batch, target_channels, tile, tile]`, the noised residual. Empty
    /// under [`Objective::Direct`].
    pub x_t: Vec<f32>,
    /// `[batch, time_input_dim]`. Empty under [`Objective::Direct`].
    pub t_emb: Vec<f32>,
    /// `[batch, target_channels, tile, tile]`: the clean scaled residual,
    /// under either objective. See [`ommatidia::diffusion`] for why the
    /// diffusion target is the signal rather than the noise.
    pub target: Vec<f32>,
}

pub struct Batcher {
    reader: Reader,
    layout: Layout,
    config: ModelConfig,
    schedule: Schedule,
    /// Samples the batcher is allowed to draw from.
    ///
    /// Everything past this is held out for scoring, and the batcher never
    /// touches it — that is the whole point of the split.
    train: std::ops::Range<usize>,
    rng: Rng,
    /// Scratch for one slot's clean residual, reused across steps.
    residual: Vec<f32>,
    noise: Vec<f32>,
    noised: Vec<f32>,
}

impl Batcher {
    pub fn new(
        reader: Reader,
        config: ModelConfig,
        schedule: Schedule,
        train: std::ops::Range<usize>,
        seed: u64,
    ) -> Self {
        assert!(!train.is_empty(), "the training split is empty");
        assert!(
            train.end <= reader.len(),
            "the training split overruns the set"
        );
        let layout = *reader.layout();
        let per_slot = (config.target_channels() * config.tile * config.tile) as usize;
        Self {
            reader,
            layout,
            config,
            schedule,
            train,
            rng: Rng::new(seed),
            residual: vec![0.0; per_slot],
            noise: vec![0.0; per_slot],
            noised: vec![0.0; per_slot],
        }
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn reader(&mut self) -> &mut Reader {
        &mut self.reader
    }

    /// A crop position that fits inside the sample.
    fn random_crop(&mut self) -> Crop {
        let tile = self.config.tile;
        let span_x = self.layout.lr_width - tile + 1;
        let span_y = self.layout.lr_height - tile + 1;
        Crop {
            x: self.rng.below(span_x),
            y: self.rng.below(span_y),
            tile,
        }
    }

    /// Build one batch.
    pub fn next(&mut self) -> Result<Batch, ommatidia::dataset::Error> {
        // Cloned up front: the crop draw needs `&mut self`, and the config
        // is small and immutable through the loop.
        let config = self.config.clone();
        let batch_size = config.batch as usize;
        let per_slot = (config.target_channels() * config.tile * config.tile) as usize;

        let mut out = Batch {
            cond: vec![0.0; config.cond_len()],
            x_t: Vec::new(),
            t_emb: Vec::new(),
            target: vec![0.0; config.target_len()],
        };
        let diffusing = config.objective == Objective::Diffusion;
        if diffusing {
            out.x_t = vec![0.0; config.target_len()];
            out.t_emb = vec![0.0; config.time_len()];
        }

        for slot in 0..batch_size {
            let index = self.train.start + self.rng.below(self.train.len() as u32) as usize;
            let sample = self.reader.sample(index)?;
            let crop = self.random_crop();

            batch::write_conditioning(
                &sample,
                &self.layout,
                config.cond_planes,
                crop,
                slot,
                &mut out.cond,
            );
            batch::write_residual(&sample, &self.layout, crop, 0, &config, &mut self.residual);

            if diffusing {
                // Every slot gets its own timestep. Sharing one across the
                // batch would make each step a much noisier estimate of the
                // objective, since the loss varies strongly with the noise
                // level.
                let t = self.rng.below(self.schedule.len() as u32) as usize;
                diffusion::fill_normal(&mut self.rng, &mut self.noise);
                self.schedule
                    .add_noise(&self.residual, &self.noise, t, &mut self.noised);

                out.x_t[slot * per_slot..(slot + 1) * per_slot].copy_from_slice(&self.noised);
                // The target is the clean residual: the network predicts x0.
                out.target[slot * per_slot..(slot + 1) * per_slot].copy_from_slice(&self.residual);

                let embedding =
                    diffusion::timestep_embedding(t, config.time_input_dim as usize, MAX_PERIOD);
                let width = config.time_input_dim as usize;
                out.t_emb[slot * width..(slot + 1) * width].copy_from_slice(&embedding);
            } else {
                out.target[slot * per_slot..(slot + 1) * per_slot].copy_from_slice(&self.residual);
            }
        }

        Ok(out)
    }
}

/// Frequency spread of the timestep embedding, in the usual convention.
pub const MAX_PERIOD: f32 = 10_000.0;

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use ommatidia::dataset::{Plane, PlaneSet, Sample, Writer};

    fn write_dataset(path: &std::path::Path, count: usize) -> Layout {
        let layout = Layout {
            scale: 2,
            lr_width: 16,
            lr_height: 16,
            lr_source: ommatidia::dataset::InputSource::RawRestir,
            lr_planes: PlaneSet::new().with(Plane::Color),
            hr_planes: PlaneSet::new().with(Plane::Color),
        };
        let mut rng = Rng::new(1);
        let mut writer = Writer::create(path, layout).unwrap();
        for _ in 0..count {
            writer
                .write(&Sample {
                    lr: (0..layout.lr_len())
                        .map(|_| f16::from_f32(rng.uniform() * 3.0))
                        .collect(),
                    hr: (0..layout.hr_len())
                        .map(|_| f16::from_f32(rng.uniform() * 3.0))
                        .collect(),
                })
                .unwrap();
        }
        writer.finish().unwrap();
        layout
    }

    fn config(objective: Objective) -> ModelConfig {
        ModelConfig {
            scale: 2,
            tile: 8,
            batch: 3,
            cond_planes: PlaneSet::new().with(Plane::Color),
            time_input_dim: 16,
            objective,
            reconstruction_base: ommatidia::model::ReconstructionBase::Bilinear,
            ..ModelConfig::default()
        }
    }

    #[test]
    fn diffusion_batches_are_shaped_and_corrupted() {
        let dir = std::env::temp_dir().join("ommatidia-batcher-diffusion");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("set.omd");
        write_dataset(&path, 4);

        let config = config(Objective::Diffusion);
        let mut batcher = Batcher::new(
            Reader::open(&path).unwrap(),
            config.clone(),
            Schedule::cosine(100),
            0..4,
            7,
        );
        let batch = batcher.next().unwrap();

        assert_eq!(batch.cond.len(), config.cond_len());
        assert_eq!(batch.x_t.len(), config.target_len());
        assert_eq!(batch.target.len(), config.target_len());
        assert_eq!(batch.t_emb.len(), config.time_len());
        assert!(batch.cond.iter().all(|v| v.is_finite()));
        assert!(batch.x_t.iter().all(|v| v.is_finite()));

        // The target is the clean residual, not the noise: the network
        // predicts x0. The residual lives in compressed space, so scaling by
        // the gain bounds it there.
        let bound = config.residual_gain;
        assert!(
            batch.target.iter().all(|v| v.abs() <= bound),
            "the target is not a clean residual, it left the bound {bound}"
        );

        // The corrupted input, on the other hand, has unit noise mixed in, so
        // it has to reach well past that bound at the higher timesteps.
        let extreme = batch.x_t.iter().filter(|v| v.abs() > bound).count();
        assert!(extreme > 0, "x_t does not look corrupted");
        assert_ne!(batch.x_t, batch.target, "x_t should be the noised target");

        // Slots should draw independent timesteps rather than sharing one.
        let width = config.time_input_dim as usize;
        assert_ne!(
            batch.t_emb[..width],
            batch.t_emb[width..2 * width],
            "every slot should get its own noise level"
        );

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn direct_batches_carry_the_residual_and_no_noise() {
        let dir = std::env::temp_dir().join("ommatidia-batcher-direct");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("set.omd");
        write_dataset(&path, 4);

        let config = config(Objective::Direct);
        let mut batcher = Batcher::new(
            Reader::open(&path).unwrap(),
            config.clone(),
            Schedule::cosine(100),
            0..4,
            7,
        );
        let batch = batcher.next().unwrap();

        assert!(batch.x_t.is_empty(), "direct regression takes no x_t");
        assert!(
            batch.t_emb.is_empty(),
            "direct regression takes no timestep"
        );
        assert_eq!(batch.target.len(), config.target_len());
        // The residual lives in compressed space, so it is bounded.
        assert!(
            batch.target.iter().all(|v| v.abs() < 1.0),
            "the residual left the unit interval"
        );

        std::fs::remove_file(&path).unwrap();
    }

    /// The held-out samples have to stay untouched, or every comparison made
    /// against them is measuring memorisation.
    #[test]
    fn the_batcher_never_reaches_past_its_split() {
        const COUNT: usize = 8;
        const TRAIN: usize = 6;

        let dir = std::env::temp_dir().join("ommatidia-batcher-split");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("set.omd");

        // Sample `i` is filled with the radiance `i`, so a batch says exactly
        // which samples it was drawn from.
        let layout = Layout {
            scale: 2,
            lr_width: 8,
            lr_height: 8,
            lr_source: ommatidia::dataset::InputSource::RawRestir,
            lr_planes: PlaneSet::new().with(Plane::Color),
            hr_planes: PlaneSet::new().with(Plane::Color),
        };
        let mut writer = Writer::create(&path, layout).unwrap();
        for i in 0..COUNT {
            writer
                .write(&Sample {
                    lr: vec![f16::from_f32(i as f32); layout.lr_len()],
                    hr: vec![f16::from_f32(i as f32); layout.hr_len()],
                })
                .unwrap();
        }
        writer.finish().unwrap();

        let mut config = config(Objective::Direct);
        config.tile = layout.lr_width;
        let mut batcher = Batcher::new(
            Reader::open(&path).unwrap(),
            config,
            Schedule::cosine(100),
            0..TRAIN,
            13,
        );

        // The conditioning is the compressed radiance, so the highest value a
        // legitimate batch can carry comes from sample TRAIN - 1.
        let ceiling = ommatidia::transform::compress((TRAIN - 1) as f32);
        let mut seen: f32 = 0.0;
        for _ in 0..400 {
            let batch = batcher.next().unwrap();
            seen = seen.max(batch.cond.iter().copied().fold(0.0, f32::max));
        }
        assert!(
            seen <= ceiling + 1e-4,
            "a batch carried radiance {seen}, above the {ceiling} the training \
             split can produce, so it reached into the held-out samples"
        );
        // And it did draw from the top of the training split, so the bound is
        // tight rather than accidentally satisfied.
        assert!(
            seen > ommatidia::transform::compress((TRAIN - 2) as f32),
            "the batcher never drew the last training sample, so this proves \
             nothing"
        );

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn batches_are_reproducible_from_a_seed() {
        let dir = std::env::temp_dir().join("ommatidia-batcher-seed");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("set.omd");
        write_dataset(&path, 4);

        let make = || {
            Batcher::new(
                Reader::open(&path).unwrap(),
                config(Objective::Diffusion),
                Schedule::cosine(100),
                0..4,
                42,
            )
            .next()
            .unwrap()
        };
        assert_eq!(make().cond, make().cond);
        assert_eq!(make().x_t, make().x_t);

        std::fs::remove_file(&path).unwrap();
    }
}
