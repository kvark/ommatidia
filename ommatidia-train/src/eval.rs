//! Running a trained network over one crop and writing what it produced.
//!
//! A falling loss says the network fits its objective; it says nothing about
//! whether the frame looks right. Under diffusion it says even less, because
//! the training loss measures noise prediction at a random timestep and never
//! walks the sampler at all. So the trainer renders.

use std::path::Path;

use ommatidia::batch::{self, Crop};
use ommatidia::dataset::{Layout, Sample};
use ommatidia::diffusion::{self, Schedule};
use ommatidia::model::{ModelConfig, Objective};
use ommatidia::rng::Rng;

use crate::batcher::MAX_PERIOD;

/// Run the network over one crop and return the reconstructed high resolution
/// image as interleaved linear RGB.
///
/// `session` must have been built from an inference graph with `batch == 1`.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct(
    session: &mut meganeura::Session,
    config: &ModelConfig,
    schedule: &Schedule,
    sample: &Sample,
    layout: &Layout,
    crop: Crop,
    sampler_steps: usize,
    seed: u64,
) -> Vec<f32> {
    assert_eq!(config.batch, 1, "evaluation runs one crop at a time");
    let per_slot = (config.target_channels() * config.tile * config.tile) as usize;

    let mut cond = vec![0.0; config.cond_len()];
    batch::write_conditioning(sample, layout, config.cond_planes, crop, 0, &mut cond);
    session.set_input("cond", &cond);

    let residual = match config.objective {
        Objective::Direct => {
            session.step();
            session.wait();
            session.read_output(per_slot)
        }
        Objective::Diffusion => {
            // Start from pure noise and walk the chain down.
            let mut rng = Rng::new(seed);
            let mut x = vec![0.0; per_slot];
            diffusion::fill_normal(&mut rng, &mut x);

            let steps = schedule.sampling_timesteps(sampler_steps);
            let mut next = vec![0.0; per_slot];
            for (i, &t) in steps.iter().enumerate() {
                let embedding =
                    diffusion::timestep_embedding(t, config.time_input_dim as usize, MAX_PERIOD);
                session.set_input("t_emb", &embedding);
                session.set_input("x_t", &x);
                session.step();
                session.wait();
                let x0 = session.read_output(per_slot);
                schedule.ddim_step(
                    &x,
                    &x0,
                    t,
                    steps.get(i + 1).copied(),
                    // The compressed residual is bounded by 1, so the scaled
                    // one is bounded by the gain.
                    config.residual_gain,
                    &mut next,
                );
                x.copy_from_slice(&next);
            }
            x
        }
    };

    let low = batch::crop_color(sample, layout, crop);
    batch::assemble(
        &low,
        &residual,
        crop.tile as usize,
        crop.tile as usize,
        config.scale as usize,
        config.residual_gain,
    )
}

/// Tone map linear RGB and write it out, so the three images can be compared
/// side by side by eye.
pub fn write_png(path: &Path, rgb: &[f32], width: u32, height: u32) -> std::io::Result<()> {
    assert_eq!(rgb.len(), (width * height * 3) as usize);
    let mut bytes = Vec::with_capacity((width * height * 4) as usize);
    for texel in rgb.chunks_exact(3) {
        for &linear in texel {
            let mapped = ommatidia::transform::compress(linear);
            let encoded = if mapped <= 0.0031308 {
                12.92 * mapped
            } else {
                1.055 * mapped.powf(1.0 / 2.4) - 0.055
            };
            bytes.push((encoded.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
        bytes.push(255);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()?
        .write_image_data(&bytes)
        .map_err(std::io::Error::other)
}

/// Mean squared error between two linear images, in compressed space.
///
/// Compressed rather than linear because a single bright pixel would otherwise
/// dominate the number and hide everything else — the same reason the network
/// trains there.
pub fn error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = ommatidia::transform::compress(x) - ommatidia::transform::compress(y);
            d * d
        })
        .sum();
    sum / a.len() as f32
}

/// Nearest-neighbour upsampling of interleaved linear RGB.
///
/// The baseline the network has to beat: it is what a zero residual produces,
/// so a reported error above this means the network is actively hurting.
pub fn nearest(low: &[f32], width: usize, height: usize, scale: usize) -> Vec<f32> {
    let out_width = width * scale;
    let mut out = vec![0.0; out_width * height * scale * 3];
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let value = low[(y * width + x) * 3 + c];
                for dy in 0..scale {
                    for dx in 0..scale {
                        out[((y * scale + dy) * out_width + x * scale + dx) * 3 + c] = value;
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_replicates_every_pixel() {
        // Two pixels in one row, scaled by 2, so the output is 4x2.
        let low = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let up = nearest(&low, 2, 1, 2);
        assert_eq!(up.len(), 4 * 2 * 3);

        // Both output rows are the input row with each pixel doubled.
        for row in 0..2 {
            let base = row * 4 * 3;
            assert_eq!(&up[base..base + 3], &[1.0, 2.0, 3.0]);
            assert_eq!(&up[base + 3..base + 6], &[1.0, 2.0, 3.0]);
            assert_eq!(&up[base + 6..base + 9], &[4.0, 5.0, 6.0]);
            assert_eq!(&up[base + 9..base + 12], &[4.0, 5.0, 6.0]);
        }
    }

    #[test]
    fn error_is_zero_for_identical_images_and_bounded_otherwise() {
        let a = vec![0.0, 1.0, 100.0, 5.0];
        assert_eq!(error(&a, &a), 0.0);
        // Compressed space keeps a huge outlier from dominating.
        let b = vec![0.0, 1.0, 10_000.0, 5.0];
        assert!(error(&a, &b) < 1.0, "one bright pixel swamped the metric");
    }
}
