//! Running a trained network over one crop and writing what it produced.
//!
//! A falling loss says the network fits its objective; it says nothing about
//! whether the frame looks right. Under diffusion it says even less, because
//! the training loss measures noise prediction at a random timestep and never
//! walks the sampler at all. So the trainer renders.

use std::path::Path;

use ommatidia::batch::{self, Crop};
use ommatidia::dataset::Layout;
use ommatidia::diffusion::{self, Schedule};
use ommatidia::model::{ModelConfig, Objective, Prediction};
use ommatidia::rng::Rng;

use crate::batcher::InputSample;
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
    input: &InputSample,
    layout: &Layout,
    crop: Crop,
    guided: Option<&[f32]>,
    sampler_steps: usize,
    seed: u64,
) -> Vec<f32> {
    assert_eq!(config.batch, 1, "evaluation runs one crop at a time");
    let per_slot = (config.target_channels() * config.tile * config.tile) as usize;

    let mut cond = vec![0.0; config.cond_len()];
    let _ = input.write_conditioning(layout, config, crop, 0, &mut cond);
    session.set_input("cond", &cond);
    let sample = input.sample();

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
    match config.prediction {
        Prediction::SubpixelResidual => {
            batch::assemble(&low, guided, &residual, [crop.tile as usize; 2], config)
        }
        Prediction::LowResolutionResidual => {
            let low = batch::guided_color(sample, layout, crop, config.guide);
            let corrected = batch::assemble_low_resolution(
                &low,
                &residual,
                [crop.tile as usize; 2],
                config.residual_gain,
            );
            let full_crop = Crop {
                x: 0,
                y: 0,
                tile: layout.lr_width,
            };
            let mut full = batch::guided_color(sample, layout, full_crop, config.guide);
            let width = layout.lr_width as usize;
            let tile = crop.tile as usize;
            for y in 0..tile {
                let destination = ((crop.y as usize + y) * width + crop.x as usize) * 3;
                let source = y * tile * 3;
                full[destination..destination + tile * 3]
                    .copy_from_slice(&corrected[source..source + tile * 3]);
            }
            batch::high_resolution_guided_from_color(sample, layout, crop, config.guide, &full)
        }
    }
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

pub use ommatidia::metrics::{error, ssim};

/// Crop the current-to-previous motion and surface-validity mask used by the
/// temporal metric. Confidence above one accumulated sample means the shared
/// rejection path accepted history for that low-resolution pixel.
pub fn temporal_evidence(
    input: &InputSample,
    layout: &Layout,
    crop: Crop,
    history_frames: u32,
) -> Option<(Vec<f32>, Vec<bool>)> {
    let InputSample::Temporal(prepared) = input else {
        return None;
    };
    let motion_x = prepared
        .sample
        .lr_channel(layout, ommatidia::Plane::Motion, 0)?;
    let motion_y = prepared
        .sample
        .lr_channel(layout, ommatidia::Plane::Motion, 1)?;
    let width = layout.lr_width as usize;
    let tile = crop.tile as usize;
    let mut motion = Vec::with_capacity(tile * tile * 2);
    let mut valid = Vec::with_capacity(tile * tile);
    for y in 0..tile {
        for x in 0..tile {
            let index = (crop.y as usize + y) * width + crop.x as usize + x;
            motion.push(motion_x[index].to_f32());
            motion.push(motion_y[index].to_f32());
            valid.push(prepared.confidence[index] * history_frames as f32 > 1.001);
        }
    }
    Some((motion, valid))
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

/// Bilinear upsampling of interleaved linear RGB, with texel centers aligned.
///
/// This is the conventional non-neural reconstruction baseline. Sampling is
/// clamped at the image boundary, matching a GPU linear-filtered texture with
/// clamp-to-edge addressing.
pub fn bilinear(low: &[f32], width: usize, height: usize, scale: usize) -> Vec<f32> {
    assert_eq!(low.len(), width * height * 3);
    assert!(width > 0 && height > 0 && scale > 0);
    let out_width = width * scale;
    let out_height = height * scale;
    let mut out = vec![0.0; out_width * out_height * 3];
    for oy in 0..out_height {
        let fy = (oy as f32 + 0.5) / scale as f32 - 0.5;
        let y0 = fy.floor() as isize;
        let ty = fy - y0 as f32;
        let y0c = y0.clamp(0, height as isize - 1) as usize;
        let y1c = (y0 + 1).clamp(0, height as isize - 1) as usize;
        for ox in 0..out_width {
            let fx = (ox as f32 + 0.5) / scale as f32 - 0.5;
            let x0 = fx.floor() as isize;
            let tx = fx - x0 as f32;
            let x0c = x0.clamp(0, width as isize - 1) as usize;
            let x1c = (x0 + 1).clamp(0, width as isize - 1) as usize;
            for c in 0..3 {
                let p00 = low[(y0c * width + x0c) * 3 + c];
                let p10 = low[(y0c * width + x1c) * 3 + c];
                let p01 = low[(y1c * width + x0c) * 3 + c];
                let p11 = low[(y1c * width + x1c) * 3 + c];
                let top = p00 + tx * (p10 - p00);
                let bottom = p01 + tx * (p11 - p01);
                out[(oy * out_width + ox) * 3 + c] = top + ty * (bottom - top);
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
    fn bilinear_aligns_texel_centers_and_clamps_edges() {
        let low = vec![
            0.0, 0.0, 0.0, // left
            4.0, 8.0, 12.0, // right
        ];
        let up = bilinear(&low, 2, 1, 2);
        let red: Vec<_> = up.chunks_exact(3).map(|rgb| rgb[0]).collect();
        assert_eq!(red, vec![0.0, 1.0, 3.0, 4.0, 0.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn error_is_zero_for_identical_images_and_bounded_otherwise() {
        let a = vec![0.0, 1.0, 100.0, 5.0];
        assert_eq!(error(&a, &a), 0.0);
        // Compressed space keeps a huge outlier from dominating.
        let b = vec![0.0, 1.0, 10_000.0, 5.0];
        assert!(error(&a, &b) < 1.0, "one bright pixel swamped the metric");
    }

    #[test]
    fn ssim_is_one_for_identical_images_and_falls_for_lost_structure() {
        let mut image = vec![0.0; 8 * 8 * 3];
        for (index, value) in image.iter_mut().enumerate() {
            *value = (index % 11) as f32;
        }
        assert!((ssim(&image, &image, 8, 8) - 1.0).abs() < 1e-6);
        let flat = vec![1.0; image.len()];
        assert!(ssim(&image, &flat, 8, 8) < 0.5);
    }
}
