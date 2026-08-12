//! Turning stored samples into network tensors, and back.
//!
//! This is the contract the trainer and the runtime both have to honour. The
//! trainer runs it on the CPU over dataset records; the runtime runs the same
//! arithmetic in `shaders/pack.wgsl` and `shaders/unpack.wgsl` over textures.
//! If the two ever disagree, a network that trained perfectly will produce
//! garbage in the renderer, and nothing will point at why — so the round trip
//! is tested here rather than being left as a convention.
//!
//! # Layouts
//!
//! Conditioning is `[batch, channels, tile, tile]`, channel-major, matching
//! NCHW and the dataset's own plane order.
//!
//! The target is `[batch, 3 * scale^2, tile, tile]`. Sub-pixel `(dy, dx)` of
//! colour channel `c` lives at channel `c * scale^2 + dy * scale + dx`. That
//! ordering keeps a single colour channel's sub-pixels contiguous, so the
//! unpack shader walks them with one strided read per output texel.
//!
//! # Spaces
//!
//! Everything the network sees is in the compressed colour space of
//! [`crate::transform`]. The residual is
//!
//! ```text
//! residual = compress(HR[sub-pixel]) - compress(LR[pixel])
//! ```
//!
//! which is bounded in `(-1, 1)` and centred near zero, and the base it is
//! taken over is nearest-neighbour rather than bilinear so the shader can
//! reproduce it with no edge convention to get wrong.

use half::f16;

use crate::dataset::{Layout, Plane, PlaneSet, Sample};
use crate::transform;

/// A rectangular crop of one sample, in low resolution pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Crop {
    pub x: u32,
    pub y: u32,
    pub tile: u32,
}

impl Crop {
    /// Every crop position that fits, given a stride.
    ///
    /// Used to enumerate a validation set deterministically rather than
    /// sampling it.
    pub fn grid(layout: &Layout, tile: u32, stride: u32) -> Vec<Self> {
        let mut out = Vec::new();
        if layout.lr_width < tile || layout.lr_height < tile {
            return out;
        }
        let mut y = 0;
        while y + tile <= layout.lr_height {
            let mut x = 0;
            while x + tile <= layout.lr_width {
                out.push(Self { x, y, tile });
                x += stride;
            }
            y += stride;
        }
        out
    }
}

/// Convert one stored value into what the network should see for that plane.
fn encode(plane: Plane, value: f16) -> f32 {
    let value = value.to_f32();
    match plane {
        // Radiance is unbounded; the rest already arrive in a sane range.
        Plane::Color => transform::compress(value),
        Plane::Depth => transform::encode_depth(value),
        Plane::Normal
        | Plane::DiffuseAlbedo
        | Plane::SpecularF0
        | Plane::Roughness
        | Plane::Motion => value,
    }
}

/// Write the conditioning planes of one crop into `out` at `slot`.
///
/// `out` is the whole batch tensor, `slot` the index within it.
pub fn write_conditioning(
    sample: &Sample,
    layout: &Layout,
    planes: PlaneSet,
    crop: Crop,
    slot: usize,
    out: &mut [f32],
) {
    let tile = crop.tile as usize;
    let channels = planes.channels();
    let stride = layout.lr_width as usize;
    let texels = layout.lr_texels();
    let per_slot = channels * tile * tile;
    assert!(
        out.len() >= (slot + 1) * per_slot,
        "conditioning tensor is too small for slot {slot}"
    );

    let mut channel = 0;
    for plane in planes.iter() {
        let base = layout
            .lr_planes
            .channel_offset(plane)
            .unwrap_or_else(|| panic!("dataset has no {plane:?} plane"));
        for component in 0..plane.channels() {
            let source = &sample.lr[(base + component) * texels..(base + component + 1) * texels];
            let destination = slot * per_slot + channel * tile * tile;
            for y in 0..tile {
                let row = (crop.y as usize + y) * stride + crop.x as usize;
                for x in 0..tile {
                    out[destination + y * tile + x] = encode(plane, source[row + x]);
                }
            }
            channel += 1;
        }
    }
}

/// Write the sub-pixel residual target of one crop into `out` at `slot`,
/// multiplied by `gain`.
///
/// Both the reference and the base come from the same sample, so the result is
/// exactly what [`assemble`] has to invert.
///
/// # Why the gain
///
/// The raw residual is small: most of a frame is already correct at low
/// resolution, so its standard deviation is a few hundredths. A diffusion
/// schedule assumes unit-variance data, and against unit noise a signal that
/// size is invisible at nearly every timestep. The network then learns the
/// degenerate solution `eps = x_t`, which scores well on the training loss and
/// samples to pure noise, because recovering `x0` from it is a difference of
/// two nearly equal numbers. Scaling the residual to unit variance is what
/// makes the schedule's noise levels meaningful — the same reason latent
/// diffusion models rescale their latents. Use [`estimate_gain`] to measure it.
pub fn write_residual(
    sample: &Sample,
    layout: &Layout,
    crop: Crop,
    slot: usize,
    gain: f32,
    out: &mut [f32],
) {
    let scale = layout.scale as usize;
    let tile = crop.tile as usize;
    let sub = scale * scale;
    let per_slot = 3 * sub * tile * tile;
    assert!(
        out.len() >= (slot + 1) * per_slot,
        "target tensor is too small for slot {slot}"
    );

    let lr_base = layout
        .lr_planes
        .channel_offset(Plane::Color)
        .expect("dataset has no low resolution colour");
    let hr_base = layout
        .hr_planes
        .channel_offset(Plane::Color)
        .expect("dataset has no high resolution colour");
    let lr_texels = layout.lr_texels();
    let hr_texels = layout.hr_texels();
    let lr_stride = layout.lr_width as usize;
    let hr_stride = layout.hr_width() as usize;

    for c in 0..3 {
        let low = &sample.lr[(lr_base + c) * lr_texels..(lr_base + c + 1) * lr_texels];
        let high = &sample.hr[(hr_base + c) * hr_texels..(hr_base + c + 1) * hr_texels];
        for y in 0..tile {
            let source_y = crop.y as usize + y;
            for x in 0..tile {
                let source_x = crop.x as usize + x;
                let base = transform::compress(low[source_y * lr_stride + source_x].to_f32());
                for dy in 0..scale {
                    for dx in 0..scale {
                        let hy = source_y * scale + dy;
                        let hx = source_x * scale + dx;
                        let reference = transform::compress(high[hy * hr_stride + hx].to_f32());
                        let channel = c * sub + dy * scale + dx;
                        out[slot * per_slot + (channel * tile + y) * tile + x] =
                            (reference - base) * gain;
                    }
                }
            }
        }
    }
}

/// Reassemble a high resolution image from a low resolution one and a
/// predicted sub-pixel residual.
///
/// `low` is interleaved RGB linear radiance at `width * height`; `residual` is
/// one slot of the network's output, still multiplied by `gain`. The result is
/// interleaved RGB linear radiance at `scale` times the extent — exactly what
/// the unpack shader writes into the output texture.
pub fn assemble(
    low: &[f32],
    residual: &[f32],
    width: usize,
    height: usize,
    scale: usize,
    gain: f32,
) -> Vec<f32> {
    assert_eq!(low.len(), width * height * 3);
    let sub = scale * scale;
    assert_eq!(residual.len(), 3 * sub * width * height);

    // A zero gain would mean the residual carries no information at all, so
    // fall back to the base rather than dividing by it.
    let inverse_gain = if gain.abs() > f32::MIN_POSITIVE {
        1.0 / gain
    } else {
        0.0
    };
    let out_width = width * scale;
    let mut out = vec![0.0; out_width * height * scale * 3];
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let base = transform::compress(low[(y * width + x) * 3 + c]);
                for dy in 0..scale {
                    for dx in 0..scale {
                        let channel = c * sub + dy * scale + dx;
                        let delta = residual[(channel * height + y) * width + x] * inverse_gain;
                        let value = transform::decompress(base + delta);
                        let hy = y * scale + dy;
                        let hx = x * scale + dx;
                        out[(hy * out_width + hx) * 3 + c] = value;
                    }
                }
            }
        }
    }
    out
}

/// Measure the gain that brings a dataset's residuals to unit variance.
///
/// Returns `1 / std`, computed over whole samples with the gain held at 1.
/// Falls back to 1.0 for a set whose residuals are identically zero, which
/// only happens if the low and high resolution renders are the same image.
///
/// Measuring rather than hardcoding matters because the right value depends on
/// the scale factor, the renderer, and the content: a set of mostly flat walls
/// and one of foliage differ by an order of magnitude.
pub fn estimate_gain(samples: impl IntoIterator<Item = Sample>, layout: &Layout) -> f32 {
    let tile = layout.lr_width.min(layout.lr_height);
    let sub = (layout.scale * layout.scale) as usize;
    let mut scratch = vec![0.0; 3 * sub * (tile * tile) as usize];

    let mut sum = 0.0f64;
    let mut count = 0usize;
    for sample in samples {
        let crop = Crop { x: 0, y: 0, tile };
        write_residual(&sample, layout, crop, 0, 1.0, &mut scratch);
        // The residual is centred on zero by construction, so the mean square
        // is the variance and there is no mean to subtract.
        sum += scratch
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>();
        count += scratch.len();
    }

    if count == 0 {
        return 1.0;
    }
    let deviation = (sum / count as f64).sqrt() as f32;
    if deviation > 1e-6 {
        1.0 / deviation
    } else {
        1.0
    }
}

/// Extract one crop's low resolution colour as interleaved linear RGB.
///
/// The companion to [`write_residual`] for [`assemble`], and what a preview
/// renderer needs to show the input beside the output.
pub fn crop_color(sample: &Sample, layout: &Layout, crop: Crop) -> Vec<f32> {
    let tile = crop.tile as usize;
    let base = layout
        .lr_planes
        .channel_offset(Plane::Color)
        .expect("dataset has no low resolution colour");
    let texels = layout.lr_texels();
    let stride = layout.lr_width as usize;

    let mut out = vec![0.0; tile * tile * 3];
    for c in 0..3 {
        let source = &sample.lr[(base + c) * texels..(base + c + 1) * texels];
        for y in 0..tile {
            let row = (crop.y as usize + y) * stride + crop.x as usize;
            for x in 0..tile {
                out[(y * tile + x) * 3 + c] = source[row + x].to_f32();
            }
        }
    }
    out
}

/// Extract one crop's high resolution colour as interleaved linear RGB.
pub fn crop_reference(sample: &Sample, layout: &Layout, crop: Crop) -> Vec<f32> {
    let scale = layout.scale as usize;
    let tile = crop.tile as usize * scale;
    let base = layout
        .hr_planes
        .channel_offset(Plane::Color)
        .expect("dataset has no high resolution colour");
    let texels = layout.hr_texels();
    let stride = layout.hr_width() as usize;

    let mut out = vec![0.0; tile * tile * 3];
    for c in 0..3 {
        let source = &sample.hr[(base + c) * texels..(base + c + 1) * texels];
        for y in 0..tile {
            let row = (crop.y as usize * scale + y) * stride + crop.x as usize * scale;
            for x in 0..tile {
                out[(y * tile + x) * 3 + c] = source[row + x].to_f32();
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn layout(scale: u32, width: u32, height: u32) -> Layout {
        Layout {
            scale,
            lr_width: width,
            lr_height: height,
            lr_source: crate::dataset::InputSource::RawRestir,
            lr_planes: PlaneSet::new().with(Plane::Color).with(Plane::Depth),
            hr_planes: PlaneSet::new().with(Plane::Color),
        }
    }

    /// A sample whose values are all distinct, so a layout mistake cannot
    /// accidentally land on the right number.
    fn sample(layout: &Layout, seed: u64) -> Sample {
        let mut rng = Rng::new(seed);
        Sample {
            lr: (0..layout.lr_len())
                .map(|_| f16::from_f32(rng.uniform() * 4.0))
                .collect(),
            hr: (0..layout.hr_len())
                .map(|_| f16::from_f32(rng.uniform() * 4.0))
                .collect(),
        }
    }

    #[test]
    fn residual_then_assemble_recovers_the_reference() {
        // The round trip that keeps the trainer and the shader honest.
        for scale in [2u32, 3] {
            let l = layout(scale, 8, 8);
            let s = sample(&l, 1);
            let crop = Crop {
                x: 0,
                y: 0,
                tile: 8,
            };

            let mut residual = vec![0.0; 3 * (scale * scale * 64) as usize];
            write_residual(&s, &l, crop, 0, 1.0, &mut residual);

            let low = crop_color(&s, &l, crop);
            let rebuilt = assemble(&low, &residual, 8, 8, scale as usize, 1.0);
            let reference = crop_reference(&s, &l, crop);

            assert_eq!(rebuilt.len(), reference.len());
            for (i, (a, b)) in rebuilt.iter().zip(reference.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-3,
                    "scale {scale}, element {i}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn a_zero_residual_gives_nearest_neighbour() {
        // What an untrained network produces, and the fallback the runtime
        // degrades to. Every sub-pixel should be its source pixel.
        let l = layout(2, 4, 4);
        let s = sample(&l, 2);
        let crop = Crop {
            x: 0,
            y: 0,
            tile: 4,
        };
        let low = crop_color(&s, &l, crop);
        let rebuilt = assemble(&low, &vec![0.0; 3 * 4 * 16], 4, 4, 2, 1.0);

        for y in 0..4 {
            for x in 0..4 {
                for c in 0..3 {
                    let expected = low[(y * 4 + x) * 3 + c];
                    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let got = rebuilt[((y * 2 + dy) * 8 + x * 2 + dx) * 3 + c];
                        assert!(
                            (got - expected).abs() < 1e-3,
                            "({x},{y}) sub ({dx},{dy}) channel {c}: {got} vs {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn crops_read_the_right_window() {
        let l = layout(2, 8, 8);
        let s = sample(&l, 3);
        let crop = Crop {
            x: 2,
            y: 3,
            tile: 4,
        };
        let cropped = crop_color(&s, &l, crop);

        let base = l.lr_planes.channel_offset(Plane::Color).unwrap();
        let texels = l.lr_texels();
        for y in 0..4 {
            for x in 0..4 {
                for c in 0..3 {
                    let expected = s.lr[(base + c) * texels + (3 + y) * 8 + (2 + x)].to_f32();
                    assert_eq!(cropped[(y * 4 + x) * 3 + c], expected);
                }
            }
        }

        // And the reference crop lines up with it at the scaled offset.
        let reference = crop_reference(&s, &l, crop);
        let hr_base = l.hr_planes.channel_offset(Plane::Color).unwrap();
        let hr_texels = l.hr_texels();
        assert_eq!(
            reference[0],
            s.hr[hr_base * hr_texels + (3 * 2) * 16 + 2 * 2].to_f32()
        );
    }

    #[test]
    fn conditioning_lays_planes_out_in_order() {
        let l = layout(2, 6, 6);
        let s = sample(&l, 4);
        let planes = l.lr_planes;
        let crop = Crop {
            x: 1,
            y: 1,
            tile: 4,
        };

        let per_slot = planes.channels() * 16;
        let mut out = vec![0.0; per_slot * 2];
        write_conditioning(&s, &l, planes, crop, 1, &mut out);

        // Slot 0 untouched, slot 1 written.
        assert!(out[..per_slot].iter().all(|&v| v == 0.0));

        // Colour occupies channels 0..3 and is compressed; depth is channel 3
        // and is inverted.
        let texels = l.lr_texels();
        let colour_base = planes.channel_offset(Plane::Color).unwrap();
        let expected_colour = transform::compress(s.lr[colour_base * texels + 6 + 1].to_f32());
        assert!((out[per_slot] - expected_colour).abs() < 1e-6);

        let depth_base = planes.channel_offset(Plane::Depth).unwrap();
        let expected_depth = transform::encode_depth(s.lr[depth_base * texels + 6 + 1].to_f32());
        assert!((out[per_slot + 3 * 16] - expected_depth).abs() < 1e-6);
    }

    #[test]
    fn residual_is_bounded_and_centred() {
        // The whole reason for working in compressed space: whatever the
        // radiance, the thing the network fits stays inside (-1, 1).
        let mut l = layout(2, 8, 8);
        l.lr_planes = PlaneSet::new().with(Plane::Color);
        let mut rng = Rng::new(9);
        let s = Sample {
            // Deliberately high dynamic range on both sides.
            lr: (0..l.lr_len())
                .map(|_| f16::from_f32(rng.uniform() * 5000.0))
                .collect(),
            hr: (0..l.hr_len())
                .map(|_| f16::from_f32(rng.uniform() * 5000.0))
                .collect(),
        };

        let mut residual = vec![0.0; 3 * 4 * 64];
        write_residual(
            &s,
            &l,
            Crop {
                x: 0,
                y: 0,
                tile: 8,
            },
            0,
            1.0,
            &mut residual,
        );
        assert!(
            residual.iter().all(|v| v.abs() < 1.0),
            "residual left the unit interval"
        );
    }

    #[test]
    fn crop_grid_covers_without_overrunning() {
        let l = layout(2, 10, 6);
        let crops = Crop::grid(&l, 4, 4);
        assert_eq!(crops.len(), 2, "one row of two fits in 10x6");
        assert!(
            crops
                .iter()
                .all(|c| c.x + c.tile <= 10 && c.y + c.tile <= 6)
        );
        // A tile larger than the image yields nothing rather than panicking.
        assert!(Crop::grid(&l, 16, 4).is_empty());
    }
}
