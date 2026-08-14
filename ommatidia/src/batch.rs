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
//! residual = compress(HR[sub-pixel]) - compress(reconstruct(LR, sub-pixel))
//! ```
//!
//! which is bounded in `(-1, 1)` and centred near zero. New checkpoints use an
//! explicitly specified renderer-guided, texel-center-aligned bilinear base;
//! the historical nearest and controlled plain-bilinear bases remain in the
//! sidecar contract so old and ablation weights are interpreted correctly.

use half::f16;

use crate::dataset::{Layout, Plane, PlaneSet, Sample};
use crate::model::{GuideConfig, ModelConfig, Prediction, ReconstructionBase};
use crate::temporal::PreparedSample;
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

/// Write the conditioning expected by a temporal checkpoint.
///
/// Stored planes come from the prepared sample, whose colour is the safely
/// accumulated estimate. Original current-frame RGB and normalized history
/// confidence follow those planes. Keeping these as ordinary channels adds
/// only stem-convolution weights; it needs no new graph operation.
pub fn write_temporal_conditioning(
    prepared: &PreparedSample,
    layout: &Layout,
    config: &ModelConfig,
    crop: Crop,
    slot: usize,
    out: &mut [f32],
) -> Vec<f32> {
    assert!(config.temporal.is_some(), "checkpoint is not temporal");
    let tile = crop.tile as usize;
    let texels = tile * tile;
    let channels = config.cond_channels() as usize;
    let per_slot = channels * texels;
    assert!(out.len() >= (slot + 1) * per_slot);
    let destination = &mut out[slot * per_slot..(slot + 1) * per_slot];
    write_conditioning(
        &prepared.sample,
        layout,
        config.cond_planes,
        crop,
        0,
        destination,
    );

    let stored_channels = config.cond_planes.channels();
    let stride = layout.lr_width as usize;
    for component in 0..3 {
        let base = (stored_channels + component) * texels;
        for y in 0..tile {
            let source_row = (crop.y as usize + y) * stride + crop.x as usize;
            for x in 0..tile {
                let value = prepared.current_color[(source_row + x) * 3 + component];
                destination[base + y * tile + x] = transform::compress(value);
            }
        }
    }
    let base = (stored_channels + 3) * texels;
    for y in 0..tile {
        let source_row = (crop.y as usize + y) * stride + crop.x as usize;
        for x in 0..tile {
            destination[base + y * tile + x] = prepared.confidence[source_row + x];
        }
    }
    let guided = guided_color(&prepared.sample, layout, crop, config.guide);
    for component in 0..3 {
        let base = (stored_channels + 4 + component) * texels;
        for y in 0..tile {
            for x in 0..tile {
                destination[base + y * tile + x] =
                    transform::compress(guided[(y * tile + x) * 3 + component]);
            }
        }
    }
    guided
}

const GUIDE_RADIUS: i32 = 6;

fn plane_value(
    sample: &Sample,
    layout: &Layout,
    plane: Plane,
    component: usize,
    x: usize,
    y: usize,
) -> f32 {
    let channel = layout
        .lr_planes
        .channel_offset(plane)
        .unwrap_or_else(|| panic!("dataset has no {plane:?} plane"))
        + component;
    sample.lr[channel * layout.lr_texels() + y * layout.lr_width as usize + x].to_f32()
}

fn hr_plane_value(
    sample: &Sample,
    layout: &Layout,
    plane: Plane,
    component: usize,
    x: usize,
    y: usize,
) -> f32 {
    let channel = layout
        .hr_planes
        .channel_offset(plane)
        .unwrap_or_else(|| panic!("dataset has no high-resolution {plane:?} plane"))
        + component;
    sample.hr[channel * layout.hr_texels() + y * layout.hr_width() as usize + x].to_f32()
}

fn guide_similarity(
    guide: GuideConfig,
    center_depth: f32,
    center_normal: [f32; 3],
    center_albedo: [f32; 3],
    depth: f32,
    normal: [f32; 3],
    albedo: [f32; 3],
) -> f32 {
    let depth_denominator = 2.0 * guide.depth_sigma * guide.depth_sigma;
    let albedo_denominator = 2.0 * guide.albedo_sigma * guide.albedo_sigma;
    let mut weight = (-(depth - center_depth).powi(2) / depth_denominator).exp();

    let center_normal_len2 = center_normal.iter().map(|v| v * v).sum::<f32>();
    let normal_len2 = normal.iter().map(|v| v * v).sum::<f32>();
    let normal_weight = if center_normal_len2 < 0.25 {
        (normal_len2 < 0.25) as u8 as f32
    } else if normal_len2 < 0.25 {
        0.0
    } else {
        let dot = normal
            .iter()
            .zip(center_normal)
            .map(|(a, b)| a * b)
            .sum::<f32>()
            / (normal_len2 * center_normal_len2).sqrt();
        dot.max(0.0).powf(guide.normal_power)
    };
    weight *= normal_weight;

    let albedo_delta2 = (0..3)
        .map(|component| (albedo[component] - center_albedo[component]).powi(2))
        .sum::<f32>();
    weight * (-albedo_delta2 / albedo_denominator).exp()
}

fn guided_texel(
    sample: &Sample,
    layout: &Layout,
    center_x: i32,
    center_y: i32,
    guide: GuideConfig,
) -> [f32; 3] {
    let width = layout.lr_width as i32;
    let height = layout.lr_height as i32;
    let cx = center_x.clamp(0, width - 1) as usize;
    let cy = center_y.clamp(0, height - 1) as usize;
    let center_depth =
        transform::encode_depth(plane_value(sample, layout, Plane::Depth, 0, cx, cy));
    let center_normal = [
        plane_value(sample, layout, Plane::Normal, 0, cx, cy),
        plane_value(sample, layout, Plane::Normal, 1, cx, cy),
        plane_value(sample, layout, Plane::Normal, 2, cx, cy),
    ];
    let center_albedo = [
        plane_value(sample, layout, Plane::DiffuseAlbedo, 0, cx, cy),
        plane_value(sample, layout, Plane::DiffuseAlbedo, 1, cx, cy),
        plane_value(sample, layout, Plane::DiffuseAlbedo, 2, cx, cy),
    ];
    let spatial_denominator = 2.0 * guide.spatial_sigma * guide.spatial_sigma;
    let mut sum = [0.0f32; 3];
    let mut weight_sum = 0.0f32;

    for dy in -GUIDE_RADIUS..=GUIDE_RADIUS {
        let y = (center_y + dy).clamp(0, height - 1) as usize;
        for dx in -GUIDE_RADIUS..=GUIDE_RADIUS {
            let x = (center_x + dx).clamp(0, width - 1) as usize;
            let distance2 = (dx * dx + dy * dy) as f32;
            let mut weight = (-distance2 / spatial_denominator).exp();
            let depth = transform::encode_depth(plane_value(sample, layout, Plane::Depth, 0, x, y));

            let normal = [
                plane_value(sample, layout, Plane::Normal, 0, x, y),
                plane_value(sample, layout, Plane::Normal, 1, x, y),
                plane_value(sample, layout, Plane::Normal, 2, x, y),
            ];
            let albedo = [
                plane_value(sample, layout, Plane::DiffuseAlbedo, 0, x, y),
                plane_value(sample, layout, Plane::DiffuseAlbedo, 1, x, y),
                plane_value(sample, layout, Plane::DiffuseAlbedo, 2, x, y),
            ];
            weight *= guide_similarity(
                guide,
                center_depth,
                center_normal,
                center_albedo,
                depth,
                normal,
                albedo,
            );

            for (component, value) in sum.iter_mut().enumerate() {
                *value += weight * plane_value(sample, layout, Plane::Color, component, x, y);
            }
            weight_sum += weight;
        }
    }
    for value in &mut sum {
        *value /= weight_sum.max(1e-12);
    }
    sum
}

/// Geometry-guided denoising at input resolution followed by exact bilinear
/// reconstruction of one crop. The filter reads beyond crop boundaries, just
/// as the full-frame GPU path does.
pub fn guided_base(sample: &Sample, layout: &Layout, crop: Crop, guide: GuideConfig) -> Vec<f32> {
    let tile = crop.tile as usize;
    let padded_width = tile + 2;
    let origin_x = crop.x as i32 - 1;
    let origin_y = crop.y as i32 - 1;
    let mut padded = vec![0.0f32; padded_width * padded_width * 3];
    for y in 0..padded_width {
        for x in 0..padded_width {
            let color = guided_texel(
                sample,
                layout,
                origin_x + x as i32,
                origin_y + y as i32,
                guide,
            );
            padded[(y * padded_width + x) * 3..(y * padded_width + x) * 3 + 3]
                .copy_from_slice(&color);
        }
    }

    let scale = layout.scale as usize;
    let out_width = tile * scale;
    let mut out = vec![0.0f32; out_width * out_width * 3];
    for oy in 0..out_width {
        let global_y = crop.y as usize * scale + oy;
        let (y0, y1, ty) = bilinear_axis(global_y, layout.lr_height as usize, scale);
        for ox in 0..out_width {
            let global_x = crop.x as usize * scale + ox;
            let (x0, x1, tx) = bilinear_axis(global_x, layout.lr_width as usize, scale);
            let local = |x: usize, y: usize, component: usize| {
                let px = (x as i32 - origin_x) as usize;
                let py = (y as i32 - origin_y) as usize;
                padded[(py * padded_width + px) * 3 + component]
            };
            for component in 0..3 {
                let top = local(x0, y0, component)
                    + tx * (local(x1, y0, component) - local(x0, y0, component));
                let bottom = local(x0, y1, component)
                    + tx * (local(x1, y1, component) - local(x0, y1, component));
                out[(oy * out_width + ox) * 3 + component] = top + ty * (bottom - top);
            }
        }
    }
    out
}

/// Geometry-guided denoised colour at input resolution.
pub fn guided_color(sample: &Sample, layout: &Layout, crop: Crop, guide: GuideConfig) -> Vec<f32> {
    let tile = crop.tile as usize;
    let mut out = vec![0.0; tile * tile * 3];
    for y in 0..tile {
        for x in 0..tile {
            let color = guided_texel(
                sample,
                layout,
                crop.x as i32 + x as i32,
                crop.y as i32 + y as i32,
                guide,
            );
            out[(y * tile + x) * 3..(y * tile + x) * 3 + 3].copy_from_slice(&color);
        }
    }
    out
}

/// Joint bilateral upsampling using an optional high-resolution primary
/// surface pass. Unlike bilinear reconstruction, this can place an edge
/// between low-resolution texel centres when the renderer supplies its exact
/// high-resolution depth, normal, and albedo.
pub fn high_resolution_guided_base(
    sample: &Sample,
    layout: &Layout,
    crop: Crop,
    guide: GuideConfig,
) -> Vec<f32> {
    high_resolution_guided(sample, layout, crop, guide, None)
}

/// Joint bilateral upsampling of an already-denoised low-resolution colour.
///
/// Unlike [`high_resolution_guided_base`], this skips the 13x13 low-resolution
/// filter. It is the reconstruction path for a model that predicts clean RGB
/// rather than an unpredictable high-resolution sub-pixel residual.
pub fn high_resolution_guided_from_color(
    sample: &Sample,
    layout: &Layout,
    crop: Crop,
    guide: GuideConfig,
    low_color: &[f32],
) -> Vec<f32> {
    assert_eq!(low_color.len(), layout.lr_texels() * 3);
    high_resolution_guided(sample, layout, crop, guide, Some(low_color))
}

fn high_resolution_guided(
    sample: &Sample,
    layout: &Layout,
    crop: Crop,
    guide: GuideConfig,
    low_color: Option<&[f32]>,
) -> Vec<f32> {
    const RADIUS: i32 = 2;
    const PADDING: i32 = RADIUS + 1;
    const SPATIAL_SIGMA: f32 = 1.5;
    let tile = crop.tile as usize;
    let scale = layout.scale as usize;
    let padded_width = tile + 2 * PADDING as usize;
    let origin_x = crop.x as i32 - PADDING;
    let origin_y = crop.y as i32 - PADDING;
    let mut low = vec![[0.0f32; 3]; padded_width * padded_width];
    for y in 0..padded_width {
        for x in 0..padded_width {
            let source_x = (origin_x + x as i32).clamp(0, layout.lr_width as i32 - 1);
            let source_y = (origin_y + y as i32).clamp(0, layout.lr_height as i32 - 1);
            low[y * padded_width + x] = if let Some(color) = low_color {
                let index = (source_y as usize * layout.lr_width as usize + source_x as usize) * 3;
                [color[index], color[index + 1], color[index + 2]]
            } else {
                guided_texel(sample, layout, source_x, source_y, guide)
            };
        }
    }

    let out_width = tile * scale;
    let mut out = vec![0.0f32; out_width * out_width * 3];
    let spatial_denominator = 2.0 * SPATIAL_SIGMA * SPATIAL_SIGMA;
    for oy in 0..out_width {
        let global_y = crop.y as usize * scale + oy;
        let position_y = (global_y as f32 + 0.5) / scale as f32 - 0.5;
        for ox in 0..out_width {
            let global_x = crop.x as usize * scale + ox;
            let position_x = (global_x as f32 + 0.5) / scale as f32 - 0.5;
            let center_depth = transform::encode_depth(hr_plane_value(
                sample,
                layout,
                Plane::Depth,
                0,
                global_x,
                global_y,
            ));
            let center_normal = [
                hr_plane_value(sample, layout, Plane::Normal, 0, global_x, global_y),
                hr_plane_value(sample, layout, Plane::Normal, 1, global_x, global_y),
                hr_plane_value(sample, layout, Plane::Normal, 2, global_x, global_y),
            ];
            let center_albedo = [
                hr_plane_value(sample, layout, Plane::DiffuseAlbedo, 0, global_x, global_y),
                hr_plane_value(sample, layout, Plane::DiffuseAlbedo, 1, global_x, global_y),
                hr_plane_value(sample, layout, Plane::DiffuseAlbedo, 2, global_x, global_y),
            ];

            let base_x = position_x.floor() as i32;
            let base_y = position_y.floor() as i32;
            let mut sum = [0.0f32; 3];
            let mut weight_sum = 0.0f32;
            for dy in -RADIUS..=RADIUS {
                let source_y = (base_y + dy).clamp(0, layout.lr_height as i32 - 1);
                for dx in -RADIUS..=RADIUS {
                    let source_x = (base_x + dx).clamp(0, layout.lr_width as i32 - 1);
                    let distance2 = (source_x as f32 - position_x).powi(2)
                        + (source_y as f32 - position_y).powi(2);
                    let mut weight = (-distance2 / spatial_denominator).exp();
                    let x = source_x as usize;
                    let y = source_y as usize;
                    let depth =
                        transform::encode_depth(plane_value(sample, layout, Plane::Depth, 0, x, y));
                    let normal = [
                        plane_value(sample, layout, Plane::Normal, 0, x, y),
                        plane_value(sample, layout, Plane::Normal, 1, x, y),
                        plane_value(sample, layout, Plane::Normal, 2, x, y),
                    ];
                    let albedo = [
                        plane_value(sample, layout, Plane::DiffuseAlbedo, 0, x, y),
                        plane_value(sample, layout, Plane::DiffuseAlbedo, 1, x, y),
                        plane_value(sample, layout, Plane::DiffuseAlbedo, 2, x, y),
                    ];
                    weight *= guide_similarity(
                        guide,
                        center_depth,
                        center_normal,
                        center_albedo,
                        depth,
                        normal,
                        albedo,
                    );
                    let local_x = (source_x - origin_x) as usize;
                    let local_y = (source_y - origin_y) as usize;
                    for component in 0..3 {
                        sum[component] += weight * low[local_y * padded_width + local_x][component];
                    }
                    weight_sum += weight;
                }
            }
            for component in 0..3 {
                out[(oy * out_width + ox) * 3 + component] = sum[component] / weight_sum.max(1e-12);
            }
        }
    }
    out
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
    config: &ModelConfig,
    out: &mut [f32],
) {
    let gain = config.residual_gain;
    let reconstruction_base = config.reconstruction_base;
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
    let guided = match reconstruction_base {
        ReconstructionBase::GuidedBilinear => Some(guided_base(sample, layout, crop, config.guide)),
        ReconstructionBase::HighResolutionGuided => Some(high_resolution_guided_base(
            sample,
            layout,
            crop,
            config.guide,
        )),
        ReconstructionBase::Nearest | ReconstructionBase::Bilinear => None,
    };

    for c in 0..3 {
        let low = &sample.lr[(lr_base + c) * lr_texels..(lr_base + c + 1) * lr_texels];
        let high = &sample.hr[(hr_base + c) * hr_texels..(hr_base + c + 1) * hr_texels];
        for y in 0..tile {
            let source_y = crop.y as usize + y;
            for x in 0..tile {
                let source_x = crop.x as usize + x;
                for dy in 0..scale {
                    for dx in 0..scale {
                        let base = match reconstruction_base {
                            ReconstructionBase::Nearest => {
                                low[source_y * lr_stride + source_x].to_f32()
                            }
                            ReconstructionBase::Bilinear => sample_bilinear_planar(
                                low,
                                layout.lr_width as usize,
                                layout.lr_height as usize,
                                source_x * scale + dx,
                                source_y * scale + dy,
                                scale,
                            ),
                            ReconstructionBase::GuidedBilinear
                            | ReconstructionBase::HighResolutionGuided => guided.as_ref().unwrap()
                                [((y * scale + dy) * tile * scale + x * scale + dx) * 3 + c],
                        };
                        let base = transform::compress(base);
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

/// Write the target selected by [`ModelConfig::prediction`].
pub fn write_target(
    sample: &Sample,
    layout: &Layout,
    crop: Crop,
    slot: usize,
    config: &ModelConfig,
    out: &mut [f32],
) {
    match config.prediction {
        Prediction::SubpixelResidual => write_residual(sample, layout, crop, slot, config, out),
        Prediction::LowResolutionResidual => {
            write_low_resolution_residual(sample, layout, crop, slot, config, out)
        }
    }
}

/// Train a three-channel low-resolution correction against a box-filtered
/// canonical target. This is a denoising target: unlike sub-pixel residuals,
/// it asks the network to estimate one stable radiance value per input pixel.
pub fn write_low_resolution_residual(
    sample: &Sample,
    layout: &Layout,
    crop: Crop,
    slot: usize,
    config: &ModelConfig,
    out: &mut [f32],
) {
    write_low_resolution_residual_from_base(sample, layout, crop, slot, config, None, out)
}

/// Variant of [`write_low_resolution_residual`] that reuses a guided crop
/// already computed while packing temporal conditioning.
pub fn write_low_resolution_residual_from_base(
    sample: &Sample,
    layout: &Layout,
    crop: Crop,
    slot: usize,
    config: &ModelConfig,
    guided: Option<&[f32]>,
    out: &mut [f32],
) {
    let tile = crop.tile as usize;
    let per_slot = 3 * tile * tile;
    assert!(out.len() >= (slot + 1) * per_slot);
    let scale = layout.scale as usize;
    let hr_texels = layout.hr_texels();
    let hr_base = layout.hr_planes.channel_offset(Plane::Color).unwrap();
    let hr_width = layout.hr_width() as usize;
    for channel in 0..3 {
        let high = &sample.hr[(hr_base + channel) * hr_texels..(hr_base + channel + 1) * hr_texels];
        for y in 0..tile {
            let source_y = crop.y as usize + y;
            for x in 0..tile {
                let source_x = crop.x as usize + x;
                let mut reference = 0.0;
                for dy in 0..scale {
                    for dx in 0..scale {
                        reference += high
                            [(source_y * scale + dy) * hr_width + source_x * scale + dx]
                            .to_f32();
                    }
                }
                reference /= (scale * scale) as f32;
                let base = guided.map_or_else(
                    || {
                        guided_texel(
                            sample,
                            layout,
                            source_x as i32,
                            source_y as i32,
                            config.guide,
                        )[channel]
                    },
                    |guided| guided[(y * tile + x) * 3 + channel],
                );
                out[slot * per_slot + (channel * tile + y) * tile + x] =
                    (transform::compress(reference) - transform::compress(base))
                        * config.residual_gain;
            }
        }
    }
}

/// Apply a predicted planar low-resolution RGB correction to an interleaved
/// linear colour crop.
pub fn assemble_low_resolution(
    low: &[f32],
    residual: &[f32],
    extent: [usize; 2],
    gain: f32,
) -> Vec<f32> {
    let [width, height] = extent;
    assert_eq!(low.len(), width * height * 3);
    assert_eq!(residual.len(), width * height * 3);
    let inverse_gain = gain.recip();
    let mut out = vec![0.0; low.len()];
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                let delta = residual[(channel * height + y) * width + x] * inverse_gain;
                out[(y * width + x) * 3 + channel] = transform::decompress(
                    transform::compress(low[(y * width + x) * 3 + channel]) + delta,
                );
            }
        }
    }
    out
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
    guided: Option<&[f32]>,
    residual: &[f32],
    extent: [usize; 2],
    config: &ModelConfig,
) -> Vec<f32> {
    let [width, height] = extent;
    let scale = config.scale as usize;
    let gain = config.residual_gain;
    let reconstruction_base = config.reconstruction_base;
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
                for dy in 0..scale {
                    for dx in 0..scale {
                        let base = match reconstruction_base {
                            ReconstructionBase::Nearest => low[(y * width + x) * 3 + c],
                            ReconstructionBase::Bilinear => sample_bilinear_interleaved(
                                low,
                                width,
                                height,
                                x * scale + dx,
                                y * scale + dy,
                                scale,
                                c,
                            ),
                            ReconstructionBase::GuidedBilinear
                            | ReconstructionBase::HighResolutionGuided => guided
                                .expect("guided reconstruction needs a prefiltered base")
                                [((y * scale + dy) * out_width + x * scale + dx) * 3 + c],
                        };
                        let base = transform::compress(base);
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
pub fn estimate_gain(
    samples: impl IntoIterator<Item = Sample>,
    layout: &Layout,
    config: &ModelConfig,
) -> f32 {
    let tile = layout.lr_width.min(layout.lr_height);
    let mut scratch = vec![0.0; (config.target_channels() * tile * tile) as usize];

    let mut sum = 0.0f64;
    let mut count = 0usize;
    let mut unit = config.clone();
    unit.residual_gain = 1.0;
    for sample in samples {
        let crop = Crop { x: 0, y: 0, tile };
        write_target(&sample, layout, crop, 0, &unit, &mut scratch);
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

fn bilinear_axis(output: usize, input_extent: usize, scale: usize) -> (usize, usize, f32) {
    let position = (output as f32 + 0.5) / scale as f32 - 0.5;
    let lower = position.floor() as isize;
    let fraction = position - lower as f32;
    let a = lower.clamp(0, input_extent as isize - 1) as usize;
    let b = (lower + 1).clamp(0, input_extent as isize - 1) as usize;
    (a, b, fraction)
}

fn sample_bilinear_planar(
    image: &[f16],
    width: usize,
    height: usize,
    output_x: usize,
    output_y: usize,
    scale: usize,
) -> f32 {
    let (x0, x1, tx) = bilinear_axis(output_x, width, scale);
    let (y0, y1, ty) = bilinear_axis(output_y, height, scale);
    let p00 = image[y0 * width + x0].to_f32();
    let p10 = image[y0 * width + x1].to_f32();
    let p01 = image[y1 * width + x0].to_f32();
    let p11 = image[y1 * width + x1].to_f32();
    let top = p00 + tx * (p10 - p00);
    let bottom = p01 + tx * (p11 - p01);
    top + ty * (bottom - top)
}

fn sample_bilinear_interleaved(
    image: &[f32],
    width: usize,
    height: usize,
    output_x: usize,
    output_y: usize,
    scale: usize,
    channel: usize,
) -> f32 {
    let (x0, x1, tx) = bilinear_axis(output_x, width, scale);
    let (y0, y1, ty) = bilinear_axis(output_y, height, scale);
    let at = |x, y| image[(y * width + x) * 3 + channel];
    let top = at(x0, y0) + tx * (at(x1, y0) - at(x0, y0));
    let bottom = at(x0, y1) + tx * (at(x1, y1) - at(x0, y1));
    top + ty * (bottom - top)
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

    fn reconstruction_config(scale: u32, base: ReconstructionBase) -> ModelConfig {
        ModelConfig {
            scale,
            reconstruction_base: base,
            guide: GuideConfig::TUNED,
            ..ModelConfig::default()
        }
    }

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
            let config = reconstruction_config(scale, ReconstructionBase::Bilinear);
            write_residual(&s, &l, crop, 0, &config, &mut residual);

            let low = crop_color(&s, &l, crop);
            let rebuilt = assemble(&low, None, &residual, [8, 8], &config);
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
    fn guided_residual_round_trip_recovers_the_reference() {
        let mut l = layout(2, 8, 8);
        l.lr_planes = l.lr_planes.with(Plane::Normal).with(Plane::DiffuseAlbedo);
        let s = sample(&l, 7);
        let crop = Crop {
            x: 1,
            y: 1,
            tile: 6,
        };
        let config = reconstruction_config(2, ReconstructionBase::GuidedBilinear);
        let guided = guided_base(&s, &l, crop, config.guide);
        let mut residual = vec![0.0; 3 * 4 * 36];
        write_residual(&s, &l, crop, 0, &config, &mut residual);
        let low = crop_color(&s, &l, crop);
        let rebuilt = assemble(&low, Some(&guided), &residual, [6, 6], &config);
        let reference = crop_reference(&s, &l, crop);
        for (actual, expected) in rebuilt.iter().zip(reference) {
            assert!((actual - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn high_resolution_guided_residual_round_trip_recovers_the_reference() {
        let mut l = layout(2, 8, 8);
        l.lr_planes = l.lr_planes.with(Plane::Normal).with(Plane::DiffuseAlbedo);
        l.hr_planes = l
            .hr_planes
            .with(Plane::Depth)
            .with(Plane::Normal)
            .with(Plane::DiffuseAlbedo);
        let s = sample(&l, 9);
        let crop = Crop {
            x: 1,
            y: 1,
            tile: 6,
        };
        let config = reconstruction_config(2, ReconstructionBase::HighResolutionGuided);
        let guided = high_resolution_guided_base(&s, &l, crop, config.guide);
        let mut residual = vec![0.0; 3 * 4 * 36];
        write_residual(&s, &l, crop, 0, &config, &mut residual);
        let low = crop_color(&s, &l, crop);
        let rebuilt = assemble(&low, Some(&guided), &residual, [6, 6], &config);
        let reference = crop_reference(&s, &l, crop);
        for (actual, expected) in rebuilt.iter().zip(reference) {
            assert!((actual - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn zero_low_resolution_correction_preserves_the_guided_base() {
        let mut l = layout(2, 8, 8);
        l.lr_planes = l.lr_planes.with(Plane::Normal).with(Plane::DiffuseAlbedo);
        l.hr_planes = l
            .hr_planes
            .with(Plane::Depth)
            .with(Plane::Normal)
            .with(Plane::DiffuseAlbedo);
        let s = sample(&l, 10);
        let crop = Crop {
            x: 0,
            y: 0,
            tile: 8,
        };
        let config = reconstruction_config(2, ReconstructionBase::HighResolutionGuided);
        let expected = high_resolution_guided_base(&s, &l, crop, config.guide);
        let guided = guided_color(&s, &l, crop, config.guide);
        let corrected = assemble_low_resolution(&guided, &[0.0; 3 * 64], [8, 8], 1.0);
        let actual = high_resolution_guided_from_color(&s, &l, crop, config.guide, &corrected);
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
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
        let config = reconstruction_config(2, ReconstructionBase::Nearest);
        let rebuilt = assemble(&low, None, &vec![0.0; 3 * 4 * 16], [4, 4], &config);

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
        let config = reconstruction_config(2, ReconstructionBase::Bilinear);
        write_residual(
            &s,
            &l,
            Crop {
                x: 0,
                y: 0,
                tile: 8,
            },
            0,
            &config,
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
