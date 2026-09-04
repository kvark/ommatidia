//! CPU reference for reprojecting sparse path samples across a sequence.
//!
//! Training uses this implementation to construct the exact temporal evidence
//! a future GPU pack path will provide. The accumulated colour replaces the
//! sample's noisy colour for deterministic reconstruction, while the original
//! current-frame colour and a confidence value remain available to the model.

use half::f16;
use serde::{Deserialize, Serialize};

use crate::dataset::{Layout, Plane, Reader, Sample};

/// Conservative primary-surface rejection thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RejectionConfig {
    /// Maximum absolute difference in encoded depth, `1 / (1 + depth)`.
    pub depth_delta: f32,
    /// Minimum cosine between current and previous world normals.
    pub normal_cosine: f32,
    /// Maximum squared RGB diffuse-albedo difference.
    pub albedo_delta2: f32,
}

/// A primary surface at one pixel.
///
/// History accumulation reads this from the noisy low-resolution G-buffer.
/// The teacher's reprojection reads it from the converged high-resolution
/// one. Same test, different buffers — that is the point of giving the
/// teacher its own occlusion handling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surface {
    /// View-space distance from the camera, as stored.
    pub depth: f32,
    pub normal: [f32; 3],
    pub albedo: [f32; 3],
}

impl Surface {
    /// Depth at or beyond this is treated as sky.
    pub const SKY_DEPTH: f32 = 60_000.0;

    pub fn is_sky(self) -> bool {
        self.depth >= Self::SKY_DEPTH
    }

    /// Whether `previous` is the same primary surface as `self`.
    pub fn matches(self, previous: Self, config: RejectionConfig) -> bool {
        if self.is_sky() || previous.is_sky() {
            return self.is_sky() == previous.is_sky();
        }
        let encoded = |depth: f32| crate::transform::encode_depth(depth);
        if (encoded(self.depth) - encoded(previous.depth)).abs() > config.depth_delta {
            return false;
        }
        let mut normal_dot = 0.0;
        let mut current_len2 = 0.0;
        let mut previous_len2 = 0.0;
        let mut albedo_delta2 = 0.0;
        for channel in 0..3 {
            normal_dot += self.normal[channel] * previous.normal[channel];
            current_len2 += self.normal[channel] * self.normal[channel];
            previous_len2 += previous.normal[channel] * previous.normal[channel];
            let delta = self.albedo[channel] - previous.albedo[channel];
            albedo_delta2 += delta * delta;
        }
        let cosine = normal_dot / (current_len2 * previous_len2).sqrt().max(1.0e-6);
        cosine > config.normal_cosine && albedo_delta2 < config.albedo_delta2
    }
}

/// The geometry a motion-compensated sample needs, and the test that decides
/// whether a bilinear tap is the same surface.
#[derive(Clone, Copy)]
pub struct Reprojection<'a> {
    /// Current-to-previous motion at either input or output resolution.
    ///
    /// Input-resolution vectors are in input pixels and are scaled for every
    /// output sub-pixel. Output-resolution vectors are in output pixels and
    /// preserve motion at silhouettes instead of sharing one vector across an
    /// entire reconstruction footprint.
    pub motion: &'a [f32],
    /// High-resolution surfaces of this frame, output-pixel major.
    pub current: &'a [Surface],
    /// High-resolution surfaces of the previous frame, output-pixel major.
    pub previous: &'a [Surface],
    pub rejection: RejectionConfig,
}

impl Reprojection<'_> {
    /// Motion for one output pixel, expressed in output pixels.
    ///
    /// Keeping the two supported layouts distinguishable by their exact size
    /// lets old datasets remain valid while newer captures carry precise
    /// output-resolution motion.
    pub fn output_motion(
        self,
        x: usize,
        y: usize,
        low_extent: [usize; 2],
        scale: usize,
    ) -> [f32; 2] {
        let [low_width, low_height] = low_extent;
        let width = low_width * scale;
        let height = low_height * scale;
        assert!(x < width && y < height);
        let high_len = width * height * 2;
        if self.motion.len() == high_len {
            let index = (y * width + x) * 2;
            return [self.motion[index], self.motion[index + 1]];
        }
        assert_eq!(self.motion.len(), low_width * low_height * 2);
        let index = ((y / scale) * low_width + x / scale) * 2;
        [
            self.motion[index] * scale as f32,
            self.motion[index + 1] * scale as f32,
        ]
    }
}

/// Bilinear-sample an interleaved linear RGB image, keeping only the taps
/// whose previous surface matches `current`.
///
/// Ordinary bilinear mixes whatever four texels surround the sample point.
/// Across a silhouette that is two surfaces at once, which is a wrong
/// teacher. Dropping the taps that fail the surface test — and the pixel
/// when none survive — is what makes the reprojection own occlusion
/// instead of inheriting the sample-history mask.
///
/// `position` is in output pixels. A sample that left the image returns
/// `None`, matching [`crate::metrics::temporal_error`]: it is missing, not
/// clamped.
pub fn sample_reprojected(
    image: &[f32],
    previous: &[Surface],
    current: Surface,
    position: [f32; 2],
    width: usize,
    height: usize,
    config: RejectionConfig,
) -> Option<[f32; 3]> {
    assert_eq!(image.len(), width * height * 3);
    assert_eq!(previous.len(), width * height);
    let [x, y] = position;
    if x < 0.0 || y < 0.0 || x > (width - 1) as f32 || y > (height - 1) as f32 {
        return None;
    }
    let x0 = x.floor();
    let y0 = y.floor();
    let tx = x - x0;
    let ty = y - y0;
    let mut color = [0.0; 3];
    let mut weight = 0.0;
    for (dx, wx) in [(0.0, 1.0 - tx), (1.0, tx)] {
        for (dy, wy) in [(0.0, 1.0 - ty), (1.0, ty)] {
            let tap = wx * wy;
            if tap == 0.0 {
                continue;
            }
            let sx = (x0 + dx).clamp(0.0, (width - 1) as f32) as usize;
            let sy = (y0 + dy).clamp(0.0, (height - 1) as f32) as usize;
            let index = sy * width + sx;
            if !current.matches(previous[index], config) {
                continue;
            }
            let base = index * 3;
            for channel in 0..3 {
                color[channel] += tap * image[base + channel];
            }
            weight += tap;
        }
    }
    (weight > 0.0).then(|| [color[0] / weight, color[1] / weight, color[2] / weight])
}

impl Default for RejectionConfig {
    fn default() -> Self {
        Self {
            depth_delta: 0.01,
            normal_cosine: 0.9,
            albedo_delta2: 0.04,
        }
    }
}

/// Temporal evidence expected by a checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Config {
    /// Maximum number of independent frames in the accumulated estimate.
    pub frames: u32,
    pub rejection: RejectionConfig,
    /// Versioned auxiliary-channel layout used by the checkpoint.
    #[serde(default)]
    pub features: Features,
    /// Second history tap: the motion-reprojected previous estimate, with
    /// no surface gate. The first tap is still the rejected accumulation;
    /// this one is what the gate threw away. Missing in older sidecars.
    #[serde(default)]
    pub unrejected_tap: bool,
    /// Mix the previous reconstructed frame, warped to now, after the
    /// spatial gather — one gate per output sub-pixel. That is what a
    /// temporal upscaler reuses: a finished picture, not extra sparse
    /// taps. Missing in older sidecars.
    #[serde(default)]
    pub previous_output: bool,
}

impl Config {
    /// Extra low-resolution channels appended after the stored plane set:
    /// current-frame RGB, normalized sample count, and the exact guided RGB
    /// over which a low-resolution checkpoint predicts its correction.
    pub fn auxiliary_channels(self) -> u32 {
        7 + u32::from(self.features.has_variance())
    }

    /// The same, for a checkpoint that gathers the samples itself.
    ///
    /// The guided RGB is dropped. It exists to tell a residual model what it is
    /// correcting, and a kernel checkpoint corrects nothing — carrying it would
    /// put the 13x13 filter back into a pipeline built to remove it, to describe
    /// a base that is no longer there.
    pub fn gather_auxiliary_channels(self) -> u32 {
        4 + u32::from(self.features.has_variance())
    }

    /// Raw stratified sample history, space-to-depth colour plus sample count.
    pub fn phase_channels(self, scale: u32) -> u32 {
        let values = 4 + 9 * u32::from(self.features.has_phase_lobes());
        u32::from(self.features.has_phase()) * values * scale * scale
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum Features {
    /// Current RGB, confidence, and guided RGB.
    #[default]
    Basic,
    /// Basic inputs plus accepted-history luminance deviation.
    Variance,
    /// Variance inputs plus raw samples accumulated at their exact output
    /// subpixel positions. This preserves the information an LR accumulator
    /// necessarily collapses before a temporal upscaler can see it.
    Phase,
    /// Phase history plus diffuse illumination, specular radiance, and direct
    /// emission retained separately at every output subpixel.
    PhaseLobes,
}

impl Features {
    pub fn has_variance(self) -> bool {
        self != Self::Basic
    }

    pub fn has_phase(self) -> bool {
        matches!(self, Self::Phase | Self::PhaseLobes)
    }

    pub fn has_phase_lobes(self) -> bool {
        self == Self::PhaseLobes
    }
}

/// A sample prepared for the temporal model.
pub struct PreparedSample {
    /// Current sample with its colour plane replaced by accumulated colour.
    pub sample: Sample,
    /// Original current-frame linear RGB, interleaved.
    pub current_color: Vec<f32>,
    /// Motion-reprojected previous estimate, with no surface gate.
    pub unrejected: Vec<f32>,
    /// Accumulated sample count divided by [`Config::frames`].
    pub confidence: Vec<f32>,
    /// Standard deviation of compressed luminance across accepted history.
    pub deviation: Vec<f32>,
    /// Per input texel, then output subpixel, then RGB; linear radiance.
    pub phase_color: Vec<f32>,
    /// Per input texel and output subpixel, normalized to the expected maximum.
    pub phase_count: Vec<f32>,
    /// Per input texel, output subpixel, and nine split-radiance components.
    /// Empty unless [`Features::PhaseLobes`] is selected.
    pub phase_radiance: Vec<f32>,
}

#[derive(Clone)]
struct History {
    color: Vec<f32>,
    radiance: Vec<f32>,
    count: Vec<f32>,
    luminance: Vec<f32>,
    luminance_square: Vec<f32>,
    unrejected: Vec<f32>,
    phase_color: Vec<f32>,
    phase_count: Vec<f32>,
    phase_radiance: Vec<f32>,
}

const RADIANCE_PLANES: [Plane; 3] = [
    Plane::DiffuseIllumination,
    Plane::SpecularRadiance,
    Plane::EmissiveRadiance,
];

fn read_split_radiance(sample: &Sample, layout: &Layout) -> Option<Vec<f32>> {
    if !RADIANCE_PLANES
        .into_iter()
        .all(|plane| layout.lr_planes.contains(plane))
    {
        return None;
    }
    let texels = layout.lr_texels();
    let mut out = vec![0.0; texels * 9];
    for (plane_index, plane) in RADIANCE_PLANES.into_iter().enumerate() {
        for component in 0..3 {
            for index in 0..texels {
                out[index * 9 + plane_index * 3 + component] =
                    self::plane(sample, layout, plane, component, index);
            }
        }
    }
    Some(out)
}

fn phase_slot(sample: &Sample, layout: &Layout) -> Option<usize> {
    let base = layout.lr_planes.channel_offset(Plane::Jitter)?;
    let texels = layout.lr_texels();
    let scale = layout.scale as f32;
    let x = ((sample.lr[base * texels].to_f32() + 0.5) * scale - 0.5).round();
    let y = ((sample.lr[(base + 1) * texels].to_f32() + 0.5) * scale - 0.5).round();
    let scale_i = layout.scale as i32;
    let (x, y) = (x as i32, y as i32);
    (x >= 0 && y >= 0 && x < scale_i && y < scale_i).then_some((y * scale_i + x) as usize)
}

fn compressed_luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * crate::transform::compress(rgb[0])
        + 0.7152 * crate::transform::compress(rgb[1])
        + 0.0722 * crate::transform::compress(rgb[2])
}

fn plane(sample: &Sample, layout: &Layout, plane: Plane, channel: usize, index: usize) -> f32 {
    sample.lr_channel(layout, plane, channel).unwrap()[index].to_f32()
}

fn hr_plane(sample: &Sample, layout: &Layout, plane: Plane, channel: usize, index: usize) -> f32 {
    sample.hr_channel(layout, plane, channel).unwrap()[index].to_f32()
}

fn bilinear<const N: usize>(
    values: &[[f32; N]],
    width: usize,
    height: usize,
    position: [f32; 2],
) -> [f32; N] {
    let x0 = position[0].floor() as i32;
    let y0 = position[1].floor() as i32;
    let fraction = [position[0] - x0 as f32, position[1] - y0 as f32];
    let mut out = [0.0; N];
    for (dy, wy) in [(0, 1.0 - fraction[1]), (1, fraction[1])] {
        for (dx, wx) in [(0, 1.0 - fraction[0]), (1, fraction[0])] {
            let x = (x0 + dx).clamp(0, width as i32 - 1) as usize;
            let y = (y0 + dy).clamp(0, height as i32 - 1) as usize;
            let weight = wx * wy;
            for (component, value) in out.iter_mut().enumerate() {
                *value += weight * values[y * width + x][component];
            }
        }
    }
    out
}

fn surface_at(sample: &Sample, layout: &Layout, index: usize) -> Surface {
    Surface {
        depth: plane(sample, layout, Plane::Depth, 0, index),
        normal: [
            plane(sample, layout, Plane::Normal, 0, index),
            plane(sample, layout, Plane::Normal, 1, index),
            plane(sample, layout, Plane::Normal, 2, index),
        ],
        albedo: [
            plane(sample, layout, Plane::DiffuseAlbedo, 0, index),
            plane(sample, layout, Plane::DiffuseAlbedo, 1, index),
            plane(sample, layout, Plane::DiffuseAlbedo, 2, index),
        ],
    }
}

fn hr_surface_at(sample: &Sample, layout: &Layout, index: usize) -> Surface {
    Surface {
        depth: hr_plane(sample, layout, Plane::Depth, 0, index),
        normal: [
            hr_plane(sample, layout, Plane::Normal, 0, index),
            hr_plane(sample, layout, Plane::Normal, 1, index),
            hr_plane(sample, layout, Plane::Normal, 2, index),
        ],
        albedo: [
            hr_plane(sample, layout, Plane::DiffuseAlbedo, 0, index),
            hr_plane(sample, layout, Plane::DiffuseAlbedo, 1, index),
            hr_plane(sample, layout, Plane::DiffuseAlbedo, 2, index),
        ],
    }
}

/// Warp phase samples as one sparse output-resolution image.
///
/// A phase slot names an output sub-pixel, not a value permanently attached to
/// its low-resolution texel. Moving it with exact output motion allows history
/// to cross both low texels and phase slots without mixing foreground and
/// background vectors. Older static-jitter datasets have no HR motion and keep
/// their screen-aligned phase storage unchanged.
fn reproject_phases(
    history: &History,
    current: &Sample,
    previous: &Sample,
    layout: &Layout,
    config: Config,
) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if !config.features.has_phase() || !layout.hr_planes.contains(Plane::Motion) {
        return None;
    }
    let scale = layout.scale as usize;
    let low_width = layout.lr_width as usize;
    let width = layout.hr_width() as usize;
    let height = layout.hr_height() as usize;
    let slots = scale * scale;
    let storage_index = |x: usize, y: usize| {
        let low = (y / scale) * low_width + x / scale;
        let slot = (y % scale) * scale + x % scale;
        low * slots + slot
    };
    let mut color = vec![0.0; history.phase_color.len()];
    let mut count = vec![0.0; history.phase_count.len()];
    let mut radiance = vec![0.0; history.phase_radiance.len()];

    for y in 0..height {
        for x in 0..width {
            let output = y * width + x;
            let position = [
                x as f32 + hr_plane(current, layout, Plane::Motion, 0, output),
                y as f32 + hr_plane(current, layout, Plane::Motion, 1, output),
            ];
            if position[0] < 0.0
                || position[1] < 0.0
                || position[0] > (width - 1) as f32
                || position[1] > (height - 1) as f32
            {
                continue;
            }

            let x0 = position[0].floor();
            let y0 = position[1].floor();
            let tx = position[0] - x0;
            let ty = position[1] - y0;
            let surface = hr_surface_at(current, layout, output);
            let mut color_sum = [0.0; 3];
            let mut radiance_sum = [0.0; 9];
            let mut weighted_count = 0.0;
            let mut valid_weight = 0.0;
            for (dy, wy) in [(0.0, 1.0 - ty), (1.0, ty)] {
                for (dx, wx) in [(0.0, 1.0 - tx), (1.0, tx)] {
                    let bilinear_weight = wx * wy;
                    if bilinear_weight == 0.0 {
                        continue;
                    }
                    let sx = (x0 + dx).clamp(0.0, (width - 1) as f32) as usize;
                    let sy = (y0 + dy).clamp(0.0, (height - 1) as f32) as usize;
                    let previous_output = sy * width + sx;
                    if !surface.matches(
                        hr_surface_at(previous, layout, previous_output),
                        config.rejection,
                    ) {
                        continue;
                    }
                    valid_weight += bilinear_weight;
                    let source = storage_index(sx, sy);
                    let weight = bilinear_weight * history.phase_count[source];
                    weighted_count += weight;
                    for (channel, sum) in color_sum.iter_mut().enumerate() {
                        *sum += weight * history.phase_color[source * 3 + channel];
                    }
                    if !radiance.is_empty() {
                        for (channel, sum) in radiance_sum.iter_mut().enumerate() {
                            *sum += weight * history.phase_radiance[source * 9 + channel];
                        }
                    }
                }
            }
            if weighted_count == 0.0 {
                continue;
            }
            let destination = storage_index(x, y);
            count[destination] =
                (weighted_count / valid_weight).min(config.frames.saturating_sub(1) as f32);
            for (channel, sum) in color_sum.into_iter().enumerate() {
                color[destination * 3 + channel] = sum / weighted_count;
            }
            if !radiance.is_empty() {
                for (channel, sum) in radiance_sum.into_iter().enumerate() {
                    radiance[destination * 9 + channel] = sum / weighted_count;
                }
            }
        }
    }
    Some((color, count, radiance))
}

fn initial(sample: &Sample, layout: &Layout) -> History {
    let texels = layout.lr_texels();
    let mut color = vec![[0.0; 3]; texels];
    for (index, value) in color.iter_mut().enumerate() {
        for (channel, component) in value.iter_mut().enumerate() {
            *component = plane(sample, layout, Plane::Color, channel, index);
        }
    }
    let luminance: Vec<_> = color.iter().copied().map(compressed_luminance).collect();
    let color: Vec<f32> = color.into_iter().flatten().collect();
    let slots = (layout.scale * layout.scale) as usize;
    let mut phase_color = vec![0.0; texels * slots * 3];
    let mut phase_count = vec![0.0; texels * slots];
    let split_radiance = read_split_radiance(sample, layout);
    let mut phase_radiance = split_radiance
        .as_ref()
        .map_or_else(Vec::new, |_| vec![0.0; texels * slots * 9]);
    if let Some(slot) = phase_slot(sample, layout) {
        for index in 0..texels {
            phase_count[index * slots + slot] = 1.0;
            phase_color[(index * slots + slot) * 3..(index * slots + slot + 1) * 3]
                .copy_from_slice(&color[index * 3..index * 3 + 3]);
            if let Some(split) = &split_radiance {
                phase_radiance[(index * slots + slot) * 9..(index * slots + slot + 1) * 9]
                    .copy_from_slice(&split[index * 9..index * 9 + 9]);
            }
        }
    }
    History {
        unrejected: color.clone(),
        color,
        radiance: split_radiance.unwrap_or_default(),
        count: vec![1.0; texels],
        luminance_square: luminance.iter().map(|value| value * value).collect(),
        luminance,
        phase_color,
        phase_count,
        phase_radiance,
    }
}

/// Reproject an accumulated estimate without mixing surfaces or giving a
/// one-sample texel the same influence as a mature one.
///
/// History texels store means, so bilinear interpolation has to weight each
/// mean by its sample count before combining it. Each of the four taps is also
/// validated independently: accepting the nearest tap and then interpolating
/// all four leaks unrelated radiance across silhouettes.
fn reproject_history(
    history: &History,
    previous: &Sample,
    layout: &Layout,
    current: Surface,
    position: [f32; 2],
    config: Config,
) -> Option<([f32; 3], f32, f32, f32)> {
    let width = layout.lr_width as usize;
    let height = layout.lr_height as usize;
    let [x, y] = position;
    if x < 0.0 || y < 0.0 || x > (width - 1) as f32 || y > (height - 1) as f32 {
        return None;
    }

    let x0 = x.floor();
    let y0 = y.floor();
    let tx = x - x0;
    let ty = y - y0;
    let mut color_sum = [0.0; 3];
    let mut luminance_sum = 0.0;
    let mut luminance_square_sum = 0.0;
    let mut weighted_count = 0.0;
    let mut valid_weight = 0.0;
    for (dx, wx) in [(0.0, 1.0 - tx), (1.0, tx)] {
        for (dy, wy) in [(0.0, 1.0 - ty), (1.0, ty)] {
            let bilinear_weight = wx * wy;
            if bilinear_weight == 0.0 {
                continue;
            }
            let sx = (x0 + dx).clamp(0.0, (width - 1) as f32) as usize;
            let sy = (y0 + dy).clamp(0.0, (height - 1) as f32) as usize;
            let index = sy * width + sx;
            if !current.matches(surface_at(previous, layout, index), config.rejection) {
                continue;
            }
            let weight = bilinear_weight * history.count[index];
            for (channel, value) in color_sum.iter_mut().enumerate() {
                *value += weight * history.color[index * 3 + channel];
            }
            luminance_sum += weight * history.luminance[index];
            luminance_square_sum += weight * history.luminance_square[index];
            weighted_count += weight;
            valid_weight += bilinear_weight;
        }
    }
    (weighted_count > 0.0).then(|| {
        let inverse = weighted_count.recip();
        (
            color_sum.map(|value| value * inverse),
            (weighted_count / valid_weight).min(config.frames.saturating_sub(1) as f32),
            luminance_sum * inverse,
            luminance_square_sum * inverse,
        )
    })
}

fn reproject_radiance(
    history: &History,
    previous: &Sample,
    layout: &Layout,
    current: Surface,
    position: [f32; 2],
    config: Config,
) -> Option<Vec<f32>> {
    if history.radiance.is_empty() {
        return None;
    }
    let width = layout.lr_width as usize;
    let height = layout.lr_height as usize;
    let [x, y] = position;
    if x < 0.0 || y < 0.0 || x > (width - 1) as f32 || y > (height - 1) as f32 {
        return None;
    }
    let x0 = x.floor();
    let y0 = y.floor();
    let tx = x - x0;
    let ty = y - y0;
    let mut sum = vec![0.0; 9];
    let mut weighted_count = 0.0;
    for (dx, wx) in [(0.0, 1.0 - tx), (1.0, tx)] {
        for (dy, wy) in [(0.0, 1.0 - ty), (1.0, ty)] {
            let bilinear_weight = wx * wy;
            if bilinear_weight == 0.0 {
                continue;
            }
            let sx = (x0 + dx).clamp(0.0, (width - 1) as f32) as usize;
            let sy = (y0 + dy).clamp(0.0, (height - 1) as f32) as usize;
            let index = sy * width + sx;
            if !current.matches(surface_at(previous, layout, index), config.rejection) {
                continue;
            }
            let weight = bilinear_weight * history.count[index];
            for (channel, value) in sum.iter_mut().enumerate() {
                *value += weight * history.radiance[index * 9 + channel];
            }
            weighted_count += weight;
        }
    }
    (weighted_count > 0.0).then(|| {
        for value in &mut sum {
            *value /= weighted_count;
        }
        sum
    })
}

fn accumulate(
    current: &Sample,
    previous: &Sample,
    layout: &Layout,
    history: &History,
    config: Config,
) -> History {
    let width = layout.lr_width as usize;
    let height = layout.lr_height as usize;
    let texels = width * height;
    let history_color: Vec<[f32; 3]> = history
        .color
        .chunks_exact(3)
        .map(|rgb| [rgb[0], rgb[1], rgb[2]])
        .collect();
    let current_radiance = read_split_radiance(current, layout);
    let reprojected_phases = reproject_phases(history, current, previous, layout, config);
    let (phase_color, phase_count, phase_radiance) = reprojected_phases.unwrap_or_else(|| {
        (
            history.phase_color.clone(),
            history.phase_count.clone(),
            history.phase_radiance.clone(),
        )
    });
    let mut next = History {
        color: vec![0.0; texels * 3],
        radiance: vec![0.0; history.radiance.len()],
        count: vec![1.0; texels],
        luminance: vec![0.0; texels],
        luminance_square: vec![0.0; texels],
        unrejected: vec![0.0; texels * 3],
        phase_color,
        phase_count,
        phase_radiance,
    };
    let phase = phase_slot(current, layout);
    let slots = (layout.scale * layout.scale) as usize;
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let motion = [
                plane(current, layout, Plane::Motion, 0, index),
                plane(current, layout, Plane::Motion, 1, index),
            ];
            let position = [x as f32 + motion[0], y as f32 + motion[1]];
            let inside = position[0] >= 0.0
                && position[1] >= 0.0
                && position[0] <= (width - 1) as f32
                && position[1] <= (height - 1) as f32;
            let surface = surface_at(current, layout, index);
            let prior = reproject_history(history, previous, layout, surface, position, config);
            let prior_radiance =
                reproject_radiance(history, previous, layout, surface, position, config);
            let (prior_color, count, prior_luminance, prior_luminance_square) =
                prior.unwrap_or(([0.0; 3], 0.0, 0.0, 0.0));
            let current_rgb = [
                plane(current, layout, Plane::Color, 0, index),
                plane(current, layout, Plane::Color, 1, index),
                plane(current, layout, Plane::Color, 2, index),
            ];
            if let Some(slot) = phase {
                let phase_index = index * slots + slot;
                let count = next.phase_count[phase_index];
                for (channel, &current_value) in current_rgb.iter().enumerate() {
                    let destination = phase_index * 3 + channel;
                    next.phase_color[destination] =
                        (next.phase_color[destination] * count + current_value) / (count + 1.0);
                }
                if let Some(current_radiance) = &current_radiance {
                    for channel in 0..9 {
                        let destination = phase_index * 9 + channel;
                        next.phase_radiance[destination] = (next.phase_radiance[destination]
                            * count
                            + current_radiance[index * 9 + channel])
                            / (count + 1.0);
                    }
                }
                next.phase_count[phase_index] = count + 1.0;
            }
            // The un-rejected tap is the previous estimate wherever motion
            // lands, even when the surface test says no. Out of the frame
            // there is no previous, so the tap is this frame — a no-op
            // rather than a black ghost.
            let warped = if inside {
                bilinear(&history_color, width, height, position)
            } else {
                current_rgb
            };
            for channel in 0..3 {
                let current_value = current_rgb[channel];
                next.color[index * 3 + channel] =
                    (current_value + count * prior_color[channel]) / (count + 1.0);
                next.unrejected[index * 3 + channel] = warped[channel];
            }
            if let Some(current_radiance) = &current_radiance {
                let lobe_count = if prior_radiance.is_some() { count } else { 0.0 };
                for channel in 0..9 {
                    let current_value = current_radiance[index * 9 + channel];
                    let prior_value = prior_radiance
                        .as_ref()
                        .map_or(0.0, |radiance| radiance[channel]);
                    next.radiance[index * 9 + channel] =
                        (current_value + lobe_count * prior_value) / (lobe_count + 1.0);
                }
            }
            let luminance = compressed_luminance(current_rgb);
            next.luminance[index] = (luminance + count * prior_luminance) / (count + 1.0);
            next.luminance_square[index] =
                (luminance * luminance + count * prior_luminance_square) / (count + 1.0);
            next.count[index] = count + 1.0;
        }
    }
    next
}

/// Prepare record `index`, accumulating only frames from the same sequence.
pub fn prepare(
    reader: &mut Reader,
    index: usize,
    config: Config,
) -> Result<PreparedSample, crate::dataset::Error> {
    assert!(
        config.frames >= 2,
        "temporal accumulation needs at least two frames"
    );
    let layout = *reader.layout();
    let sequence_length = reader.sequence_length();
    assert!(sequence_length > 1, "the dataset has no frame sequences");
    let sequence_start = index / sequence_length * sequence_length;
    let first = index
        .saturating_add(1)
        .saturating_sub(config.frames as usize)
        .max(sequence_start);
    let mut previous = reader.sample(first)?;
    let mut history = initial(&previous, &layout);
    for frame in first + 1..=index {
        let current = reader.sample(frame)?;
        history = accumulate(&current, &previous, &layout, &history, config);
        previous = current;
    }
    Ok(finish(previous, history, &layout, config))
}

/// Prepare both `index` and the frame before it, in one walk.
///
/// A temporal loss needs the network's answer for the previous frame as well as
/// this one, and preparing them separately walks the sequence twice — up to
/// `2 * frames` reads of a 1.4 MB record for every crop of every batch. Walking
/// once and snapshotting costs one extra read instead.
///
/// The window starts one frame earlier than [`prepare`] alone would need, so
/// the earlier snapshot has as much history behind it as the later one and the
/// two are the same kind of estimate.
pub fn prepare_pair(
    reader: &mut Reader,
    index: usize,
    config: Config,
) -> Result<(PreparedSample, PreparedSample), crate::dataset::Error> {
    assert!(
        config.frames >= 2,
        "temporal accumulation needs at least two frames"
    );
    let layout = *reader.layout();
    let sequence_length = reader.sequence_length();
    assert!(sequence_length > 1, "the dataset has no frame sequences");
    let sequence_start = index / sequence_length * sequence_length;
    assert!(
        index > sequence_start,
        "the first frame of a sequence has no predecessor"
    );
    let first = (index + 1)
        .saturating_sub(config.frames as usize + 1)
        .max(sequence_start);

    let mut previous = reader.sample(first)?;
    let mut history = initial(&previous, &layout);
    // When `index` is the second frame of its sequence the walk starts on the
    // frame before it, so that snapshot is taken before the first accumulation
    // rather than inside the loop.
    let mut earlier =
        (first + 1 == index).then(|| finish(previous.clone(), history.clone(), &layout, config));
    for frame in first + 1..=index {
        let current = reader.sample(frame)?;
        history = accumulate(&current, &previous, &layout, &history, config);
        previous = current;
        if frame + 1 == index {
            earlier = Some(finish(previous.clone(), history.clone(), &layout, config));
        }
    }
    let earlier = earlier.expect("the frame before index is inside the walk");
    Ok((earlier, finish(previous, history, &layout, config)))
}

/// Turn a sample and the history accumulated up to it into a prepared frame.
fn finish(mut sample: Sample, history: History, layout: &Layout, config: Config) -> PreparedSample {
    let current_color = initial(&sample, layout).color;
    let texels = layout.lr_texels();
    let base = layout.lr_planes.channel_offset(Plane::Color).unwrap();
    for channel in 0..3 {
        for index in 0..texels {
            sample.lr[(base + channel) * texels + index] =
                f16::from_f32(history.color[index * 3 + channel]);
        }
    }
    if !history.radiance.is_empty() {
        for (plane_index, plane) in RADIANCE_PLANES.into_iter().enumerate() {
            let base = layout.lr_planes.channel_offset(plane).unwrap();
            for component in 0..3 {
                for index in 0..texels {
                    sample.lr[(base + component) * texels + index] =
                        f16::from_f32(history.radiance[index * 9 + plane_index * 3 + component]);
                }
            }
        }
    }
    PreparedSample {
        sample,
        current_color,
        unrejected: history.unrejected,
        confidence: history
            .count
            .iter()
            .map(|&count| count / config.frames as f32)
            .collect(),
        deviation: history
            .luminance_square
            .iter()
            .zip(&history.luminance)
            .map(|(&square, &mean)| (square - mean * mean).max(0.0).sqrt())
            .collect(),
        phase_color: history.phase_color,
        phase_count: history
            .phase_count
            .into_iter()
            .map(|count| {
                count / (config.frames as f32 / (layout.scale * layout.scale) as f32).max(1.0)
            })
            .collect(),
        phase_radiance: history.phase_radiance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{InputSource, PlaneSet, Writer};

    #[test]
    fn reprojection_accepts_low_or_output_resolution_motion() {
        let surfaces = [flat_surface(1.0); 16];
        let low = [1.5, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let low_warp = Reprojection {
            motion: &low,
            current: &surfaces,
            previous: &surfaces,
            rejection: RejectionConfig::default(),
        };
        assert_eq!(low_warp.output_motion(1, 1, [2, 2], 2), [3.0, -0.5]);

        let mut high = [0.0; 32];
        high[10] = 2.25;
        high[11] = -1.75;
        let high_warp = Reprojection {
            motion: &high,
            current: &surfaces,
            previous: &surfaces,
            rejection: RejectionConfig::default(),
        };
        assert_eq!(high_warp.output_motion(1, 1, [2, 2], 2), [2.25, -1.75]);
    }

    #[test]
    fn exact_motion_reprojects_phase_history_across_subpixel_slots() {
        let layout = Layout {
            scale: 2,
            lr_width: 1,
            lr_height: 1,
            lr_source: InputSource::PathTrace,
            lr_planes: PlaneSet::new().with(Plane::Color),
            hr_planes: PlaneSet::new()
                .with(Plane::Color)
                .with(Plane::Depth)
                .with(Plane::Normal)
                .with(Plane::DiffuseAlbedo)
                .with(Plane::Motion),
        };
        let mut current = Sample {
            lr: vec![f16::ZERO; layout.lr_len()],
            hr: vec![f16::ZERO; layout.hr_len()],
        };
        let mut previous = current.clone();
        let hr_texels = layout.hr_texels();
        for sample in [&mut current, &mut previous] {
            let depth = layout.hr_planes.channel_offset(Plane::Depth).unwrap();
            let normal = layout.hr_planes.channel_offset(Plane::Normal).unwrap();
            let albedo = layout
                .hr_planes
                .channel_offset(Plane::DiffuseAlbedo)
                .unwrap();
            sample.hr[depth * hr_texels..(depth + 1) * hr_texels].fill(f16::ONE);
            sample.hr[(normal + 2) * hr_texels..(normal + 3) * hr_texels].fill(f16::ONE);
            sample.hr[albedo * hr_texels..(albedo + 3) * hr_texels].fill(f16::ONE);
        }
        let motion = layout.hr_planes.channel_offset(Plane::Motion).unwrap();
        current.hr[motion * hr_texels..(motion + 1) * hr_texels].fill(f16::ONE);

        let mut phase_color = vec![0.0; 4 * 3];
        let mut phase_radiance = vec![0.0; 4 * 9];
        for phase in 0..4 {
            phase_color[phase * 3] = 10.0 * (phase + 1) as f32;
            phase_radiance[phase * 9] = 100.0 * (phase + 1) as f32;
        }
        let history = History {
            color: Vec::new(),
            radiance: Vec::new(),
            count: Vec::new(),
            luminance: Vec::new(),
            luminance_square: Vec::new(),
            unrejected: Vec::new(),
            phase_color,
            phase_count: vec![1.0; 4],
            phase_radiance,
        };
        let (color, count, radiance) = reproject_phases(
            &history,
            &current,
            &previous,
            &layout,
            Config {
                frames: 4,
                rejection: RejectionConfig::default(),
                features: Features::PhaseLobes,
                unrejected_tap: false,
                previous_output: false,
            },
        )
        .unwrap();

        // +1 output-pixel motion moves the right subpixel of each row into the
        // left subpixel, crossing phase slots rather than leaving it attached
        // to the original low-resolution texel.
        assert_eq!(color[0], 20.0);
        assert_eq!(color[2 * 3], 40.0);
        assert_eq!(radiance[0], 200.0);
        assert_eq!(radiance[2 * 9], 400.0);
        assert_eq!(count, vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn phase_history_keeps_each_output_subpixel_separate() {
        let layout = Layout {
            scale: 2,
            lr_width: 1,
            lr_height: 1,
            lr_source: InputSource::PathTrace,
            lr_planes: PlaneSet::new()
                .with(Plane::Color)
                .with(Plane::DiffuseIllumination)
                .with(Plane::SpecularRadiance)
                .with(Plane::EmissiveRadiance)
                .with(Plane::Depth)
                .with(Plane::Normal)
                .with(Plane::DiffuseAlbedo)
                .with(Plane::Motion)
                .with(Plane::Jitter),
            hr_planes: PlaneSet::new().with(Plane::Color),
        };
        let path = std::env::temp_dir().join("ommatidia-temporal-phase.omd");
        let mut writer = Writer::create_sequence(&path, layout, 4).unwrap();
        for (frame, jitter) in [[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]]
            .into_iter()
            .enumerate()
        {
            let mut sample = Sample {
                lr: vec![f16::ZERO; layout.lr_len()],
                hr: vec![f16::ZERO; layout.hr_len()],
            };
            let color = layout.lr_planes.channel_offset(Plane::Color).unwrap();
            sample.lr[color] = f16::from_f32(frame as f32 + 1.0);
            for (plane_index, plane) in RADIANCE_PLANES.into_iter().enumerate() {
                let base = layout.lr_planes.channel_offset(plane).unwrap();
                for component in 0..3 {
                    sample.lr[base + component] =
                        f16::from_f32((100 * plane_index + 10 * component + frame) as f32);
                }
            }
            let normal = layout.lr_planes.channel_offset(Plane::Normal).unwrap();
            sample.lr[normal + 2] = f16::ONE;
            let jitter_base = layout.lr_planes.channel_offset(Plane::Jitter).unwrap();
            sample.lr[jitter_base] = f16::from_f32(jitter[0]);
            sample.lr[jitter_base + 1] = f16::from_f32(jitter[1]);
            writer.write(&sample).unwrap();
        }
        writer.finish().unwrap();

        let mut reader = Reader::open(&path).unwrap();
        let prepared = prepare(
            &mut reader,
            3,
            Config {
                frames: 4,
                rejection: RejectionConfig::default(),
                features: Features::PhaseLobes,
                unrejected_tap: false,
                previous_output: false,
            },
        )
        .unwrap();
        let red: Vec<_> = (0..4)
            .map(|phase| prepared.phase_color[phase * 3])
            .collect();
        assert_eq!(red, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(prepared.phase_count, vec![1.0; 4]);
        assert_eq!(prepared.phase_radiance.len(), 4 * 9);
        for phase in 0..4 {
            for component in 0..9 {
                let plane = component / 3;
                let channel = component % 3;
                assert_eq!(
                    prepared.phase_radiance[phase * 9 + component],
                    (100 * plane + 10 * channel + phase) as f32
                );
            }
        }
        for (plane_index, plane) in RADIANCE_PLANES.into_iter().enumerate() {
            for component in 0..3 {
                let accumulated = prepared
                    .sample
                    .lr_channel(&layout, plane, component)
                    .unwrap()[0]
                    .to_f32();
                assert_eq!(
                    accumulated,
                    (100 * plane_index + 10 * component) as f32 + 1.5
                );
            }
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn motion_points_from_current_to_previous() {
        let layout = Layout {
            scale: 2,
            lr_width: 3,
            lr_height: 1,
            lr_source: InputSource::PathTrace,
            lr_planes: PlaneSet::new()
                .with(Plane::Color)
                .with(Plane::Depth)
                .with(Plane::Normal)
                .with(Plane::DiffuseAlbedo)
                .with(Plane::Motion),
            hr_planes: PlaneSet::new().with(Plane::Color),
        };
        let path = std::env::temp_dir().join("ommatidia-temporal-motion.omd");
        let mut first = Sample {
            lr: vec![f16::ZERO; layout.lr_len()],
            hr: vec![f16::ZERO; layout.hr_len()],
        };
        let color = layout.lr_planes.channel_offset(Plane::Color).unwrap();
        for (component, value) in [2.0, 4.0, 6.0].into_iter().enumerate() {
            first.lr[(color + component) * layout.lr_texels() + 1] = f16::from_f32(value);
        }
        let mut current = first.clone();
        current.lr.fill(f16::ZERO);
        let normal = layout.lr_planes.channel_offset(Plane::Normal).unwrap();
        for sample in [&mut first, &mut current] {
            for index in 0..layout.lr_texels() {
                sample.lr[(normal + 2) * layout.lr_texels() + index] = f16::ONE;
            }
        }
        let motion = layout.lr_planes.channel_offset(Plane::Motion).unwrap();
        current.lr[motion * layout.lr_texels()] = f16::ONE;
        let mut writer = Writer::create_sequence(&path, layout, 2).unwrap();
        writer.write(&first).unwrap();
        writer.write(&current).unwrap();
        writer.finish().unwrap();

        let mut reader = Reader::open(&path).unwrap();
        let prepared = prepare(
            &mut reader,
            1,
            Config {
                frames: 2,
                rejection: RejectionConfig::default(),
                features: Features::Basic,
                unrejected_tap: false,
                previous_output: false,
            },
        )
        .unwrap();
        let red = prepared
            .sample
            .lr_channel(&layout, Plane::Color, 0)
            .unwrap();
        assert_eq!(red[0].to_f32(), 1.0);
        assert_eq!(prepared.confidence[0], 1.0);
        assert!(prepared.deviation[0] > 0.0);
        std::fs::remove_file(path).unwrap();
    }

    /// The surface gate zeros the accumulation; the un-rejected tap still
    /// carries the previous colour, which is the whole reason it exists.
    #[test]
    fn the_unrejected_tap_survives_the_surface_gate() {
        let layout = Layout {
            scale: 2,
            lr_width: 2,
            lr_height: 1,
            lr_source: InputSource::PathTrace,
            lr_planes: PlaneSet::new()
                .with(Plane::Color)
                .with(Plane::Depth)
                .with(Plane::Normal)
                .with(Plane::DiffuseAlbedo)
                .with(Plane::Motion),
            hr_planes: PlaneSet::new().with(Plane::Color),
        };
        let path = std::env::temp_dir().join("ommatidia-temporal-unrejected.omd");
        let texels = layout.lr_texels();
        let mut first = Sample {
            lr: vec![f16::ZERO; layout.lr_len()],
            hr: vec![f16::ZERO; layout.hr_len()],
        };
        let color = layout.lr_planes.channel_offset(Plane::Color).unwrap();
        // Previous colour lives in pixel 1; current pixel 0 looks there.
        first.lr[color * texels + 1] = f16::from_f32(4.0);
        first.lr[(color + 1) * texels + 1] = f16::from_f32(6.0);
        first.lr[(color + 2) * texels + 1] = f16::from_f32(8.0);
        let mut current = first.clone();
        current.lr.fill(f16::ZERO);
        let normal = layout.lr_planes.channel_offset(Plane::Normal).unwrap();
        // Opposite normals, so the gate rejects.
        first.lr[(normal + 2) * texels + 1] = f16::ONE;
        current.lr[(normal + 2) * texels] = f16::from_f32(-1.0);
        let motion = layout.lr_planes.channel_offset(Plane::Motion).unwrap();
        current.lr[motion * texels] = f16::ONE;
        let mut writer = Writer::create_sequence(&path, layout, 2).unwrap();
        writer.write(&first).unwrap();
        writer.write(&current).unwrap();
        writer.finish().unwrap();

        let mut reader = Reader::open(&path).unwrap();
        let prepared = prepare(
            &mut reader,
            1,
            Config {
                frames: 2,
                rejection: RejectionConfig::default(),
                features: Features::Basic,
                unrejected_tap: true,
                previous_output: false,
            },
        )
        .unwrap();
        let red = prepared
            .sample
            .lr_channel(&layout, Plane::Color, 0)
            .unwrap();
        assert_eq!(
            red[0].to_f32(),
            0.0,
            "the rejected accumulation has to drop the previous colour"
        );
        assert!(
            prepared.confidence[0] * 2.0 < 1.001,
            "the gate has to have rejected, or this is not the case under test"
        );
        assert!(
            (prepared.unrejected[0] - 4.0).abs() < 1e-3,
            "the un-rejected tap should still be the previous red, got {}",
            prepared.unrejected[0]
        );
        std::fs::remove_file(path).unwrap();
    }

    fn flat_surface(depth: f32) -> Surface {
        Surface {
            depth,
            normal: [0.0, 0.0, 1.0],
            albedo: [0.5, 0.5, 0.5],
        }
    }

    #[test]
    fn matching_surfaces_reproduce_bilinear() {
        let image = [1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 5.0, 0.0, 0.0, 7.0, 0.0, 0.0];
        let surfaces = [flat_surface(1.0); 4];
        let sampled = sample_reprojected(
            &image,
            &surfaces,
            flat_surface(1.0),
            [0.25, 0.5],
            2,
            2,
            RejectionConfig::default(),
        )
        .unwrap();
        // (1-0.25)*(1-0.5)*1 + 0.25*(1-0.5)*3 + (1-0.25)*0.5*5 + 0.25*0.5*7
        assert!((sampled[0] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn a_silhouette_tap_is_dropped_not_mixed() {
        let image = [
            1.0, 0.0, 0.0, 100.0, 0.0, 0.0, 1.0, 0.0, 0.0, 100.0, 0.0, 0.0,
        ];
        let mut surfaces = [flat_surface(1.0); 4];
        surfaces[1] = flat_surface(10.0);
        surfaces[3] = flat_surface(10.0);
        let sampled = sample_reprojected(
            &image,
            &surfaces,
            flat_surface(1.0),
            [0.5, 0.5],
            2,
            2,
            RejectionConfig::default(),
        )
        .unwrap();
        // Only the depth-1 column survives; mixing in 100 would be the bug.
        assert!((sampled[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn accumulated_reprojection_rejects_each_bilinear_tap() {
        let layout = Layout {
            scale: 2,
            lr_width: 2,
            lr_height: 1,
            lr_source: InputSource::PathTrace,
            lr_planes: PlaneSet::new()
                .with(Plane::Color)
                .with(Plane::Depth)
                .with(Plane::Normal)
                .with(Plane::DiffuseAlbedo),
            hr_planes: PlaneSet::new().with(Plane::Color),
        };
        let mut previous = Sample {
            lr: vec![f16::ZERO; layout.lr_len()],
            hr: vec![f16::ZERO; layout.hr_len()],
        };
        let texels = layout.lr_texels();
        let depth = layout.lr_planes.channel_offset(Plane::Depth).unwrap();
        previous.lr[depth * texels] = f16::from_f32(1.0);
        previous.lr[depth * texels + 1] = f16::from_f32(10.0);
        let normal = layout.lr_planes.channel_offset(Plane::Normal).unwrap();
        let albedo = layout
            .lr_planes
            .channel_offset(Plane::DiffuseAlbedo)
            .unwrap();
        for index in 0..texels {
            previous.lr[(normal + 2) * texels + index] = f16::ONE;
            for channel in 0..3 {
                previous.lr[(albedo + channel) * texels + index] = f16::from_f32(0.5);
            }
        }
        let mut history = History {
            color: vec![2.0, 0.0, 0.0, 100.0, 0.0, 0.0],
            radiance: Vec::new(),
            count: vec![4.0, 4.0],
            luminance: vec![0.2, 0.9],
            luminance_square: vec![0.04, 0.81],
            unrejected: Vec::new(),
            phase_color: Vec::new(),
            phase_count: Vec::new(),
            phase_radiance: Vec::new(),
        };
        let config = Config {
            frames: 8,
            rejection: RejectionConfig::default(),
            features: Features::Variance,
            unrejected_tap: false,
            previous_output: true,
        };
        let (color, count, luminance, square) = reproject_history(
            &history,
            &previous,
            &layout,
            flat_surface(1.0),
            [0.5, 0.0],
            config,
        )
        .unwrap();
        assert_eq!(color, [2.0, 0.0, 0.0]);
        assert_eq!(count, 4.0);
        assert!((luminance - 0.2).abs() < 1e-6);
        assert!((square - 0.04).abs() < 1e-6);

        // With both surfaces valid, the mature texel contributes four times
        // as much as the newly reset one. Interpolating the two means first
        // would incorrectly return 51 instead of 21.6.
        previous.lr[depth * texels + 1] = f16::from_f32(1.0);
        history.count[1] = 1.0;
        let (color, count, _, _) = reproject_history(
            &history,
            &previous,
            &layout,
            flat_surface(1.0),
            [0.5, 0.0],
            config,
        )
        .unwrap();
        assert!((color[0] - 21.6).abs() < 1e-5);
        assert!((count - 2.5).abs() < 1e-6);
    }

    /// The reason the teacher owns occlusion: on real sequences the
    /// high-resolution surface test and the sample-history mask disagree.
    /// Ignored so `cargo test` does not open a gigabyte file; run with
    /// `--ignored` when the validation set is present.
    #[test]
    #[ignore = "reads data/rich-temporal-validation-32.omd"]
    fn teacher_and_history_masks_disagree_on_real_sequences() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../data/rich-temporal-validation-32.omd");
        assert!(
            path.exists(),
            "generate or copy the rich temporal validation set to {}",
            path.display()
        );
        let mut reader = Reader::open(&path).unwrap();
        let layout = *reader.layout();
        let config = Config {
            frames: 4,
            rejection: RejectionConfig::default(),
            features: Features::Variance,
            unrejected_tap: false,
            previous_output: false,
        };
        let scale = layout.scale as usize;
        let width = layout.lr_width as usize;
        let height = layout.lr_height as usize;
        let extent_x = width * scale;
        let extent_y = height * scale;
        let sequence = reader.sequence_length();
        let mut history_kept = 0usize;
        let mut teacher_kept = 0usize;
        let mut both = 0usize;
        let mut only_teacher = 0usize;
        let mut only_history = 0usize;
        let mut pixels = 0usize;
        let mut moving_teacher = 0usize;
        let mut moving_history = 0usize;
        let mut moving = 0usize;
        // Eight sequences, last frame of each, is enough to see the masks
        // are not the same thing.
        for sequence_index in 0..8.min(reader.len() / sequence) {
            let index = sequence_index * sequence + sequence - 1;
            let prepared = prepare(&mut reader, index, config).unwrap();
            let previous = reader.sample(index - 1).unwrap();
            let current = prepared.sample;
            let now = crate::batch::crop_hr_surfaces(
                &current,
                &layout,
                crate::batch::Crop {
                    x: 0,
                    y: 0,
                    tile: layout.lr_width,
                },
            );
            let then = crate::batch::crop_hr_surfaces(
                &previous,
                &layout,
                crate::batch::Crop {
                    x: 0,
                    y: 0,
                    tile: layout.lr_width,
                },
            );
            let dummy = vec![0.0; extent_x * extent_y * 3];
            for y in 0..height {
                for x in 0..width {
                    let lr = y * width + x;
                    let history = prepared.confidence[lr] * config.frames as f32 > 1.001;
                    let mx = plane(&current, &layout, crate::dataset::Plane::Motion, 0, lr);
                    let my = plane(&current, &layout, crate::dataset::Plane::Motion, 1, lr);
                    let moved = mx != 0.0 || my != 0.0;
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let px = x * scale + dx;
                            let py = y * scale + dy;
                            let teacher = sample_reprojected(
                                &dummy,
                                &then,
                                now[py * extent_x + px],
                                [px as f32 + mx * scale as f32, py as f32 + my * scale as f32],
                                extent_x,
                                extent_y,
                                config.rejection,
                            )
                            .is_some();
                            pixels += 1;
                            history_kept += usize::from(history);
                            teacher_kept += usize::from(teacher);
                            both += usize::from(history && teacher);
                            only_teacher += usize::from(teacher && !history);
                            only_history += usize::from(history && !teacher);
                            if moved {
                                moving += 1;
                                moving_teacher += usize::from(teacher);
                                moving_history += usize::from(history);
                            }
                        }
                    }
                }
            }
        }
        eprintln!(
            "teacher {teacher_kept}/{pixels} ({:.1}%), history {history_kept}/{pixels} ({:.1}%), \
             only teacher {only_teacher}, only history {only_history}, both {both}; \
             moving teacher {moving_teacher}/{moving}, history {moving_history}/{moving}",
            100.0 * teacher_kept as f64 / pixels as f64,
            100.0 * history_kept as f64 / pixels as f64,
        );
        assert!(
            only_teacher + only_history > 0,
            "the two masks agreed on every pixel, so the teacher is still inheriting"
        );
    }

    #[test]
    fn a_fully_occluded_sample_is_missing() {
        let image = [1.0; 12];
        let previous = [flat_surface(10.0); 4];
        assert_eq!(
            sample_reprojected(
                &image,
                &previous,
                flat_surface(1.0),
                [0.5, 0.5],
                2,
                2,
                RejectionConfig::default(),
            ),
            None
        );
        assert_eq!(
            sample_reprojected(
                &image,
                &previous,
                flat_surface(1.0),
                [-0.1, 0.0],
                2,
                2,
                RejectionConfig::default(),
            ),
            None
        );
    }
}
