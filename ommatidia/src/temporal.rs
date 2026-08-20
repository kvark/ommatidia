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
    /// Current-to-previous motion at input resolution, in input pixels.
    pub motion: &'a [f32],
    /// High-resolution surfaces of this frame, output-pixel major.
    pub current: &'a [Surface],
    /// High-resolution surfaces of the previous frame, output-pixel major.
    pub previous: &'a [Surface],
    pub rejection: RejectionConfig,
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
        7 + u32::from(self.features == Features::Variance)
    }

    /// The same, for a checkpoint that gathers the samples itself.
    ///
    /// The guided RGB is dropped. It exists to tell a residual model what it is
    /// correcting, and a kernel checkpoint corrects nothing — carrying it would
    /// put the 13x13 filter back into a pipeline built to remove it, to describe
    /// a base that is no longer there.
    pub fn gather_auxiliary_channels(self) -> u32 {
        4 + u32::from(self.features == Features::Variance)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum Features {
    /// Current RGB, confidence, and guided RGB.
    #[default]
    Basic,
    /// Basic inputs plus accepted-history luminance deviation.
    Variance,
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
}

#[derive(Clone)]
struct History {
    color: Vec<f32>,
    count: Vec<f32>,
    luminance: Vec<f32>,
    luminance_square: Vec<f32>,
    unrejected: Vec<f32>,
}

fn compressed_luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * crate::transform::compress(rgb[0])
        + 0.7152 * crate::transform::compress(rgb[1])
        + 0.0722 * crate::transform::compress(rgb[2])
}

fn plane(sample: &Sample, layout: &Layout, plane: Plane, channel: usize, index: usize) -> f32 {
    sample.lr_channel(layout, plane, channel).unwrap()[index].to_f32()
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
    History {
        unrejected: color.clone(),
        color,
        count: vec![1.0; texels],
        luminance_square: luminance.iter().map(|value| value * value).collect(),
        luminance,
    }
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
    let history_count: Vec<[f32; 1]> = history.count.iter().map(|&v| [v]).collect();
    let history_luminance: Vec<[f32; 1]> = history.luminance.iter().map(|&v| [v]).collect();
    let history_luminance_square: Vec<[f32; 1]> =
        history.luminance_square.iter().map(|&v| [v]).collect();
    let mut next = History {
        color: vec![0.0; texels * 3],
        count: vec![1.0; texels],
        luminance: vec![0.0; texels],
        luminance_square: vec![0.0; texels],
        unrejected: vec![0.0; texels * 3],
    };
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
            let previous_x = position[0].round().clamp(0.0, (width - 1) as f32) as usize;
            let previous_y = position[1].round().clamp(0.0, (height - 1) as f32) as usize;
            let valid = inside
                && surface_at(current, layout, index).matches(
                    surface_at(previous, layout, previous_y * width + previous_x),
                    config.rejection,
                );
            let count = if valid {
                bilinear(&history_count, width, height, position)[0]
                    .min(config.frames.saturating_sub(1) as f32)
            } else {
                0.0
            };
            let prior = valid.then(|| bilinear(&history_color, width, height, position));
            let (prior_luminance, prior_luminance_square) = if valid {
                (
                    bilinear(&history_luminance, width, height, position)[0],
                    bilinear(&history_luminance_square, width, height, position)[0],
                )
            } else {
                (0.0, 0.0)
            };
            let current_rgb = [
                plane(current, layout, Plane::Color, 0, index),
                plane(current, layout, Plane::Color, 1, index),
                plane(current, layout, Plane::Color, 2, index),
            ];
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
                let prior_value = prior.as_ref().map_or(0.0, |value| value[channel]);
                next.color[index * 3 + channel] =
                    (current_value + count * prior_value) / (count + 1.0);
                next.unrejected[index * 3 + channel] = warped[channel];
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{InputSource, PlaneSet, Writer};

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
