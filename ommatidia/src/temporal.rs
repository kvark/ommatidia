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
}

impl Config {
    /// Extra low-resolution channels appended after the stored plane set:
    /// current-frame RGB, normalized sample count, and the exact guided RGB
    /// over which a low-resolution checkpoint predicts its correction.
    pub fn auxiliary_channels(self) -> u32 {
        7 + u32::from(self.features == Features::Variance)
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
    /// Accumulated sample count divided by [`Config::frames`].
    pub confidence: Vec<f32>,
    /// Standard deviation of compressed luminance across accepted history.
    pub deviation: Vec<f32>,
}

struct History {
    color: Vec<f32>,
    count: Vec<f32>,
    luminance: Vec<f32>,
    luminance_square: Vec<f32>,
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

fn surfaces_match(
    current: &Sample,
    previous: &Sample,
    layout: &Layout,
    current_index: usize,
    previous_index: usize,
    config: RejectionConfig,
) -> bool {
    let current_depth = plane(current, layout, Plane::Depth, 0, current_index);
    let previous_depth = plane(previous, layout, Plane::Depth, 0, previous_index);
    let sky = |depth: f32| depth >= 60_000.0;
    if sky(current_depth) || sky(previous_depth) {
        return sky(current_depth) == sky(previous_depth);
    }
    let encoded = |depth: f32| 1.0 / (1.0 + depth.max(0.0));
    if (encoded(current_depth) - encoded(previous_depth)).abs() > config.depth_delta {
        return false;
    }

    let mut normal_dot = 0.0;
    let mut current_len2 = 0.0;
    let mut previous_len2 = 0.0;
    let mut albedo_delta2 = 0.0;
    for channel in 0..3 {
        let a = plane(current, layout, Plane::Normal, channel, current_index);
        let b = plane(previous, layout, Plane::Normal, channel, previous_index);
        normal_dot += a * b;
        current_len2 += a * a;
        previous_len2 += b * b;
        let delta = plane(
            current,
            layout,
            Plane::DiffuseAlbedo,
            channel,
            current_index,
        ) - plane(
            previous,
            layout,
            Plane::DiffuseAlbedo,
            channel,
            previous_index,
        );
        albedo_delta2 += delta * delta;
    }
    let cosine = normal_dot / (current_len2 * previous_len2).sqrt().max(1.0e-6);
    cosine > config.normal_cosine && albedo_delta2 < config.albedo_delta2
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
    History {
        color: color.into_iter().flatten().collect(),
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
                && surfaces_match(
                    current,
                    previous,
                    layout,
                    index,
                    previous_y * width + previous_x,
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
            for channel in 0..3 {
                let current_value = current_rgb[channel];
                let prior_value = prior.as_ref().map_or(0.0, |value| value[channel]);
                next.color[index * 3 + channel] =
                    (current_value + count * prior_value) / (count + 1.0);
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

    let current_color = initial(&previous, &layout).color;
    let texels = layout.lr_texels();
    let base = layout.lr_planes.channel_offset(Plane::Color).unwrap();
    for channel in 0..3 {
        for index in 0..texels {
            previous.lr[(base + channel) * texels + index] =
                f16::from_f32(history.color[index * 3 + channel]);
        }
    }
    Ok(PreparedSample {
        sample: previous,
        current_color,
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
    })
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
}
