//! Score simple temporal accumulation before committing to a learned history path.

use half::f16;
use ommatidia::batch::{self, Crop};
use ommatidia::dataset::{Layout, Plane, Reader, Sample};
use ommatidia::model::GuideConfig;

struct History {
    color: Vec<f32>,
    count: Vec<f32>,
}

#[derive(Clone, Copy)]
struct RejectConfig {
    depth_delta: f32,
    normal_cosine: f32,
    albedo_delta2: f32,
}

impl Default for RejectConfig {
    fn default() -> Self {
        Self {
            depth_delta: 0.01,
            normal_cosine: 0.9,
            albedo_delta2: 0.04,
        }
    }
}

fn plane(sample: &Sample, layout: &Layout, plane: Plane, channel: usize, index: usize) -> f32 {
    sample.lr_channel(layout, plane, channel).unwrap()[index].to_f32()
}

fn bilinear(
    values: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    position: [f32; 2],
) -> Vec<f32> {
    let x0 = position[0].floor() as i32;
    let y0 = position[1].floor() as i32;
    let fraction = [position[0] - x0 as f32, position[1] - y0 as f32];
    let mut out = vec![0.0; channels];
    for (dy, wy) in [(0, 1.0 - fraction[1]), (1, fraction[1])] {
        for (dx, wx) in [(0, 1.0 - fraction[0]), (1, fraction[0])] {
            let x = (x0 + dx).clamp(0, width as i32 - 1) as usize;
            let y = (y0 + dy).clamp(0, height as i32 - 1) as usize;
            let weight = wx * wy;
            for channel in 0..channels {
                out[channel] += weight * values[(y * width + x) * channels + channel];
            }
        }
    }
    out
}

fn accumulate(current: &Sample, layout: &Layout, history: &History, frames: usize) -> History {
    let width = layout.lr_width as usize;
    let height = layout.lr_height as usize;
    let texels = width * height;
    let mut next = History {
        color: vec![0.0; texels * 3],
        count: vec![1.0; texels],
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
            let valid = inside;
            let count = if valid {
                bilinear(&history.count, width, height, 1, position)[0]
                    .min(frames.saturating_sub(1) as f32)
            } else {
                0.0
            };
            let prior = valid.then(|| bilinear(&history.color, width, height, 3, position));
            for channel in 0..3 {
                let current_value = plane(current, layout, Plane::Color, channel, index);
                let prior_value = prior.as_ref().map_or(0.0, |value| value[channel]);
                next.color[index * 3 + channel] =
                    (current_value + count * prior_value) / (count + 1.0);
            }
            next.count[index] = count + 1.0;
        }
    }
    next
}

fn with_color(sample: &Sample, layout: &Layout, color: &[f32]) -> Sample {
    let mut out = sample.clone();
    let texels = layout.lr_texels();
    let base = layout.lr_planes.channel_offset(Plane::Color).unwrap();
    for channel in 0..3 {
        for index in 0..texels {
            out.lr[(base + channel) * texels + index] = f16::from_f32(color[index * 3 + channel]);
        }
    }
    out
}

fn initial_history(sample: &Sample, layout: &Layout) -> History {
    History {
        color: batch::crop_color(
            sample,
            layout,
            Crop {
                x: 0,
                y: 0,
                tile: layout.lr_width,
            },
        ),
        count: vec![1.0; layout.lr_texels()],
    }
}

fn downsampled_reference(sample: &Sample, layout: &Layout) -> Vec<f32> {
    let scale = layout.scale as usize;
    let width = layout.lr_width as usize;
    let height = layout.lr_height as usize;
    let hr_width = layout.hr_width() as usize;
    let hr_texels = layout.hr_texels();
    let base = layout.hr_planes.channel_offset(Plane::Color).unwrap();
    let mut out = vec![0.0; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                let source =
                    &sample.hr[(base + channel) * hr_texels..(base + channel + 1) * hr_texels];
                let mut sum = 0.0;
                for dy in 0..scale {
                    for dx in 0..scale {
                        sum += source[(y * scale + dy) * hr_width + x * scale + dx].to_f32();
                    }
                }
                out[(y * width + x) * 3 + channel] = sum / (scale * scale) as f32;
            }
        }
    }
    out
}

#[derive(Default)]
struct Score {
    mse: f64,
    ssim: f64,
    frames: usize,
}

impl Score {
    fn add(&mut self, image: &[f32], reference: &[f32], extent: usize) {
        self.mse += ommatidia::metrics::error(image, reference) as f64;
        self.ssim += ommatidia::metrics::ssim(image, reference, extent, extent) as f64;
        self.frames += 1;
    }

    fn print(&self, name: &str) {
        let mse = self.mse / self.frames as f64;
        println!(
            "{name:<24} MSE {mse:.6}, PSNR {:.2} dB, SSIM {:.4}",
            -10.0 * mse.log10(),
            self.ssim / self.frames as f64,
        );
    }
}

fn main() {
    let mut args = std::env::args_os().skip(1);
    let path = std::path::PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!(
            "usage: temporal-oracle DATASET.omd \
             [DEPTH_DELTA NORMAL_COSINE ALBEDO_DELTA2 [HISTORY_FRAMES]]"
        );
        std::process::exit(2);
    }));
    let mut rejection = RejectConfig::default();
    for (name, target) in [
        ("depth delta", &mut rejection.depth_delta),
        ("normal cosine", &mut rejection.normal_cosine),
        ("albedo delta²", &mut rejection.albedo_delta2),
    ] {
        if let Some(value) = args.next() {
            *target = value
                .to_str()
                .and_then(|text| text.parse().ok())
                .unwrap_or_else(|| panic!("invalid {name}"));
        }
    }
    let mut reader = Reader::open(&path).expect("open sequence dataset");
    let mut temporal_reader = Reader::open(&path).expect("open temporal sequence dataset");
    let layout = *reader.layout();
    let sequence_length = reader.sequence_length();
    assert!(sequence_length > 1, "dataset does not contain sequences");
    let history_frames = args
        .next()
        .map(|value| {
            value
                .to_str()
                .and_then(|text| text.parse::<usize>().ok())
                .filter(|&frames| frames >= 2 && frames <= sequence_length)
                .expect("history frames must be between 2 and the sequence length")
        })
        .unwrap_or(sequence_length);
    assert_eq!(
        layout.lr_width, layout.lr_height,
        "oracle expects square frames"
    );
    for plane in [
        Plane::Color,
        Plane::Depth,
        Plane::Normal,
        Plane::DiffuseAlbedo,
        Plane::Motion,
    ] {
        assert!(layout.lr_planes.contains(plane), "missing LR {plane:?}");
    }
    for plane in [
        Plane::Color,
        Plane::Depth,
        Plane::Normal,
        Plane::DiffuseAlbedo,
    ] {
        assert!(layout.hr_planes.contains(plane), "missing HR {plane:?}");
    }

    let crop = Crop {
        x: 0,
        y: 0,
        tile: layout.lr_width,
    };
    let extent = layout.hr_width() as usize;
    let mut single = Score::default();
    let mut motion_only = Score::default();
    let mut rejected = Score::default();
    let mut rejected_without_spatial_filter = Score::default();
    let mut canonical_low_oracle = Score::default();
    let mut accepted_pixels = 0usize;
    let mut history_pixels = 0usize;
    for sequence in 0..reader.len() / sequence_length {
        let first = reader.sample(sequence * sequence_length).unwrap();
        let mut raw_history = initial_history(&first, &layout);
        for frame in 1..sequence_length {
            let index = sequence * sequence_length + frame;
            let current = reader.sample(index).unwrap();
            raw_history = accumulate(&current, &layout, &raw_history, history_frames);
            let prepared = ommatidia::temporal::prepare(
                &mut temporal_reader,
                index,
                ommatidia::temporal::Config {
                    frames: history_frames as u32,
                    rejection: ommatidia::temporal::RejectionConfig {
                        depth_delta: rejection.depth_delta,
                        normal_cosine: rejection.normal_cosine,
                        albedo_delta2: rejection.albedo_delta2,
                    },
                    features: ommatidia::temporal::Features::Basic,
                    unrejected_tap: false,
                    previous_output: false,
                },
            )
            .unwrap();
            accepted_pixels += prepared
                .confidence
                .iter()
                .filter(|&&confidence| confidence * history_frames as f32 > 1.001)
                .count();
            history_pixels += prepared.confidence.len();
            let reference = batch::crop_reference(&current, &layout, crop);
            let rejected_color = batch::crop_color(&prepared.sample, &layout, crop);
            rejected_without_spatial_filter.add(
                &batch::high_resolution_guided_from_color(
                    &current,
                    &layout,
                    crop,
                    GuideConfig::TUNED,
                    &rejected_color,
                ),
                &reference,
                extent,
            );
            canonical_low_oracle.add(
                &batch::high_resolution_guided_from_color(
                    &current,
                    &layout,
                    crop,
                    GuideConfig::TUNED,
                    &downsampled_reference(&current, &layout),
                ),
                &reference,
                extent,
            );
            for (score, sample) in [
                (&mut single, current.clone()),
                (
                    &mut motion_only,
                    with_color(&current, &layout, &raw_history.color),
                ),
                (&mut rejected, prepared.sample),
            ] {
                let image =
                    batch::high_resolution_guided_base(&sample, &layout, crop, GuideConfig::TUNED);
                score.add(&image, &reference, extent);
            }
        }
    }
    println!(
        "{} sequences × {} scored frames, up to {} history samples",
        reader.len() / sequence_length,
        sequence_length - 1,
        history_frames,
    );
    single.print("single frame");
    motion_only.print("motion-only history");
    rejected.print("surface-rejected history");
    rejected_without_spatial_filter.print("history, direct HR gather");
    canonical_low_oracle.print("canonical-low oracle");
    println!(
        "surface history accepted at {:.1}% of pixels",
        100.0 * accepted_pixels as f64 / history_pixels as f64,
    );
    println!(
        "rejection: depth {:.4}, normal {:.3}, albedo² {:.4}",
        rejection.depth_delta, rejection.normal_cosine, rejection.albedo_delta2,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ommatidia::dataset::{InputSource, PlaneSet};

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
        let mut sample = Sample {
            lr: vec![f16::ZERO; layout.lr_len()],
            hr: vec![f16::ZERO; layout.hr_len()],
        };
        let motion_x = layout.lr_planes.channel_offset(Plane::Motion).unwrap();
        sample.lr[motion_x * layout.lr_texels()] = f16::ONE;
        let history = History {
            color: vec![0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0, 0.0, 0.0],
            count: vec![1.0; 3],
        };
        let result = accumulate(&sample, &layout, &history, 4);
        assert_eq!(&result.color[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(result.count[0], 2.0);
    }
}
