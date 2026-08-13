//! Score simple temporal accumulation before committing to a learned history path.

use half::f16;
use ommatidia::batch::{self, Crop};
use ommatidia::dataset::{Layout, Plane, Reader, Sample};
use ommatidia::model::GuideConfig;

struct History {
    color: Vec<f32>,
    count: Vec<f32>,
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

fn surfaces_match(
    current: &Sample,
    previous: &Sample,
    layout: &Layout,
    current_index: usize,
    previous_index: usize,
) -> bool {
    let current_depth = plane(current, layout, Plane::Depth, 0, current_index);
    let previous_depth = plane(previous, layout, Plane::Depth, 0, previous_index);
    let sky = |depth: f32| depth >= 60_000.0;
    if sky(current_depth) || sky(previous_depth) {
        return sky(current_depth) == sky(previous_depth);
    }
    let encoded = |depth: f32| 1.0 / (1.0 + depth.max(0.0));
    if (encoded(current_depth) - encoded(previous_depth)).abs() > 0.01 {
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
    cosine > 0.9 && albedo_delta2 < 0.04
}

fn accumulate(
    current: &Sample,
    previous: &Sample,
    layout: &Layout,
    history: &History,
    reject: bool,
) -> History {
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
            let previous_x = position[0].round().clamp(0.0, (width - 1) as f32) as usize;
            let previous_y = position[1].round().clamp(0.0, (height - 1) as f32) as usize;
            let valid = inside
                && (!reject
                    || surfaces_match(
                        current,
                        previous,
                        layout,
                        index,
                        previous_y * width + previous_x,
                    ));
            let count = if valid {
                bilinear(&history.count, width, height, 1, position)[0].min(3.0)
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
    let path = std::env::args_os().nth(1).unwrap_or_else(|| {
        eprintln!("usage: temporal-oracle DATASET.omd");
        std::process::exit(2);
    });
    let mut reader = Reader::open(path).expect("open sequence dataset");
    let layout = *reader.layout();
    let sequence_length = reader.sequence_length();
    assert!(sequence_length > 1, "dataset does not contain sequences");
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
    let mut accepted_pixels = 0usize;
    let mut history_pixels = 0usize;
    for sequence in 0..reader.len() / sequence_length {
        let mut previous = reader.sample(sequence * sequence_length).unwrap();
        let mut raw_history = initial_history(&previous, &layout);
        let mut valid_history = initial_history(&previous, &layout);
        for frame in 1..sequence_length {
            let current = reader.sample(sequence * sequence_length + frame).unwrap();
            raw_history = accumulate(&current, &previous, &layout, &raw_history, false);
            valid_history = accumulate(&current, &previous, &layout, &valid_history, true);
            accepted_pixels += valid_history
                .count
                .iter()
                .filter(|&&count| count > 1.0)
                .count();
            history_pixels += valid_history.count.len();
            let reference = batch::crop_reference(&current, &layout, crop);
            for (score, sample) in [
                (&mut single, current.clone()),
                (
                    &mut motion_only,
                    with_color(&current, &layout, &raw_history.color),
                ),
                (
                    &mut rejected,
                    with_color(&current, &layout, &valid_history.color),
                ),
            ] {
                let image =
                    batch::high_resolution_guided_base(&sample, &layout, crop, GuideConfig::TUNED);
                score.add(&image, &reference, extent);
            }
            previous = current;
        }
    }
    println!(
        "{} sequences × {} scored history frames",
        reader.len() / sequence_length,
        sequence_length - 1
    );
    single.print("single frame");
    motion_only.print("motion-only history");
    rejected.print("surface-rejected history");
    println!(
        "surface history accepted at {:.1}% of pixels",
        100.0 * accepted_pixels as f64 / history_pixels as f64,
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
        let result = accumulate(&sample, &sample, &layout, &history, false);
        assert_eq!(&result.color[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(result.count[0], 2.0);
    }
}
