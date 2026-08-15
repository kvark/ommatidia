//! Image-quality metrics shared by spatial and temporal experiments.

/// Mean squared error between two linear images, in compressed space.
pub fn error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = crate::transform::compress(x) - crate::transform::compress(y);
            d * d
        })
        .sum();
    sum / a.len() as f32
}

/// Structural similarity over 8×8 luminance windows in compressed space.
pub fn ssim(a: &[f32], b: &[f32], width: usize, height: usize) -> f32 {
    assert_eq!(a.len(), width * height * 3);
    assert_eq!(a.len(), b.len());
    const BLOCK: usize = 8;
    const C1: f64 = 0.0001;
    const C2: f64 = 0.0009;
    let blocks_x = width.div_ceil(BLOCK);
    let blocks_y = height.div_ceil(BLOCK);
    let luminance = |rgb: &[f32]| -> f64 {
        let r = crate::transform::compress(rgb[0]) as f64;
        let g = crate::transform::compress(rgb[1]) as f64;
        let b = crate::transform::compress(rgb[2]) as f64;
        0.2126 * r + 0.7152 * g + 0.0722 * b
    };
    let mut total = 0.0;
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let x_end = ((bx + 1) * BLOCK).min(width);
            let y_end = ((by + 1) * BLOCK).min(height);
            let count = ((x_end - bx * BLOCK) * (y_end - by * BLOCK)) as f64;
            let (mut sum_a, mut sum_b) = (0.0, 0.0);
            for y in by * BLOCK..y_end {
                for x in bx * BLOCK..x_end {
                    let offset = (y * width + x) * 3;
                    sum_a += luminance(&a[offset..]);
                    sum_b += luminance(&b[offset..]);
                }
            }
            let (mean_a, mean_b) = (sum_a / count, sum_b / count);
            let (mut var_a, mut var_b, mut covariance) = (0.0, 0.0, 0.0);
            for y in by * BLOCK..y_end {
                for x in bx * BLOCK..x_end {
                    let offset = (y * width + x) * 3;
                    let da = luminance(&a[offset..]) - mean_a;
                    let db = luminance(&b[offset..]) - mean_b;
                    var_a += da * da;
                    var_b += db * db;
                    covariance += da * db;
                }
            }
            var_a /= count;
            var_b /= count;
            covariance /= count;
            total += ((2.0 * mean_a * mean_b + C1) * (2.0 * covariance + C2))
                / ((mean_a * mean_a + mean_b * mean_b + C1) * (var_a + var_b + C2));
        }
    }
    (total / (blocks_x * blocks_y) as f64) as f32
}

/// Sufficient statistics for aggregating temporal error without giving a crop
/// with fewer valid pixels the same weight as a fully valid crop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalError {
    pub squared_sum: f64,
    pub values: usize,
}

impl TemporalError {
    pub fn mean(self) -> f64 {
        self.squared_sum / self.values as f64
    }
}

/// Motion-compensated temporal error in compressed radiance space.
///
/// `motion` is interleaved current-to-previous motion at low resolution, in
/// low-resolution pixels. `valid` rejects disocclusions and surface changes.
/// The error compares the predicted frame-to-frame change with the reference
/// change, rather than rewarding a temporally stable but biased image.
pub fn temporal_error(
    prediction: [&[f32]; 2],
    reference: [&[f32]; 2],
    motion: &[f32],
    valid: &[bool],
    low_extent: [usize; 2],
    scale: usize,
) -> Option<TemporalError> {
    let [current, previous] = prediction;
    let [current_reference, previous_reference] = reference;
    let [low_width, low_height] = low_extent;
    let width = low_width * scale;
    let height = low_height * scale;
    let image_len = width * height * 3;
    assert_eq!(current.len(), image_len);
    assert_eq!(previous.len(), image_len);
    assert_eq!(current_reference.len(), image_len);
    assert_eq!(previous_reference.len(), image_len);
    assert_eq!(motion.len(), low_width * low_height * 2);
    assert_eq!(valid.len(), low_width * low_height);

    let sample = |image: &[f32], x: f32, y: f32, channel: usize| {
        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let tx = x - x0 as f32;
        let ty = y - y0 as f32;
        let at = |dx: isize, dy: isize| {
            let sx = (x0 + dx).clamp(0, width as isize - 1) as usize;
            let sy = (y0 + dy).clamp(0, height as isize - 1) as usize;
            image[(sy * width + sx) * 3 + channel]
        };
        let top = at(0, 0) + tx * (at(1, 0) - at(0, 0));
        let bottom = at(0, 1) + tx * (at(1, 1) - at(0, 1));
        top + ty * (bottom - top)
    };

    let mut sum = 0.0f64;
    let mut count = 0usize;
    for y in 0..height {
        for x in 0..width {
            let low_index = (y / scale) * low_width + x / scale;
            if !valid[low_index] {
                continue;
            }
            let previous_x = x as f32 + motion[low_index * 2] * scale as f32;
            let previous_y = y as f32 + motion[low_index * 2 + 1] * scale as f32;
            if previous_x < 0.0
                || previous_y < 0.0
                || previous_x > (width - 1) as f32
                || previous_y > (height - 1) as f32
            {
                continue;
            }
            let offset = (y * width + x) * 3;
            for channel in 0..3 {
                let predicted_change = crate::transform::compress(current[offset + channel])
                    - crate::transform::compress(sample(previous, previous_x, previous_y, channel));
                let reference_change =
                    crate::transform::compress(current_reference[offset + channel])
                        - crate::transform::compress(sample(
                            previous_reference,
                            previous_x,
                            previous_y,
                            channel,
                        ));
                let delta = predicted_change - reference_change;
                sum += (delta * delta) as f64;
                count += 1;
            }
        }
    }
    (count != 0).then_some(TemporalError {
        squared_sum: sum,
        values: count,
    })
}

#[cfg(test)]
mod temporal_tests {
    use super::temporal_error;

    #[test]
    fn temporal_error_measures_excess_change_not_scene_motion() {
        let previous_reference = vec![1.0; 2 * 2 * 3];
        let current_reference = vec![2.0; 2 * 2 * 3];
        let motion = vec![0.0; 2 * 2 * 2];
        let valid = vec![true; 2 * 2];
        assert_eq!(
            temporal_error(
                [&current_reference, &previous_reference],
                [&current_reference, &previous_reference],
                &motion,
                &valid,
                [2, 2],
                1,
            ),
            Some(super::TemporalError {
                squared_sum: 0.0,
                values: 12,
            })
        );

        let flickering = vec![3.0; 2 * 2 * 3];
        assert!(
            temporal_error(
                [&flickering, &previous_reference],
                [&current_reference, &previous_reference],
                &motion,
                &valid,
                [2, 2],
                1,
            )
            .unwrap()
            .mean()
                > 0.0
        );
    }

    #[test]
    fn temporal_error_excludes_invalid_and_out_of_crop_history() {
        let image = vec![1.0; 2 * 2 * 3];
        let motion = vec![10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0];
        assert_eq!(
            temporal_error(
                [&image, &image],
                [&image, &image],
                &motion,
                &[true; 4],
                [2, 2],
                1,
            ),
            None
        );
        assert_eq!(
            temporal_error(
                [&image, &image],
                [&image, &image],
                &[0.0; 8],
                &[false; 4],
                [2, 2],
                1,
            ),
            None
        );
    }
}
