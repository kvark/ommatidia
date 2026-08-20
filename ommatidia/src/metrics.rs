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

/// Relative mean squared error, after Rousselle.
///
/// [`error`] is an absolute difference in a compressed space, so a scene's
/// bright regions decide its value. Dividing by the reference lets a dark
/// region be wrong by a visible fraction of itself and have that count.
pub fn relative_error(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = (x - y) as f64;
            d * d / (y as f64 * y as f64 + 0.01)
        })
        .sum();
    sum / a.len() as f64
}

/// Mean gradient magnitude of displayed luminance: how much detail an image
/// carries, in the space it is looked at.
///
/// The metric [`error`] does not contain. Regression toward a blurred image is
/// what least-squares reconstruction does when the input is noisy, and it costs
/// almost nothing in PSNR — a box blur that removes a further eighth of the
/// reference's gradient energy moves [`error`] by 0.06 dB and improves the same
/// measure taken in display space. Comparing this against the reference's own
/// value says whether a gain came from reconstructing the frame or from giving
/// up on it.
///
/// A gradient cannot tell detail from noise, so this only means what it sounds
/// like between images that have already been denoised. Above the reference's
/// own value it is reporting the opposite: an unfiltered 4-spp input scores
/// several hundred percent of the canonical frame, all of it sample noise.
/// Read it as "kept 63%" for a reconstruction and as "still noisy" beyond 100%.
pub fn detail(image: &[f32], width: usize, height: usize) -> f64 {
    assert_eq!(image.len(), width * height * 3);
    assert!(width > 1 && height > 1, "a gradient needs two pixels");
    let at = |x: usize, y: usize| {
        let rgb = &image[(y * width + x) * 3..];
        (0.2126 * crate::transform::display(rgb[0])
            + 0.7152 * crate::transform::display(rgb[1])
            + 0.0722 * crate::transform::display(rgb[2])) as f64
    };
    let mut sum = 0.0;
    for y in 0..height - 1 {
        for x in 0..width - 1 {
            let here = at(x, y);
            sum += ((at(x + 1, y) - here).powi(2) + (at(x, y + 1) - here).powi(2)).sqrt();
        }
    }
    sum / ((width - 1) * (height - 1)) as f64
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
/// low-resolution pixels. Occlusion is the teacher's job: each output pixel
/// is reprojected with [`crate::temporal::sample_reprojected`] against the
/// high-resolution surfaces, so a bilinear mix across a silhouette is
/// dropped rather than inherited from the sample-history mask.
///
/// `region`, when present, further restricts which low-resolution texels
/// are scored (moving pixels, for example). It is not an occlusion test.
/// The error compares the predicted frame-to-frame change with the
/// reference change, rather than rewarding a temporally stable but biased
/// image.
pub fn temporal_error(
    prediction: [&[f32]; 2],
    reference: [&[f32]; 2],
    warp: crate::temporal::Reprojection<'_>,
    region: Option<&[bool]>,
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
    assert_eq!(warp.motion.len(), low_width * low_height * 2);
    assert_eq!(warp.current.len(), width * height);
    assert_eq!(warp.previous.len(), width * height);
    if let Some(region) = region {
        assert_eq!(region.len(), low_width * low_height);
    }

    let mut sum = 0.0f64;
    let mut count = 0usize;
    for y in 0..height {
        for x in 0..width {
            let low_index = (y / scale) * low_width + x / scale;
            if region.is_some_and(|keep| !keep[low_index]) {
                continue;
            }
            let previous_x = x as f32 + warp.motion[low_index * 2] * scale as f32;
            let previous_y = y as f32 + warp.motion[low_index * 2 + 1] * scale as f32;
            let position = [previous_x, previous_y];
            let current_surface = warp.current[y * width + x];
            let Some(predicted_prev) = crate::temporal::sample_reprojected(
                previous,
                warp.previous,
                current_surface,
                position,
                width,
                height,
                warp.rejection,
            ) else {
                continue;
            };
            let Some(reference_prev) = crate::temporal::sample_reprojected(
                previous_reference,
                warp.previous,
                current_surface,
                position,
                width,
                height,
                warp.rejection,
            ) else {
                continue;
            };
            let offset = (y * width + x) * 3;
            for channel in 0..3 {
                let predicted_change = crate::transform::compress(current[offset + channel])
                    - crate::transform::compress(predicted_prev[channel]);
                let reference_change =
                    crate::transform::compress(current_reference[offset + channel])
                        - crate::transform::compress(reference_prev[channel]);
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
mod tests {
    use super::{detail, error, relative_error};

    const EXTENT: usize = 32;

    fn box_blur(image: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; image.len()];
        for y in 0..EXTENT {
            for x in 0..EXTENT {
                for c in 0..3 {
                    let mut sum = 0.0;
                    for dy in -1i32..=1 {
                        let sy = (y as i32 + dy).clamp(0, EXTENT as i32 - 1) as usize;
                        for dx in -1i32..=1 {
                            let sx = (x as i32 + dx).clamp(0, EXTENT as i32 - 1) as usize;
                            sum += image[(sy * EXTENT + sx) * 3 + c];
                        }
                    }
                    out[(y * EXTENT + x) * 3 + c] = sum / 9.0;
                }
            }
        }
        out
    }

    /// The reason [`detail`] exists.
    ///
    /// A rendered frame is mostly smooth with sparse edges, so an area-averaged
    /// score is decided by the smooth part while the eye is drawn to the edges.
    /// Softening every edge here leaves the reported error at a value the
    /// project would call a good result, and takes most of the detail with it.
    #[test]
    fn detail_moves_where_error_barely_does() {
        let mut reference = vec![0.4f32; EXTENT * EXTENT * 3];
        for y in 0..EXTENT {
            for x in [10usize, 21] {
                for c in 0..3 {
                    reference[(y * EXTENT + x) * 3 + c] = 0.6;
                }
            }
        }
        let blurred = box_blur(&reference);

        let psnr = -10.0 * (error(&blurred, &reference) as f64).log10();
        assert!(
            psnr > 30.0,
            "the premise is that this blur is cheap in error, but it cost {psnr:.2} dB"
        );
        let kept = detail(&blurred, EXTENT, EXTENT) / detail(&reference, EXTENT, EXTENT);
        assert!(
            kept < 0.7,
            "a {psnr:.2} dB blur kept {:.0}% of the detail, and only this metric says so",
            100.0 * kept
        );
    }

    /// What [`relative_error`] is for: the same proportional mistake costs the
    /// same wherever it is made. Compressed error instead calls the brighter of
    /// two identical mistakes an order of magnitude smaller, because it squashes
    /// the top of the range hardest.
    #[test]
    fn relative_error_scores_the_same_ratio_alike() {
        let (dim, dim_reference) = (vec![2.0f32], vec![1.0f32]);
        let (bright, bright_reference) = (vec![20.0f32], vec![10.0f32]);

        let dim_relative = relative_error(&dim, &dim_reference);
        let bright_relative = relative_error(&bright, &bright_reference);
        assert!(
            (dim_relative - bright_relative).abs() < 0.05 * bright_relative,
            "both are a factor of two wrong: {dim_relative:.4} against {bright_relative:.4}"
        );

        let ratio = error(&dim, &dim_reference) / error(&bright, &bright_reference);
        assert!(
            ratio > 5.0,
            "compressed error should rank the bright mistake far cheaper, got {ratio:.1}x"
        );
    }
}

#[cfg(test)]
mod temporal_tests {
    use super::temporal_error;
    use crate::temporal::{RejectionConfig, Surface};

    fn flat(depth: f32) -> Surface {
        Surface {
            depth,
            normal: [0.0, 0.0, 1.0],
            albedo: [0.5, 0.5, 0.5],
        }
    }

    fn surfaces(depth: f32) -> [Surface; 4] {
        [flat(depth); 4]
    }

    #[test]
    fn temporal_error_measures_excess_change_not_scene_motion() {
        let previous_reference = vec![1.0; 2 * 2 * 3];
        let current_reference = vec![2.0; 2 * 2 * 3];
        let motion = vec![0.0; 2 * 2 * 2];
        let now = surfaces(1.0);
        let then = surfaces(1.0);
        let warp = crate::temporal::Reprojection {
            motion: &motion,
            current: &now,
            previous: &then,
            rejection: RejectionConfig::default(),
        };
        assert_eq!(
            temporal_error(
                [&current_reference, &previous_reference],
                [&current_reference, &previous_reference],
                warp,
                None,
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
                warp,
                None,
                [2, 2],
                1,
            )
            .unwrap()
            .mean()
                > 0.0
        );
    }

    #[test]
    fn temporal_error_excludes_occluded_and_out_of_crop_history() {
        let image = vec![1.0; 2 * 2 * 3];
        let motion = vec![10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0];
        let now = surfaces(1.0);
        let then = surfaces(1.0);
        assert_eq!(
            temporal_error(
                [&image, &image],
                [&image, &image],
                crate::temporal::Reprojection {
                    motion: &motion,
                    current: &now,
                    previous: &then,
                    rejection: RejectionConfig::default(),
                },
                None,
                [2, 2],
                1,
            ),
            None
        );
        assert_eq!(
            temporal_error(
                [&image, &image],
                [&image, &image],
                crate::temporal::Reprojection {
                    motion: &[0.0; 8],
                    current: &now,
                    previous: &surfaces(10.0),
                    rejection: RejectionConfig::default(),
                },
                None,
                [2, 2],
                1,
            ),
            None
        );
    }
}
