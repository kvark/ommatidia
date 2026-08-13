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
