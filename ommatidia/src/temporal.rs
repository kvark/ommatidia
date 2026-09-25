use serde::{Deserialize, Serialize};

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
