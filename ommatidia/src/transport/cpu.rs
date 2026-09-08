//! Scalar reference for native preparation, state update and supervision.
use super::*;

#[derive(Clone)]
pub struct Prepared {
    pub features: Vec<f32>,
    pub candidates: Vec<f32>, // scale-major, then six lobe/RGB planes
    pub history: Vec<f32>,
    pub prior: Vec<f32>, // candidate-major, then two lobe planes
    pub validity: Vec<f32>,
    pub moments: Vec<[f32; 4]>,
    pub ages: Vec<[f32; 2]>,
    /// Four accepted bilinear maps. Reused by differentiable short unrolls.
    pub indices: [Vec<u32>; 4],
    pub coefficients: [Vec<f32>; 4],
}
pub fn luminance(v: [f32; 3]) -> f32 {
    0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
}
pub fn encode(v: f32, exposure: f32) -> f32 {
    let v = v.max(0.0) * exposure;
    v / (1.0 + v)
}
pub fn geometry(a: [f32; 4], b: [f32; 4]) -> f32 {
    if a[3] >= 60000.0 || b[3] >= 60000.0 {
        return if a[3] >= 60000.0 && b[3] >= 60000.0 {
            1.0
        } else {
            0.0
        };
    }
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).max(0.0);
    let d = (a[3] - b[3]).abs() / (0.01 + 0.02 * a[3].abs());
    dot.powi(16) * (-d).exp()
}
pub fn matches(s: &Surface, p: &State) -> bool {
    let mut expected = s.normal_depth;
    if s.motion[2] > 0.0 {
        expected[3] = s.motion[2];
    }
    geometry(expected, p.normal_depth) > 0.1
        && (0..3)
            .map(|c| (s.albedo_roughness[c] - p.albedo_roughness[c]).powi(2))
            .sum::<f32>()
            < 0.04
}

pub fn prepare(frame: &Frame, previous: &[State], config: Config) -> Prepared {
    frame.validate(config).expect("invalid transport frame");
    let low = frame.low;
    let width = (low[0] * config.scale) as usize;
    let height = (low[1] * config.scale) as usize;
    let n = width * height;
    assert!(previous.is_empty() || previous.len() == n);
    let index = |c, x, y| config.index(low, c, x, y);
    let mut p = Prepared {
        features: vec![0.0; FEATURES * n],
        candidates: vec![0.0; SCALES * 6 * n],
        history: vec![0.0; 6 * n],
        prior: vec![0.0; CANDIDATES * 2 * n],
        validity: vec![0.0; 2 * n],
        moments: vec![[0.0; 4]; n],
        ages: vec![[1.0; 2]; n],
        indices: std::array::from_fn(|_| vec![0; 6 * n]),
        coefficients: std::array::from_fn(|_| vec![0.0; 6 * n]),
    };
    // Geometry-guided reconstruction of the exact jittered low-resolution taps.
    for y in 0..height {
        for x in 0..width {
            let s = frame.surfaces[y * width + x];
            let qx = (x as f32 + 0.5) / config.scale as f32 - 0.5 - frame.jitter[0];
            let qy = (y as f32 + 0.5) / config.scale as f32 - 0.5 - frame.jitter[1];
            let mut sum = [0.0; 6];
            let mut total = 0.0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let sx = ((qx + 0.5).floor() as i32 + dx).clamp(0, low[0] as i32 - 1) as usize;
                    let sy = ((qy + 0.5).floor() as i32 + dy).clamp(0, low[1] as i32 - 1) as usize;
                    let r = frame.rays[sy * low[0] as usize + sx];
                    let distance = (sx as f32 - qx).powi(2) + (sy as f32 - qy).powi(2);
                    let w = geometry(s.normal_depth, r.normal_depth) * (-2.0 * distance).exp();
                    for c in 0..3 {
                        sum[c] += w * r.diffuse[c];
                        sum[3 + c] += w * r.specular[c];
                    }
                    total += w;
                }
            }
            if total <= 1e-12 {
                // Missing support is explicit in the design; nearest finite ray is a reset fallback.
                let sx = ((qx + 0.5).floor() as i32).clamp(0, low[0] as i32 - 1) as usize;
                let sy = ((qy + 0.5).floor() as i32).clamp(0, low[1] as i32 - 1) as usize;
                let r = frame.rays[sy * low[0] as usize + sx];
                sum[..3].copy_from_slice(&r.diffuse[..3]);
                sum[3..].copy_from_slice(&r.specular[..3]);
                total = 1.0;
            }
            for (c, v) in sum.into_iter().enumerate() {
                p.candidates[index(c, x, y)] = v.max(0.0) / total;
            }
        }
    }
    // Fixed multiscale support is separate from the predictor's receptive field.
    for level in 1..SCALES {
        let step = 1i32 << (level - 1);
        for y in 0..height {
            for x in 0..width {
                let s = frame.surfaces[y * width + x];
                for lobe in 0..2 {
                    let mut sum = [0.0; 3];
                    let mut total = 0.0;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let sx = (x as i32 + dx * step).clamp(0, width as i32 - 1) as usize;
                            let sy = (y as i32 + dy * step).clamp(0, height as i32 - 1) as usize;
                            let t = frame.surfaces[sy * width + sx];
                            let mut w = geometry(s.normal_depth, t.normal_depth);
                            if lobe == 1 {
                                w *= (-(s.albedo_roughness[3] - t.albedo_roughness[3]).abs()
                                    * 16.0)
                                    .exp();
                            }
                            w *= if dx == 0 { 2.0 } else { 1.0 };
                            w *= if dy == 0 { 2.0 } else { 1.0 };
                            for (c, v) in sum.iter_mut().enumerate() {
                                *v += w * p.candidates
                                    [(level - 1) * 6 * n + index(lobe * 3 + c, sx, sy)];
                            }
                            total += w;
                        }
                    }
                    for (c, v) in sum.into_iter().enumerate() {
                        p.candidates[level * 6 * n + index(lobe * 3 + c, x, y)] =
                            v / total.max(1e-12);
                    }
                }
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let s = frame.surfaces[i];
            for lobe in 0..2 {
                let motion = if lobe == 1 && s.specular_motion[2] > 0.5 {
                    s.specular_motion
                } else {
                    s.motion
                };
                let q = [x as f32 + motion[0], y as f32 + motion[1]];
                let mut history = [0.0; 3];
                let mut age = 0.0;
                let mut moments = [0.0; 2];
                let mut coverage = 0.0;
                let mut weights = [0.0; 4];
                let mut ids = [0usize; 4];
                if !previous.is_empty()
                    && q[0] >= 0.0
                    && q[1] >= 0.0
                    && q[0] <= (width - 1) as f32
                    && q[1] <= (height - 1) as f32
                {
                    let tx = q[0] - q[0].floor();
                    let ty = q[1] - q[1].floor();
                    for k in 0..4 {
                        let sx = (q[0].floor() as usize + k % 2).min(width - 1);
                        let sy = (q[1].floor() as usize + k / 2).min(height - 1);
                        let old = previous[sy * width + sx];
                        let h = if lobe == 0 { old.diffuse } else { old.specular };
                        if h[3] <= 0.0 || !matches(&s, &old) {
                            continue;
                        }
                        let w = if k % 2 == 0 { 1.0 - tx } else { tx }
                            * if k / 2 == 0 { 1.0 - ty } else { ty };
                        weights[k] = w;
                        ids[k] = sy * width + sx;
                        coverage += w;
                        age += w * h[3];
                        for c in 0..3 {
                            history[c] += w * h[c];
                        }
                        for (c, v) in moments.iter_mut().enumerate() {
                            *v += w * old.moments[2 * lobe + c];
                        }
                    }
                }
                if coverage > 1e-6 {
                    for v in &mut history {
                        *v /= coverage;
                    }
                    age /= coverage;
                    for v in &mut moments {
                        *v /= coverage;
                    }
                }
                let rgb: [f32; 3] =
                    std::array::from_fn(|c| p.candidates[index(lobe * 3 + c, x, y)]);
                let lum = luminance(rgb);
                let mut spatial_var = 0.0;
                let mut count = 0.0;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let sx = (x as i32 + dx * config.scale as i32).clamp(0, width as i32 - 1)
                            as usize;
                        let sy = (y as i32 + dy * config.scale as i32).clamp(0, height as i32 - 1)
                            as usize;
                        let w =
                            geometry(s.normal_depth, frame.surfaces[sy * width + sx].normal_depth);
                        let v = luminance(std::array::from_fn(|c| {
                            p.candidates[index(lobe * 3 + c, sx, sy)]
                        }));
                        spatial_var += w * (v - lum).powi(2);
                        count += w;
                    }
                }
                spatial_var /= count.max(1e-6);
                let broad = luminance(std::array::from_fn(|c| {
                    p.candidates[(SCALES - 1) * 6 * n + index(lobe * 3 + c, x, y)]
                }));
                let delta = (broad - luminance(history)).powi(2);
                let variance = spatial_var
                    + (moments[1] - moments[0] * moments[0]).max(0.0) / age.max(1.0)
                    + 0.01 * (1.0 + broad * broad);
                let reactive = s.motion[3]
                    .clamp(0.0, 1.0)
                    .max(((delta / variance - 4.0) / 16.0).clamp(0.0, 1.0));
                let max_age = if lobe == 0 {
                    config.diffuse_frames
                } else {
                    config.specular_frames
                };
                let retained = age.min(max_age - 1.0) * coverage * (1.0 - reactive);
                let h = retained / (1.0 + retained);
                p.ages[i][lobe] = 1.0 + retained;
                p.moments[i][2 * lobe] = (1.0 - h) * lum + h * moments[0];
                p.moments[i][2 * lobe + 1] = (1.0 - h) * lum * lum + h * moments[1];
                p.validity[index(lobe, x, y)] = if coverage > 1e-6 { 1.0 } else { 0.0 };
                for (c, v) in history.iter().enumerate() {
                    let j = index(lobe * 3 + c, x, y);
                    p.history[j] = *v;
                    for k in 0..4 {
                        p.indices[k][j] =
                            index(lobe * 3 + c, ids[k] % width, ids[k] / width) as u32;
                        p.coefficients[k][j] = weights[k] / coverage.max(1e-6);
                    }
                }
                let mut prior = [0.0; SCALES];
                let mut total = 0.0;
                for (k, v) in prior.iter_mut().enumerate() {
                    *v = [0.02, 0.06, 0.12, 0.3, 0.5][k];
                    if lobe == 1 {
                        *v *= (-(k as f32) * (1.0 - s.albedo_roughness[3]) * 2.0).exp();
                    }
                    total += *v;
                }
                for (k, v) in prior.iter().enumerate() {
                    p.prior[k * 2 * n + index(lobe, x, y)] = (1.0 - h) * v / total;
                }
                p.prior[SCALES * 2 * n + index(lobe, x, y)] = h;
                p.features[index(38 + lobe, x, y)] = encode(spatial_var.sqrt(), config.exposure);
                p.features[index(40 + lobe, x, y)] = p.ages[i][lobe] / max_age;
                p.features[index(42 + lobe, x, y)] = p.validity[index(lobe, x, y)];
            }
            for k in 0..SCALES {
                for c in 0..6 {
                    p.features[index(k * 6 + c, x, y)] =
                        encode(p.candidates[k * 6 * n + index(c, x, y)], config.exposure);
                }
            }
            for c in 0..3 {
                p.features[index(30 + c, x, y)] = s.normal_depth[c];
                p.features[index(34 + c, x, y)] = s.albedo_roughness[c];
            }
            p.features[index(33, x, y)] = 1.0 / (1.0 + s.normal_depth[3].max(0.0));
            p.features[index(37, x, y)] = s.albedo_roughness[3];
        }
    }
    p
}

pub fn reconstruct(p: &Prepared, multipliers: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = p.history.len() / 6;
    assert_eq!(multipliers.len(), 2 * CANDIDATES * n);
    let mut weights = vec![0.0; multipliers.len()];
    let mut image = vec![0.0; 6 * n];
    for l in 0..2 {
        for i in 0..n {
            let sum = (0..CANDIDATES)
                .map(|k| {
                    p.prior[k * 2 * n + l * n + i]
                        * multipliers[k * 2 * n + l * n + i].max(MIN_MULTIPLIER)
                })
                .sum::<f32>();
            for k in 0..CANDIDATES {
                let w = p.prior[k * 2 * n + l * n + i]
                    * multipliers[k * 2 * n + l * n + i].max(MIN_MULTIPLIER)
                    / sum.max(1e-12);
                weights[k * 2 * n + l * n + i] = w;
                for c in 0..3 {
                    let j = (l * 3 + c) * n + i;
                    image[j] += w * if k == SCALES {
                        p.history[j]
                    } else {
                        p.candidates[k * 6 * n + j]
                    };
                }
            }
        }
    }
    (image, weights)
}
pub fn commit(
    frame: &Frame,
    p: &Prepared,
    image: &[f32],
    config: Config,
) -> (Vec<State>, Vec<f32>) {
    let width = (frame.low[0] * config.scale) as usize;
    let n = frame.surfaces.len();
    assert_eq!(image.len(), 6 * n);
    let mut states = Vec::with_capacity(n);
    let mut rgb = vec![0.0; 3 * n];
    for i in 0..n {
        let s = frame.surfaces[i];
        let mut state = State {
            moments: p.moments[i],
            normal_depth: s.normal_depth,
            albedo_roughness: s.albedo_roughness,
            ..State::default()
        };
        for c in 0..3 {
            state.diffuse[c] = image[config.index(frame.low, c, i % width, i / width)];
            state.specular[c] = image[config.index(frame.low, 3 + c, i % width, i / width)];
            rgb[3 * i + c] =
                state.diffuse[c] * s.albedo_roughness[c] + state.specular[c] + s.emission[c];
        }
        state.diffuse[3] = p.ages[i][0];
        state.specular[3] = p.ages[i][1];
        states.push(state);
    }
    (states, rgb)
}
/// Detached analytic confidence labels, excluding invalid or unidentifiable pairs.
pub fn confidence(p: &Prepared, target: &Target) -> (Vec<f32>, Vec<f32>) {
    let n = p.history.len() / 6;
    let mut labels = vec![0.0; 2 * n];
    let mut mask = labels.clone();
    for l in 0..2 {
        for i in 0..n {
            let h = p.prior[SCALES * 2 * n + l * n + i];
            let current = std::array::from_fn(|c| {
                (0..SCALES)
                    .map(|k| {
                        p.prior[k * 2 * n + l * n + i]
                            * p.candidates[k * 6 * n + (3 * l + c) * n + i]
                    })
                    .sum::<f32>()
                    / (1.0 - h).max(1e-6)
            });
            let old = std::array::from_fn(|c| p.history[(3 * l + c) * n + i]);
            let reference = std::array::from_fn(|c| target.lobes[(3 * l + c) * n + i]);
            if let Some(t) = crate::fusion::confidence_target(
                current,
                old,
                reference,
                p.validity[l * n + i] > 0.0 && h > 0.0,
            ) {
                labels[l * n + i] = t.history_share;
                mask[l * n + i] = t.separation.min(1.0).sqrt();
            }
        }
    }
    (labels, mask)
}
