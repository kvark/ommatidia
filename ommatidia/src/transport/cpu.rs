//! Scalar observation packing, geometric warp and physical composition oracle.
use super::*;

#[derive(Clone, Debug)]
pub struct Prepared {
    pub features: Vec<f32>,
    pub history: Vec<f32>,
    pub validity: Vec<f32>,
    pub metadata: Vec<f32>,
    pub exposure: f32,
    pub indices: [Vec<u32>; 4],
    pub coefficients: [Vec<f32>; 4],
}

pub fn encode(value: f32, exposure: f32) -> f32 {
    let v = value * exposure;
    v / (1.0 + v)
}

/// Motion is current-to-previous in HR pixels. No appearance-based rejection.
pub fn taps(
    motion: [f32; 2],
    extent: [usize; 2],
    pixel: [usize; 2],
    ready: bool,
) -> ([usize; 4], [f32; 4]) {
    let mut ids = [0; 4];
    let mut weights = [0.0; 4];
    let q = [pixel[0] as f32 + motion[0], pixel[1] as f32 + motion[1]];
    if !ready || (0..2).any(|c| !q[c].is_finite() || q[c] <= -1.0 || q[c] >= extent[c] as f32) {
        return (ids, weights);
    }
    let base = [q[0].floor() as i32, q[1].floor() as i32];
    let f = [q[0] - base[0] as f32, q[1] - base[1] as f32];
    for k in 0..4 {
        let dx = k % 2;
        let dy = k / 2;
        let x = base[0] + dx as i32;
        let y = base[1] + dy as i32;
        if x >= 0 && y >= 0 && x < extent[0] as i32 && y < extent[1] as i32 {
            ids[k] = y as usize * extent[0] + x as usize;
            weights[k] = (if dx == 0 { 1.0 - f[0] } else { f[0] })
                * (if dy == 0 { 1.0 - f[1] } else { f[1] });
        }
    }
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    }
    (ids, weights)
}

pub fn prepare(frame: &Frame, previous: &State, config: Config) -> Prepared {
    frame.validate(config).unwrap();
    let n = frame.surfaces.len();
    let lr = frame.rays.len();
    let width = (frame.low[0] * config.scale) as usize;
    let height = (frame.low[1] * config.scale) as usize;
    let count = config.state_channels() * n;
    let ready = !previous.values.is_empty();
    assert!(!ready || previous.values.len() == count);
    let mut p = Prepared {
        features: vec![0.0; config.observation_channels() * lr],
        history: if ready {
            previous.values.clone()
        } else {
            vec![0.0; count]
        },
        validity: vec![0.0; n],
        metadata: vec![0.0; 7 * n],
        exposure: frame.exposure,
        indices: std::array::from_fn(|_| vec![0; count]),
        coefficients: std::array::from_fn(|_| vec![0.0; count]),
    };
    for (i, ray) in frame.rays.iter().enumerate() {
        for c in 0..3 {
            p.features[c * lr + i] = encode(ray.diffuse[c], frame.exposure);
            p.features[(c + 3) * lr + i] = encode(ray.specular[c], frame.exposure);
            p.features[(c + 6) * lr + i] = ray.normal_depth[c];
        }
        p.features[9 * lr + i] = encode(ray.normal_depth[3], 1.0);
        p.features[10 * lr + i] = frame.jitter[0];
        p.features[11 * lr + i] = frame.jitter[1];
    }
    for (i, s) in frame.surfaces.iter().enumerate() {
        let (x, y) = (i % width, i / width);
        let index = |c| config.index(frame.low, c, x, y);
        let mut observation = [0.0; SURFACE_FEATURES];
        observation[..4].copy_from_slice(&s.normal_depth);
        observation[3] = encode(s.normal_depth[3], 1.0);
        observation[4..8].copy_from_slice(&s.albedo_roughness);
        observation[8..11].copy_from_slice(&s.specular_f0[..3]);
        observation[11..13].copy_from_slice(&s.motion[..2]);
        let scale = config.scale as usize;
        observation[13] =
            (x % scale) as f32 / scale as f32 + 0.5 / scale as f32 - 0.5 - frame.jitter[0];
        observation[14] =
            (y % scale) as f32 / scale as f32 + 0.5 / scale as f32 - 0.5 - frame.jitter[1];
        for (c, value) in observation.into_iter().enumerate() {
            p.features[LR_FEATURES * lr + index(c)] = value;
        }
        for (c, &value) in observation[..4].iter().enumerate() {
            p.metadata[index(c)] = value;
        }
        for c in 0..3 {
            p.metadata[index(c + 4)] = s.albedo_roughness[c];
        }
        let (ids, weights) = taps([s.motion[0], s.motion[1]], [width, height], [x, y], ready);
        p.validity[index(0)] = f32::from(weights.iter().any(|w| *w > 0.0));
        for c in 0..config.state_channels() {
            for k in 0..4 {
                p.indices[k][index(c)] =
                    config.index(frame.low, c, ids[k] % width, ids[k] / width) as u32;
                p.coefficients[k][index(c)] = weights[k];
            }
        }
    }
    p
}

pub fn warp(p: &Prepared) -> Vec<f32> {
    (0..p.history.len())
        .map(|i| {
            (0..4)
                .map(|k| p.history[p.indices[k][i] as usize] * p.coefficients[k][i])
                .sum()
        })
        .collect()
}

pub fn decode(
    z: &[f32],
    gates: &[f32],
    history: &[f32],
    valid: &[f32],
    exposure: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = valid.len();
    assert_eq!(z.len(), 6 * n);
    assert_eq!(history.len(), 6 * n);
    assert_eq!(gates.len(), 2 * n);
    let alpha: Vec<_> = gates
        .iter()
        .enumerate()
        .map(|(i, a)| valid[i % n] / (1.0 + (-a).exp()))
        .collect();
    let image = z
        .iter()
        .enumerate()
        .map(|(i, z)| {
            let a = alpha[i / (3 * n) * n + i % n];
            let spatial = z.clamp(-16.0, 11.0).exp() / exposure;
            a * history[i] + (1.0 - a) * spatial
        })
        .collect();
    (image, alpha)
}

pub fn commit(frame: &Frame, lobes: &[f32], latent: &[f32], config: Config) -> (State, Vec<f32>) {
    let n = frame.surfaces.len();
    assert_eq!(lobes.len(), 6 * n);
    assert_eq!(latent.len(), config.latent_channels as usize * n);
    let mut values = Vec::with_capacity(config.state_channels() * n);
    values.extend_from_slice(lobes);
    values.extend_from_slice(latent);
    values.resize(config.state_channels() * n, 0.0);
    let offset = (6 + config.latent_channels as usize) * n;
    let width = (frame.low[0] * config.scale) as usize;
    let mut rgb = vec![0.0; 3 * n];
    for (i, s) in frame.surfaces.iter().enumerate() {
        let index = |c| config.index(frame.low, c, i % width, i / width);
        for c in 0..4 {
            values[offset + index(c)] = if c == 3 {
                encode(s.normal_depth[c], 1.0)
            } else {
                s.normal_depth[c]
            };
        }
        for c in 0..3 {
            values[offset + index(c + 4)] = s.albedo_roughness[c];
            rgb[3 * i + c] =
                s.albedo_roughness[c] * lobes[index(c)] + lobes[index(c + 3)] + s.emission[c];
        }
    }
    (State { values }, rgb)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn warp_renormalizes_partial_border_and_rejects_only_no_coverage_or_reset() {
        let (ids, weights) = taps([-0.25, 0.5], [8, 8], [0, 2], true);
        assert_eq!(weights, [0.0, 0.5, 0.0, 0.5]);
        assert_eq!((ids[1], ids[3]), (16, 24));
        for (motion, ready) in [
            ([-1.0, 0.0], true),
            ([f32::MAX, 0.0], true),
            ([0.0, 0.0], false),
        ] {
            assert_eq!(taps(motion, [8, 8], [0, 0], ready).1, [0.0; 4]);
        }
    }
    #[test]
    fn zero_alpha_is_spatial_and_invalid_history_is_ignored() {
        let z = [0.25; 6];
        let spatial = vec![0.25_f32.exp() / 2.0; 6];
        assert_eq!(
            decode(&z, &[-1000.0; 2], &[999.0; 6], &[1.0], 2.0).0,
            spatial
        );
        assert_eq!(
            decode(&z, &[1000.0; 2], &[999.0; 6], &[0.0], 2.0).0,
            spatial
        );
    }
    #[test]
    fn exposure_compensates_radiance_units() {
        let (a, _) = decode(&[0.25; 6], &[1.0; 2], &[3.0; 6], &[1.0], 2.0);
        let (b, _) = decode(&[0.25; 6], &[1.0; 2], &[24.0; 6], &[1.0], 0.25);
        for (a, b) in a.into_iter().zip(b) {
            assert!((8.0 * a - b).abs() < 1e-5);
        }
    }
}
