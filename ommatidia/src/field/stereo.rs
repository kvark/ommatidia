//! Cross-view matching on camera-ray depth hypotheses. Observations only.
//!
//! A quarter-resolution lattice avoids a full-resolution 3D CNN. Every source
//! is treated symmetrically. Missing/degenerate pairs contribute no evidence.
use super::{Config, Observations, data, dot, graph, unit, visibility::BINS};
use crate::neural::Builder;
use meganeura::{Graph, NodeId};

pub const STRIDE: u32 = 4;
const MATCH_WIDTH: u32 = 16;

#[derive(Clone, Debug, PartialEq)]
pub struct Projection {
    indices: [Vec<u32>; 4],
    weights: [Vec<f32>; 4],
}
impl Projection {
    fn new(n: usize) -> Self {
        Self {
            indices: std::array::from_fn(|_| vec![0; n]),
            weights: std::array::from_fn(|_| vec![0.0; n]),
        }
    }
    fn set(&mut self, i: usize, pixel: [f32; 2], extent: [u32; 2]) -> bool {
        let [x, y] = pixel;
        let [w, h] = extent.map(|v| v as usize);
        if !x.is_finite()
            || !y.is_finite()
            || x < 0.0
            || y < 0.0
            || x > (w - 1) as f32
            || y > (h - 1) as f32
        {
            return false;
        }
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        for k in 0..4 {
            let (dx, dy) = (k % 2, k / 2);
            self.indices[k][i] = ((y0 + dy).min(h - 1) * w + (x0 + dx).min(w - 1)) as u32;
            self.weights[k][i] =
                (if dx == 0 { 1.0 - fx } else { fx }) * (if dy == 0 { 1.0 - fy } else { fy });
        }
        true
    }
    fn feed(&self, session: &mut meganeura::Session, tag: &str) {
        for k in 0..4 {
            session.set_input_u32(&format!("{tag}.index{k}"), &self.indices[k]);
            session.set_input(&format!("{tag}.weight{k}"), &self.weights[k]);
        }
    }
    fn sample(&self, rgb: &[f32], i: usize, exposure: f32) -> [f32; 3] {
        std::array::from_fn(|c| {
            (0..4)
                .map(|k| {
                    let v = rgb[self.indices[k][i] as usize * 3 + c] * exposure;
                    self.weights[k][i] * v / (1.0 + v)
                })
                .sum()
        })
    }
}
#[derive(Clone, Debug, PartialEq)]
struct ViewSweep {
    reference: Projection,
    peers: Vec<Option<Projection>>,
    inverse: Vec<f32>,
    support: Vec<f32>,
    depth: Vec<f32>,
}
/// Reusable projection maps, keyed by camera/bounds/extent only. Pixel values
/// remain live graph inputs, so source-image ablations cannot use stale RGB costs.
#[derive(Clone, Debug, PartialEq)]
pub struct Sweep {
    key: Vec<u32>,
    extent: [u32; 2],
    sources: Vec<ViewSweep>,
}
fn key(obs: &Observations, c: &Config) -> Vec<u32> {
    let mut key = vec![c.extent[0], c.extent[1], c.views as u32];
    key.extend(obs.bounds.center.map(f32::to_bits));
    key.push(obs.bounds.radius.to_bits());
    for v in &obs.views {
        for row in [
            v.camera.origin,
            v.camera.right,
            v.camera.up,
            v.camera.forward,
        ] {
            key.extend(row.map(f32::to_bits));
        }
        key.push(v.camera.tan_half_fov_y.to_bits());
    }
    key
}
impl Sweep {
    pub fn new(obs: &Observations, c: &Config) -> Result<Self, String> {
        obs.validate(c)?;
        if c.view_fusion != super::ViewFusion::StereoRgb {
            return Err("sweep needs stereo-rgb".into());
        }
        let [w, h] = c.extent;
        let [cw, ch] = [w / STRIDE, h / STRIDE];
        let spatial = (cw * ch) as usize;
        let q = spatial * BINS;
        let mut sources = Vec::new();
        for (v, source) in obs.views.iter().enumerate() {
            let mut sweep = ViewSweep {
                reference: Projection::new(q),
                peers: (0..c.views)
                    .map(|t| (v != t).then(|| Projection::new(q)))
                    .collect(),
                inverse: vec![0.0; q],
                support: vec![0.0; q],
                depth: vec![0.0; q],
            };
            for bin in 0..BINS {
                for p in 0..spatial {
                    let i = bin * spatial + p;
                    let pixel = [
                        ((p % cw as usize) as f32 + 0.5) * STRIDE as f32 - 0.5,
                        ((p / cw as usize) as f32 + 0.5) * STRIDE as f32 - 0.5,
                    ];
                    sweep.reference.set(i, pixel, c.extent);
                    sweep.depth[i] = (bin as f32 + 0.5) / BINS as f32;
                    let ray = source.camera.ray(pixel, c.extent);
                    let (near, far) = data::ray_interval(obs.bounds, ray);
                    if far <= near {
                        continue;
                    }
                    let distance = near + (far - near) * sweep.depth[i];
                    let position =
                        std::array::from_fn(|c| ray.origin[c] + ray.direction[c] * distance);
                    for (t, peer) in obs.views.iter().enumerate() {
                        let Some(projection) = &mut sweep.peers[t] else {
                            continue;
                        };
                        let delta = std::array::from_fn(|c| position[c] - peer.camera.origin[c]);
                        if dot(delta, delta) < 1e-10 {
                            continue;
                        }
                        let direction = unit(delta);
                        // Pure rotation/coincident centers provide no depth parallax.
                        if 1.0 - dot(direction, ray.direction).powi(2) < 1e-6 {
                            continue;
                        }
                        if let Some(pixel) = peer.camera.project(position, c.extent)
                            && projection.set(i, pixel, c.extent)
                        {
                            sweep.support[i] += 1.0;
                        }
                    }
                    sweep.inverse[i] = 1.0 / sweep.support[i].max(1.0);
                    sweep.support[i] /= c.views.saturating_sub(1).max(1) as f32;
                }
            }
            sources.push(sweep);
        }
        Ok(Self {
            key: key(obs, c),
            extent: c.extent,
            sources,
        })
    }
    pub fn validate_for(&self, obs: &Observations, c: &Config) -> Result<(), String> {
        if self.key != key(obs, c) {
            Err("cached stereo projections do not match acquisition geometry".into())
        } else {
            Ok(())
        }
    }
    pub fn feed(&self, session: &mut meganeura::Session) {
        for (v, s) in self.sources.iter().enumerate() {
            let tag = format!("stereo{v}");
            s.reference.feed(session, &format!("{tag}.reference"));
            for (t, p) in s.peers.iter().enumerate() {
                if let Some(p) = p {
                    p.feed(session, &format!("{tag}.peer{t}"));
                }
            }
            session.set_input(&format!("{tag}.inverse"), &s.inverse);
            session.set_input(&format!("{tag}.support"), &s.support);
            session.set_input(&format!("{tag}.depth"), &s.depth);
        }
    }
    /// Observation-only diagnostic for a textured-plane correspondence test.
    /// No learned decoder, no depth target and no inference query placements.
    pub fn photometric_cost(
        &self,
        obs: &Observations,
        c: &Config,
        source: usize,
    ) -> Result<Vec<f32>, String> {
        obs.validate(c)?;
        self.validate_for(obs, c)?;
        let s = self
            .sources
            .get(source)
            .ok_or("source index out of range")?;
        Ok((0..s.inverse.len())
            .map(|i| {
                let reference = s.reference.sample(&obs.views[source].rgb, i, c.exposure);
                s.peers
                    .iter()
                    .enumerate()
                    .filter_map(|(t, p)| p.as_ref().map(|p| (t, p)))
                    .map(|(t, p)| {
                        let valid = (0..4).map(|k| p.weights[k][i]).sum::<f32>();
                        let color = p.sample(&obs.views[t].rgb, i, c.exposure);
                        valid
                            * color
                                .iter()
                                .zip(reference)
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f32>()
                            / 3.0
                    })
                    .sum::<f32>()
                    * s.inverse[i]
            })
            .collect())
    }
}
fn project(g: &mut Graph, tag: &str, table: NodeId, q: usize, ch: usize) -> (NodeId, NodeId) {
    let mut total = None;
    let mut valid = None;
    for k in 0..4 {
        let indices = g.input_u32(&format!("{tag}.index{k}"), &[q]);
        let scalar = g.input(&format!("{tag}.weight{k}"), &[q, 1]);
        valid = Some(valid.map_or(scalar, |a| g.add(a, scalar)));
        let w = g.broadcast_inner(scalar, ch);
        let x = g.embedding(indices, table);
        let x = g.mul(x, w);
        total = Some(total.map_or(x, |a| g.add(a, x)));
    }
    (total.unwrap(), valid.unwrap())
}
/// Residual depth-hypothesis scores learned from bounded feature/RGB mismatch,
/// pair support and normalized depth. Geometry maps are detached; features are not.
pub(super) fn correction(
    b: &mut Builder,
    c: &Config,
    v: usize,
    tables: &[NodeId],
    rgb: &[Option<NodeId>],
) -> NodeId {
    let [w, h] = c.extent;
    let n = (w * h) as usize;
    let spatial = ((w / STRIDE) * (h / STRIDE)) as usize;
    let q = BINS * spatial;
    let ch = c.channels as usize;
    let combined: Vec<_> = tables
        .iter()
        .zip(rgb)
        .map(|(&t, rgb)| {
            let rgb = graph::scaled(&mut b.g, rgb.unwrap(), c.exposure);
            let one = graph::constant(&mut b.g, rgb, 1.0);
            let den = b.g.add(one, rgb);
            let encoded = b.g.div(rgb, den);
            graph::columns(&mut b.g, t, encoded, n, ch, 3)
        })
        .collect();
    let tag = format!("stereo{v}");
    let (reference, _) = project(
        &mut b.g,
        &format!("{tag}.reference"),
        combined[v],
        q,
        ch + 3,
    );
    let mut error = b.g.constant(vec![0.0; q * (ch + 3)], &[q, ch + 3]);
    for (t, &table) in combined.iter().enumerate() {
        if t == v {
            continue;
        }
        let (peer, valid) = project(&mut b.g, &format!("{tag}.peer{t}"), table, q, ch + 3);
        let neg = b.g.neg(reference);
        let diff = b.g.add(peer, neg);
        let square = b.g.mul(diff, diff);
        let one = graph::constant(&mut b.g, square, 1.0);
        let den = b.g.add(one, square);
        let bounded = b.g.div(square, den);
        let mask = b.g.broadcast_inner(valid, ch + 3);
        let bounded = b.g.mul(bounded, mask);
        error = b.g.add(error, bounded);
    }
    let inverse = b.g.input(&format!("{tag}.inverse"), &[q, 1]);
    let inverse = b.g.broadcast_inner(inverse, ch + 3);
    error = b.g.mul(error, inverse);
    let support = b.g.input(&format!("{tag}.support"), &[q, 1]);
    let depth = b.g.input(&format!("{tag}.depth"), &[q, 1]);
    let input = graph::columns(&mut b.g, error, support, q, ch + 3, 1);
    let input = graph::columns(&mut b.g, input, depth, q, ch + 4, 1);
    let hidden = b.linear(input, "field.stereo.in", (ch + 5) as u32, MATCH_WIDTH);
    let hidden = b.g.silu(hidden);
    let score = b.linear(hidden, "field.stereo.out", MATCH_WIDTH, 1);
    for p in &mut b.params {
        if p.name == "field.stereo.out.weight" {
            p.kind = crate::model::InitKind::Zeros;
        }
    }
    let score = b.g.mul(score, support);
    let score = b.g.reshape(score, &[q]);
    let score =
        b.g.upsample_2x(score, 1, BINS as u32, h / STRIDE, w / STRIDE);
    let score = b.g.upsample_2x(score, 1, BINS as u32, h / 2, w / 2);
    // No stereo evidence for escape; the monocular head learns that probability.
    let escape = b.g.constant(vec![0.0; n], &[n]);
    let score = b.g.concat(score, escape, 1, BINS as u32, 1, n as u32);
    let score = b.g.reshape(score, &[BINS + 1, n]);
    b.g.transpose(score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Bounds, Camera, View, ViewFusion};
    fn plane() -> (Config, Observations) {
        let c = Config {
            extent: [16, 16],
            views: 2,
            channels: 4,
            hidden: 16,
            view_fusion: ViewFusion::StereoRgb,
            ..Default::default()
        };
        let views = [-0.2, 0.2]
            .into_iter()
            .map(|x| {
                let camera = Camera {
                    origin: [x, 0.0, 3.0],
                    right: [1.0, 0.0, 0.0],
                    up: [0.0, 1.0, 0.0],
                    forward: [0.0, 0.0, -1.0],
                    tan_half_fov_y: 0.2,
                };
                let rgb = (0..256)
                    .flat_map(|p| {
                        let ray = camera.ray([(p % 16) as f32, (p / 16) as f32], c.extent);
                        let d = (0.0625 - 3.0) / ray.direction[2];
                        let (x, y) = (x + d * ray.direction[0], d * ray.direction[1]);
                        [0.5 + 0.3 * x, 0.5 + 0.2 * y, 0.5 + 0.15 * (x + y)]
                    })
                    .collect();
                View { camera, rgb }
            })
            .collect();
        (
            c,
            Observations {
                bounds: Bounds {
                    center: [0.0; 3],
                    radius: 1.0,
                },
                views,
            },
        )
    }
    #[test]
    fn textured_plane_selects_geometric_depth_not_a_monocular_prior() {
        let (c, obs) = plane();
        let sweep = Sweep::new(&obs, &c).unwrap();
        let costs = sweep.photometric_cost(&obs, &c, 0).unwrap();
        let mean: Vec<f32> = (0..BINS)
            .map(|b| costs[b * 16..(b + 1) * 16].iter().sum::<f32>() / 16.0)
            .collect();
        let best = (0..BINS)
            .min_by(|a, b| mean[*a].total_cmp(&mean[*b]))
            .unwrap();
        assert_eq!(best, 7, "{mean:?}");
        let mut changed = obs.clone();
        changed.views[1].rgb.fill(0.0);
        assert_ne!(costs, sweep.photometric_cost(&changed, &c, 0).unwrap());
        // Reusing camera maps after RGB changes is valid, but after pose changes is not.
        changed.views[1].camera.origin[0] += 0.1;
        assert!(sweep.validate_for(&changed, &c).is_err());
    }
    #[test]
    fn coincident_cameras_and_single_views_have_no_stereo_support() {
        let (mut c, mut obs) = plane();
        obs.views[1] = obs.views[0].clone();
        let sweep = Sweep::new(&obs, &c).unwrap();
        assert!(
            sweep
                .sources
                .iter()
                .all(|s| s.support.iter().all(|v| *v == 0.0))
        );
        c.views = 1;
        obs.views.truncate(1);
        let sweep = Sweep::new(&obs, &c).unwrap();
        assert!(sweep.sources[0].support.iter().all(|v| *v == 0.0));
    }
}
