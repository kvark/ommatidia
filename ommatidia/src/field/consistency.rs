//! Agreement between source-ray depth distributions and the volume they construct.
//! Batches contain only observed cameras/bounds; no truth or scene identifiers.
use super::{Config, Observations, Ray, data, visibility::BINS};
use crate::rng::Rng;
use meganeura::{Graph, NodeId};

#[derive(Debug)]
pub struct Batch {
    pub rays: Vec<Ray>,
    pub views: Vec<usize>,
    pub pixels: Vec<usize>,
    pub valid: Vec<f32>,
    indices: Vec<Vec<u32>>,
    masks: Vec<Vec<f32>>,
}
impl Batch {
    pub fn new(obs: &Observations, c: &Config, count: usize, seed: u64) -> Result<Self, String> {
        obs.validate(c)?;
        if count == 0 || count > 65536 {
            return Err("consistency needs 1..65536 source rays".into());
        }
        let [w, h] = c.extent;
        let mut rng = Rng::new(seed);
        let mut out = Self {
            rays: Vec::new(),
            views: Vec::new(),
            pixels: Vec::new(),
            valid: Vec::new(),
            indices: vec![vec![0; count]; c.views],
            masks: vec![vec![0.0; count]; c.views],
        };
        for i in 0..count {
            let v = rng.below(c.views as u32) as usize;
            let p = rng.below(w * h) as usize;
            let ray = obs.views[v]
                .camera
                .ray([(p % w as usize) as f32, (p / w as usize) as f32], c.extent);
            let (near, far) = data::ray_interval(obs.bounds, ray);
            let valid = u8::from(far > near) as f32;
            out.rays.push(ray);
            out.views.push(v);
            out.pixels.push(p);
            out.valid.push(valid);
            out.indices[v][i] = p as u32;
            out.masks[v][i] = valid;
        }
        Ok(out)
    }
    pub fn feed(&self, session: &mut meganeura::Session, weight: f32) -> Result<(), String> {
        if !weight.is_finite() || weight < 0.0 {
            return Err("invalid consistency weight".into());
        }
        for v in 0..self.indices.len() {
            session.set_input_u32(&format!("consistency.view{v}.indices"), &self.indices[v]);
            session.set_input(&format!("consistency.view{v}.mask"), &self.masks[v]);
        }
        // Graph MSE averages all rows/bins; normalize by certified in-bounds rows.
        let valid = self.valid.iter().sum::<f32>();
        let scale = if valid > 0.0 {
            weight * self.rays.len() as f32 / valid
        } else {
            0.0
        };
        session.set_input("consistency.weight", &[scale]);
        Ok(())
    }
}

/// Sum uniform fine interval masses into the fixed source bins; preserve escape.
/// No expected-depth collapse: multimodal distributions must remain multimodal.
pub fn coarsen(mass: &[f32]) -> Result<Vec<f32>, String> {
    let steps = mass
        .len()
        .checked_sub(1)
        .ok_or("empty termination distribution")?;
    if steps == 0
        || !steps.is_multiple_of(BINS)
        || steps > 256
        || mass.iter().any(|v| !v.is_finite() || *v < -1e-6)
        || (mass.iter().sum::<f32>() - 1.0).abs() > 1e-4
    {
        return Err(
            "expected a normalized termination distribution at a multiple of 16 intervals".into(),
        );
    }
    let mut out: Vec<_> = mass[..steps]
        .chunks_exact(steps / BINS)
        .map(|v| v.iter().sum())
        .collect();
    out.push(mass[steps]);
    Ok(out)
}
/// Ordered squared-CDF distance, averaged over the finite interval boundaries.
/// Escape contributes via its complementary cumulative mass. Equal expected
/// depths alone cannot satisfy this metric.
pub fn cdf_error(a: &[f32], b: &[f32]) -> Result<f64, String> {
    let a = coarsen(a)?;
    let b = coarsen(b)?;
    let (mut ca, mut cb, mut error) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..BINS {
        ca += a[i] as f64;
        cb += b[i] as f64;
        error += (ca - cb).powi(2);
    }
    Ok(error / BINS as f64)
}

pub(super) fn loss(
    g: &mut Graph,
    probabilities: &[NodeId],
    termination: NodeId,
    shape: data::RenderShape,
    count: usize,
) -> NodeId {
    // Pixel-major rendering distributions, selecting only trailing source rays.
    let mass = g.reshape(termination, &[shape.steps + 1, shape.rays]);
    let mass = g.transpose(mass);
    let mass = g.reshape(mass, &[shape.rays * (shape.steps + 1)]);
    let mass = g.split_b(
        mass,
        1,
        ((shape.rays - count) * (shape.steps + 1)) as u32,
        (count * (shape.steps + 1)) as u32,
        1,
    );
    let mass = g.reshape(mass, &[count, shape.steps + 1]);
    // Fuse exact bin reduction and cumulative sum into a fixed linear map.
    let mut map = vec![0.0; (shape.steps + 1) * BINS];
    for s in 0..shape.steps {
        for b in s / (shape.steps / BINS)..BINS {
            map[s * BINS + b] = 1.0;
        }
    }
    let map = g.constant(map, &[shape.steps + 1, BINS]);
    let volume_cdf = g.matmul(mass, map);
    let mut source = None;
    let mut validity = None;
    for (v, p) in probabilities.iter().enumerate() {
        let ids = g.input_u32(&format!("consistency.view{v}.indices"), &[count]);
        let mask = g.input(&format!("consistency.view{v}.mask"), &[count, 1]);
        validity = Some(validity.map_or(mask, |a| g.add(a, mask)));
        let p = g.embedding(ids, *p);
        let mask = g.broadcast_inner(mask, BINS + 1);
        let p = g.mul(p, mask);
        source = Some(source.map_or(p, |a| g.add(a, p)));
    }
    let map: Vec<_> = (0..=BINS)
        .flat_map(|i| (0..BINS).map(move |j| u8::from(i <= j) as f32))
        .collect();
    let map = g.constant(map, &[BINS + 1, BINS]);
    let source_cdf = g.matmul(source.unwrap(), map);
    let mask = g.broadcast_inner(validity.unwrap(), BINS);
    let volume_cdf = g.mul(volume_cdf, mask);
    let mse = g.mse_loss(source_cdf, volume_cdf);
    let weight = g.input("consistency.weight", &[1]);
    g.mul(mse, weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coarsening_keeps_escape_and_distinguishes_equal_means() {
        let mut fine = vec![0.0; 65];
        fine[5] = 0.25;
        fine[59] = 0.25;
        fine[64] = 0.5;
        let coarse = coarsen(&fine).unwrap();
        assert_eq!(coarse[1], 0.25);
        assert_eq!(coarse[14], 0.25);
        assert_eq!(coarse[16], 0.5);
        assert!(cdf_error(&fine, &coarse).unwrap() < 1e-14);
        let mut a = vec![0.0; 17];
        a[1] = 0.5;
        a[13] = 0.5;
        let mut b = vec![0.0; 17];
        b[7] = 1.0;
        assert!(cdf_error(&a, &b).unwrap() > 0.1);
        assert!(coarsen(&[f32::NAN; 17]).is_err());
        assert!(coarsen(&[0.0; 17]).is_err());
    }
}
