//! Multiscale, lobe-separated reconstruction. Linear state; bounded features.
//!
//! `Frame` contains observations only. `Target` is a separate training type.
//! No ground-truth geometry, light or radiance can enter through the target API.
pub mod cpu;
pub mod graph;
pub mod native;
pub mod noise;
pub mod olat;
pub mod oracle;

use crate::dataset::{Layout, Plane, Sample};
use serde::{Deserialize, Serialize};

pub const SCALES: usize = 5;
pub const CANDIDATES: usize = SCALES + 1;
pub const FEATURES: usize = 44;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Config {
    pub version: u32,
    pub scale: u32,
    pub channels: u32,
    /// Fixed feature/loss exposure. State and output remain scene-linear.
    pub exposure: f32,
    pub diffuse_frames: f32,
    pub specular_frames: f32,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            version: 1,
            scale: 2,
            channels: 8,
            exposure: 1.0,
            diffuse_frames: 32.0,
            specular_frames: 8.0,
        }
    }
}
impl Config {
    pub fn validate(self, low: [u32; 2]) -> Result<(), String> {
        if self.version != 1
            || !(1..=4).contains(&self.scale)
            || self.channels == 0
            || !self.exposure.is_finite()
            || self.exposure <= 0.0
            || [self.diffuse_frames, self.specular_frames]
                .iter()
                .any(|v| !v.is_finite() || *v < 1.0)
            || low.iter().any(|v| *v < 4 || v % 4 != 0)
        {
            return Err("transport requires v1, positive exposure/history, scale 1..4, and LR dimensions divisible by four".into());
        }
        Ok(())
    }
    pub fn index(self, low: [u32; 2], channel: usize, x: usize, y: usize) -> usize {
        let s = self.scale as usize;
        let slot = (y % s) * s + x % s;
        ((channel * s * s + slot) * low[1] as usize + y / s) * low[0] as usize + x / s
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Ray {
    pub diffuse: [f32; 4],
    pub specular: [f32; 4],
    pub normal_depth: [f32; 4],
    pub albedo_roughness: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Surface {
    pub normal_depth: [f32; 4],
    pub albedo_roughness: [f32; 4],
    /// Output-pixel motion, expected previous-view depth (0 = unavailable), reactive mask.
    pub motion: [f32; 4],
    /// Optional reflected-surface motion: xy and availability in z.
    pub specular_motion: [f32; 4],
    /// Exact directly visible emission: a primary-surface material property.
    pub emission: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct State {
    pub diffuse: [f32; 4], // rgb, age
    pub specular: [f32; 4],
    pub moments: [f32; 4], // diffuse mean, second moment, specular mean, second moment
    pub normal_depth: [f32; 4],
    pub albedo_roughness: [f32; 4],
}
#[derive(Clone)]
pub struct Frame {
    pub low: [u32; 2],
    pub jitter: [f32; 2],
    pub rays: Vec<Ray>,
    pub surfaces: Vec<Surface>,
}
#[derive(Clone)]
pub struct Target {
    /// Six channel-major, subpixel-packed planes: diffuse illumination, specular.
    pub lobes: Vec<f32>,
    pub rgb: Vec<f32>, // interleaved linear RGB for reporting
}
impl Frame {
    pub fn from_sample(sample: &Sample, layout: Layout, config: Config) -> Result<Self, String> {
        config.validate([layout.lr_width, layout.lr_height])?;
        if layout.scale != config.scale {
            return Err("scale mismatch".into());
        }
        let low = [layout.lr_width, layout.lr_height];
        for p in [
            Plane::DiffuseIllumination,
            Plane::SpecularRadiance,
            Plane::Depth,
            Plane::Normal,
            Plane::DiffuseAlbedo,
            Plane::Roughness,
        ] {
            if !layout.lr_planes.contains(p) {
                return Err(format!("missing observed LR {p:?}"));
            }
        }
        for p in [
            Plane::Depth,
            Plane::Normal,
            Plane::DiffuseAlbedo,
            Plane::Roughness,
            Plane::EmissiveRadiance,
        ] {
            if !layout.hr_planes.contains(p) {
                return Err(format!("missing observed HR {p:?}"));
            }
        }
        let lr = |p: Plane, c: usize, i: usize| {
            sample
                .lr_channel(&layout, p, c)
                .map_or(0.0, |a| a[i].to_f32())
        };
        let hr = |p: Plane, c: usize, i: usize| {
            sample
                .hr_channel(&layout, p, c)
                .map_or(0.0, |a| a[i].to_f32())
        };
        let mut rays = Vec::with_capacity(layout.lr_texels());
        for i in 0..layout.lr_texels() {
            let mut r = Ray::default();
            for c in 0..3 {
                r.diffuse[c] = lr(Plane::DiffuseIllumination, c, i);
                r.specular[c] = lr(Plane::SpecularRadiance, c, i);
                r.normal_depth[c] = lr(Plane::Normal, c, i);
                r.albedo_roughness[c] = lr(Plane::DiffuseAlbedo, c, i);
            }
            r.normal_depth[3] = lr(Plane::Depth, 0, i);
            r.albedo_roughness[3] = lr(Plane::Roughness, 0, i);
            rays.push(r);
        }
        let mut surfaces = Vec::with_capacity(layout.hr_texels());
        let width = layout.hr_width() as usize;
        for i in 0..layout.hr_texels() {
            let mut s = Surface::default();
            for c in 0..3 {
                s.normal_depth[c] = hr(Plane::Normal, c, i);
                s.albedo_roughness[c] = hr(Plane::DiffuseAlbedo, c, i);
                s.emission[c] = hr(Plane::EmissiveRadiance, c, i);
            }
            s.normal_depth[3] = hr(Plane::Depth, 0, i);
            s.albedo_roughness[3] = hr(Plane::Roughness, 0, i);
            for c in 0..2 {
                s.motion[c] = if layout.hr_planes.contains(Plane::Motion) {
                    hr(Plane::Motion, c, i)
                } else {
                    let j = (i / width / config.scale as usize) * low[0] as usize
                        + i % width / config.scale as usize;
                    lr(Plane::Motion, c, j) * config.scale as f32
                };
            }
            surfaces.push(s);
        }
        let result = Self {
            low,
            rays,
            surfaces,
            jitter: [lr(Plane::Jitter, 0, 0), lr(Plane::Jitter, 1, 0)],
        };
        result.validate(config)?;
        Ok(result)
    }
    pub fn validate(&self, config: Config) -> Result<(), String> {
        config.validate(self.low)?;
        let n = (self.low[0] * self.low[1]) as usize;
        if self.rays.len() != n
            || self.surfaces.len() != n * config.scale.pow(2) as usize
            || !self.jitter.iter().all(|v| v.is_finite())
            || bytemuck::cast_slice::<Ray, f32>(&self.rays)
                .iter()
                .any(|v| !v.is_finite())
            || bytemuck::cast_slice::<Surface, f32>(&self.surfaces)
                .iter()
                .any(|v| !v.is_finite())
        {
            return Err("invalid observation dimensions or non-finite input".into());
        }
        Ok(())
    }
}
impl Target {
    pub fn from_sample(sample: &Sample, layout: Layout, config: Config) -> Result<Self, String> {
        let n = layout.hr_texels();
        let mut lobes = vec![0.0; 6 * n];
        let mut rgb = vec![0.0; 3 * n];
        for (l, p) in [Plane::DiffuseIllumination, Plane::SpecularRadiance]
            .into_iter()
            .enumerate()
        {
            for c in 0..3 {
                let a = sample
                    .hr_channel(&layout, p, c)
                    .ok_or(format!("missing target {p:?}"))?;
                for (i, v) in a.iter().enumerate() {
                    lobes[config.index(
                        [layout.lr_width, layout.lr_height],
                        l * 3 + c,
                        i % layout.hr_width() as usize,
                        i / layout.hr_width() as usize,
                    )] = v.to_f32();
                }
            }
        }
        for c in 0..3 {
            let a = sample
                .hr_channel(&layout, Plane::Color, c)
                .ok_or("missing target RGB")?;
            for (i, v) in a.iter().enumerate() {
                rgb[i * 3 + c] = v.to_f32();
            }
        }
        Ok(Self { lobes, rgb })
    }
}
