//! One direct-radiance recurrent model. Linear lobes and learned latent state.
//!
//! `Frame` contains observations only. `Target` is a separate training type.
//! No ground-truth geometry, light or radiance can enter through the target API.
pub mod cpu;
pub mod graph;
pub mod native;

use crate::dataset::{Layout, Plane, Sample};
use serde::{Deserialize, Serialize};

/// LR samples (6), LR normal/depth (4), and projection jitter (2).
pub const LR_FEATURES: usize = 12;
/// HR normal/depth (4), albedo/roughness (4), F0 (3), motion (2), sample offsets (2).
pub const SURFACE_FEATURES: usize = 15;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub version: u32,
    pub scale: u32,
    pub channels: u32,
    pub latent_channels: u32,
    pub levels: u32,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            version: 4,
            scale: 2,
            channels: 16,
            latent_channels: 4,
            levels: 3,
        }
    }
}
impl Config {
    pub fn validate(self, low: [u32; 2]) -> Result<(), String> {
        if self.version != 4 {
            return Err("config v4 is required; v3 guide/residual weights are archived and cannot be loaded by this runtime".into());
        }
        if !(1..=5).contains(&self.levels)
            || !(1..=4).contains(&self.scale)
            || self.channels == 0
            || self.latent_channels == 0
        {
            return Err(
                "v4 requires scale 1..4, levels 1..5, and nonzero width and latent channels".into(),
            );
        }
        let divisor = (1 << (self.levels - 1)).max(4);
        if low.iter().any(|v| *v < divisor || v % divisor != 0) {
            return Err(format!(
                "LR dimensions must be divisible by {divisor} (pyramid and loss grid)"
            ));
        }
        Ok(())
    }
    /// Packed recurrent tensor: six lobes, C latent, four normal/depth, three albedo.
    pub fn state_channels(self) -> usize {
        13 + self.latent_channels as usize
    }
    pub fn observation_channels(self) -> usize {
        LR_FEATURES + SURFACE_FEATURES * self.scale.pow(2) as usize
    }
    pub fn input_channels(self) -> usize {
        self.observation_channels()
            + (11 + self.latent_channels as usize) * self.scale.pow(2) as usize
    }
    pub fn parse(text: &str) -> Result<Self, String> {
        // Read only the version first, so old fields do not obscure the migration error.
        #[derive(Deserialize)]
        struct Version {
            version: u32,
        }
        let version: Version = ron::from_str(text).map_err(|e| e.to_string())?;
        if version.version != 4 {
            return Err(
                "config v4 is required; v3 guide/residual checkpoints are not compatible".into(),
            );
        }
        ron::from_str(text).map_err(|e| e.to_string())
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
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Surface {
    pub normal_depth: [f32; 4],
    pub albedo_roughness: [f32; 4],
    /// Output-pixel motion xy; zw are padding, not rejection or reactive controls.
    pub motion: [f32; 4],
    pub specular_f0: [f32; 4],
    /// Exact directly visible emission: a primary-surface material property.
    pub emission: [f32; 4],
}
#[derive(Clone, Debug, Default)]
pub struct State {
    /// Config::state_channels() subpixel-packed, channel-major planes.
    pub values: Vec<f32>,
}
#[derive(Clone)]
pub struct Frame {
    pub low: [u32; 2],
    pub jitter: [f32; 2],
    /// Positive host-provided normalization; never changes scene-linear output units.
    pub exposure: f32,
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
            Plane::SpecularF0,
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
            }
            r.normal_depth[3] = lr(Plane::Depth, 0, i);
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
                s.specular_f0[c] = hr(Plane::SpecularF0, c, i);
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
            exposure: 1.0,
        };
        result.validate(config)?;
        Ok(result)
    }
    pub fn validate(&self, config: Config) -> Result<(), String> {
        config.validate(self.low)?;
        let n = (self.low[0] * self.low[1]) as usize;
        if self.rays.len() != n
            || !self.exposure.is_finite()
            || self.exposure <= 0.0
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
        if self.rays.iter().any(|r| {
            r.diffuse[..3]
                .iter()
                .chain(&r.specular[..3])
                .any(|v| *v < 0.0)
        }) {
            return Err("observed radiance must be nonnegative".into());
        }
        if self.rays.iter().any(|r| r.normal_depth[3] < 0.0)
            || self.surfaces.iter().any(|s| s.normal_depth[3] < 0.0)
        {
            return Err("observed depth must be nonnegative".into());
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
