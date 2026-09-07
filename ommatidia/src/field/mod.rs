//! Experimental image-conditioned radiance field. Posed RGB in; density/radiance out.
//! Light-source labels are a separate training contract, never observations.
pub mod data;
pub mod graph;
pub mod incident;
pub mod surface;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub version: u32,
    pub extent: [u32; 2],
    pub views: usize,
    pub channels: u32,
    pub hidden: u32,
    pub position_frequencies: u32,
    pub exposure: f32,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            version: 1,
            extent: [64, 64],
            views: 3,
            channels: 64,
            hidden: 128,
            position_frequencies: 4,
            exposure: 1.0,
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if self.version != 1
            || !(1..=8).contains(&self.views)
            || self
                .extent
                .iter()
                .any(|n| *n < 4 || *n > 2048 || n % 4 != 0)
            || !(1..=256).contains(&self.channels)
            || !(4..=512).contains(&self.hidden)
            || self.position_frequencies > 8
            || !self.exposure.is_finite()
            || self.exposure <= 0.0
        {
            return Err("field v1 requires 1..8 views, image dimensions 4..2048 divisible by four, channels 1..256, hidden 4..512, <=8 frequencies and positive exposure".into());
        }
        Ok(())
    }
    pub fn position_channels(&self) -> usize {
        3 + 6 * self.position_frequencies as usize
    }
}

/// Acquisition bounds, supplied equally at train/inference. Not inferred from a target G-buffer.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bounds {
    pub center: [f32; 3],
    pub radius: f32,
}
impl Bounds {
    pub fn validate(self) -> Result<(), String> {
        if !self.center.iter().all(|x| x.is_finite())
            || !self.radius.is_finite()
            || self.radius <= 0.0
        {
            return Err("invalid acquisition bounds".into());
        }
        Ok(())
    }
    pub fn normalize(self, x: [f32; 3]) -> [f32; 3] {
        std::array::from_fn(|i| (x[i] - self.center[i]) / self.radius)
    }
}

/// Right-handed world basis, positive forward. Pixel centers; image y increases down.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Camera {
    pub origin: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub forward: [f32; 3],
    pub tan_half_fov_y: f32,
}
pub(crate) fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
pub(crate) fn unit(x: [f32; 3]) -> [f32; 3] {
    let length = dot(x, x).sqrt();
    x.map(|v| v / length)
}
impl Camera {
    pub fn validate(self) -> Result<(), String> {
        let basis = [self.right, self.up, self.forward];
        if !self.origin.iter().all(|v| v.is_finite())
            || !self.tan_half_fov_y.is_finite()
            || self.tan_half_fov_y <= 0.0
            || basis
                .iter()
                .any(|v| !v.iter().all(|x| x.is_finite()) || (dot(*v, *v) - 1.0).abs() > 1e-3)
            || dot(self.right, self.up).abs() > 1e-3
            || dot(self.right, self.forward).abs() > 1e-3
            || dot(self.up, self.forward).abs() > 1e-3
        {
            return Err(
                "camera needs finite origin, orthonormal basis and positive FOV tangent".into(),
            );
        }
        // Camera right cross up points backward, as in Blade/OpenGL.
        let r = self.right;
        let u = self.up;
        let cross = [
            r[1] * u[2] - r[2] * u[1],
            r[2] * u[0] - r[0] * u[2],
            r[0] * u[1] - r[1] * u[0],
        ];
        if dot(cross, self.forward) > -0.999 {
            return Err("camera basis has wrong handedness".into());
        }
        Ok(())
    }
    pub fn ray(self, pixel: [f32; 2], extent: [u32; 2]) -> Ray {
        let [w, h] = extent.map(|v| v as f32);
        let x = (2.0 * (pixel[0] + 0.5) / w - 1.0) * self.tan_half_fov_y * w / h;
        let y = (1.0 - 2.0 * (pixel[1] + 0.5) / h) * self.tan_half_fov_y;
        Ray {
            origin: self.origin,
            direction: unit(std::array::from_fn(|i| {
                self.forward[i] + x * self.right[i] + y * self.up[i]
            })),
        }
    }
    pub fn project(self, point: [f32; 3], extent: [u32; 2]) -> Option<[f32; 2]> {
        let d = std::array::from_fn(|i| point[i] - self.origin[i]);
        let z = dot(d, self.forward);
        if !z.is_finite() || z <= 1e-5 {
            return None;
        }
        let [w, h] = extent.map(|v| v as f32);
        Some([
            (dot(d, self.right) / (z * self.tan_half_fov_y * w / h) + 1.0) * 0.5 * w - 0.5,
            (1.0 - dot(d, self.up) / (z * self.tan_half_fov_y)) * 0.5 * h - 0.5,
        ])
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: [f32; 3],
    pub direction: [f32; 3],
}
#[derive(Clone, Copy, Debug)]
pub struct Query {
    pub position: [f32; 3],
    pub direction: [f32; 3],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct View {
    pub camera: Camera,
    pub rgb: Vec<f32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observations {
    pub bounds: Bounds,
    pub views: Vec<View>,
}
impl Observations {
    pub fn validate(&self, config: &Config) -> Result<(), String> {
        config.validate()?;
        self.bounds.validate()?;
        if self.views.len() != config.views {
            return Err("source view count differs from field config".into());
        }
        let n = config.extent[0] as usize * config.extent[1] as usize * 3;
        for view in &self.views {
            view.camera.validate()?;
            if view.rgb.len() != n || view.rgb.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err("expected finite nonnegative scene-linear RGB source images".into());
            }
        }
        Ok(())
    }
}

/// Surface emission labels, not volumetric density. Coordinates are only sample locations.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmissionProbe {
    pub position: [f32; 3],
    pub radiance: [f32; 3],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Emitter {
    pub name: String,
    pub vertices: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
    /// Radiance emitted by the material, in the renderer's linear RGB units; not power.
    pub radiance: [f32; 3],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lighting {
    pub environment: [f32; 3],
    pub emitters: Vec<Emitter>,
    pub probes: Vec<EmissionProbe>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneRecord {
    pub scene_seed: u64,
    pub lighting_seed: Option<u64>,
    pub bounds: Bounds,
    pub lighting: Lighting,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incident: Option<incident::Capture>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<surface::Capture>,
    pub sample: usize,
    pub scene: usize,
    pub camera: Camera,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    pub rgb_space: String,
    pub static_scene: bool,
    pub extent: [u32; 2],
    pub scenes: Vec<SceneRecord>,
    pub records: Vec<ViewRecord>,
}
impl Manifest {
    pub fn validate(&self, records: usize) -> Result<(), String> {
        if ![1, 2, 3].contains(&self.version)
            || self.rgb_space != "scene-linear-renderer-units"
            || !self.static_scene
            || self.extent.contains(&0)
            || self.records.len() != records
            || self.scenes.is_empty()
        {
            return Err(
                "field capture needs static v1/v2/v3 scene-linear metadata and matching record count"
                    .into(),
            );
        }
        for (i, r) in self.records.iter().enumerate() {
            if r.sample != i || r.scene >= self.scenes.len() {
                return Err("invalid scene/sample index".into());
            }
            r.camera.validate()?;
            if let Some(surface) = &r.surface {
                if self.version < 3 {
                    return Err("surface labels require manifest v3".into());
                }
                surface.validate(self.extent)?;
            }
        }
        for s in &self.scenes {
            s.bounds.validate()?;
            if let Some(capture) = &s.incident {
                if self.version < 2 {
                    return Err("incident labels require manifest v2".into());
                }
                capture.validate()?;
            }
            let good_rgb = |v: [f32; 3]| v.iter().all(|v| v.is_finite() && *v >= 0.0);
            if !good_rgb(s.lighting.environment) {
                return Err("invalid environment label".into());
            }
            for p in &s.lighting.probes {
                if !p.position.iter().all(|v| v.is_finite()) || !good_rgb(p.radiance) {
                    return Err("invalid emission probe".into());
                }
            }
            for e in &s.lighting.emitters {
                if !good_rgb(e.radiance)
                    || e.vertices.iter().flatten().any(|v| !v.is_finite())
                    || e.triangles
                        .iter()
                        .flatten()
                        .any(|i| *i as usize >= e.vertices.len())
                {
                    return Err("invalid emitter mesh".into());
                }
            }
        }
        Ok(())
    }
}
