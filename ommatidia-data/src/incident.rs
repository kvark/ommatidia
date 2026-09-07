//! Directional incident labels from Blade's own canonical integrator, not a
//! second material/visibility approximation. A 1x1, nonjittered camera is a ray.
use crate::{Harness, gbuffer, make_renderer, radiance, render, scene};
use blade_graphics as gpu;
use ommatidia::{
    field::{
        Ray,
        incident::{Capture, Moments, Probe, Proposal},
    },
    rng::Rng,
};

const EPSILON: f32 = 0.01;
const DISTANCE: f32 = 200.0;
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn unit(a: [f32; 3]) -> [f32; 3] {
    let n = dot(a, a).sqrt();
    a.map(|x| x / n)
}

/// Surface/triangle-uniform origins with hemisphere and light-aimed strata.
/// This is a pointwise regression distribution, NOT an irradiance quadrature.
pub fn queries(surfaces: &[scene::Surface], count: usize, seed: u64) -> Vec<(Ray, Proposal)> {
    let triangles = |s: &scene::Surface| {
        s.geometry
            .indices
            .chunks_exact(3)
            .filter_map(|ix| {
                let p: [[f32; 3]; 3] =
                    std::array::from_fn(|i| s.geometry.vertices[ix[i] as usize].position);
                let normal = cross(sub(p[1], p[0]), sub(p[2], p[0]));
                (dot(normal, normal) > 1e-12).then(|| (p, unit(normal)))
            })
            .collect::<Vec<_>>()
    };
    let receiving: Vec<_> = surfaces
        .iter()
        .filter(|s| s.geometry.emissive_factor == [0.0; 3])
        .map(triangles)
        .filter(|t| !t.is_empty())
        .collect();
    let lights: Vec<_> = surfaces
        .iter()
        .filter(|s| s.geometry.emissive_factor.iter().any(|v| *v > 0.0))
        .flat_map(triangles)
        .collect();
    assert!(
        !receiving.is_empty(),
        "incident probes require receiving surfaces"
    );
    let point = |p: [[f32; 3]; 3], rng: &mut Rng| {
        let a = rng.uniform().sqrt();
        let b = rng.uniform();
        std::array::from_fn(|i| (1.0 - a) * p[0][i] + a * (1.0 - b) * p[1][i] + a * b * p[2][i])
    };
    let mut rng = Rng::new(seed ^ 0xA718_9B37_502D_6F01);
    (0..count)
        .map(|i| {
            let surface = &receiving[rng.below(receiving.len() as u32) as usize];
            let (p, n) = surface[rng.below(surface.len() as u32) as usize];
            let p = point(p, &mut rng);
            let origin = std::array::from_fn(|i| p[i] + EPSILON * n[i]);
            if i % 2 == 1 && !lights.is_empty() {
                for _ in 0..8 {
                    let light = lights[rng.below(lights.len() as u32) as usize].0;
                    let delta = sub(point(light, &mut rng), origin);
                    if dot(delta, delta) > EPSILON * EPSILON && dot(delta, n) > 0.0 {
                        return (
                            Ray {
                                origin,
                                direction: unit(delta),
                            },
                            Proposal::EmitterAimed,
                        );
                    }
                }
            }
            let z = 2.0 * rng.uniform() - 1.0;
            let phi = std::f32::consts::TAU * rng.uniform();
            let r = (1.0 - z * z).max(0.0).sqrt();
            let direction = [r * phi.cos(), z, r * phi.sin()];
            let direction = if dot(direction, n) < 0.0 {
                direction.map(|v| -v)
            } else {
                direction
            };
            (Ray { origin, direction }, Proposal::UniformHemisphere)
        })
        .collect()
}

pub struct Tracer {
    renderer: blade_render::RayTracer,
    target: render::Target,
    surface: gbuffer::Probe,
    lobes: radiance::Probe,
}
impl Tracer {
    /// Caller submits setup, as for the ordinary image renderers.
    pub fn new(harness: &Harness, encoder: &mut gpu::CommandEncoder) -> Self {
        let size = gpu::Extent {
            width: 1,
            height: 1,
            depth: 1,
        };
        Self {
            renderer: make_renderer(harness, encoder, size),
            target: render::Target::new(&harness.context, size),
            surface: gbuffer::Probe::new(&harness.context, size, false),
            lobes: radiance::Probe::new(&harness.context, size),
        }
    }
    pub fn capture(
        &mut self,
        harness: &Harness,
        encoder: &mut gpu::CommandEncoder,
        objects: &mut [blade_render::Object],
        rays: &[(Ray, Proposal)],
        max_bounces: u32,
        batches: u32,
    ) -> Result<Capture, String> {
        let mut probes = Vec::with_capacity(rays.len());
        for &(ray, proposal) in rays {
            let end = std::array::from_fn(|i| ray.origin[i] + ray.direction[i]);
            let camera = blade_render::Camera {
                pos: ray.origin.into(),
                rot: scene::look_at(ray.origin, end),
                fov_y: 1.0,
                depth: DISTANCE,
                fov: None,
            };
            let actual = crate::field_capture::camera(camera).ray([0.0, 0.0], [1, 1]);
            if dot(actual.direction, ray.direction) < 0.999999 {
                return Err("probe camera does not match requested ray".into());
            }
            let mut direct = Moments::default();
            let mut indirect = Moments::default();
            let mut total = Moments::default();
            for _ in 0..batches {
                // Four fresh paths per reset batch, and a dedicated renderer:
                // probe capture never advances the image renderers' RNG/state.
                let frame = render::capture(
                    &mut self.renderer,
                    &self.target,
                    &harness.context,
                    encoder,
                    &harness.asset_hub,
                    objects,
                    &camera,
                    render::Pass::Canonical {
                        frames: 1,
                        max_bounces,
                        sample_offset: 0,
                    },
                    false,
                    Some(&self.surface),
                    Some(&self.lobes),
                );
                let rgb: [f32; 3] = frame.color[..3].try_into().unwrap();
                let sky = frame.gbuffer.as_ref().unwrap()[0] >= 1e6;
                let emission: [f32; 3] = frame.radiance.as_ref().unwrap()[6..9].try_into().unwrap();
                let d = if sky { rgb } else { emission };
                let i = std::array::from_fn(|c| rgb[c] - d[c]);
                if i.iter()
                    .zip(&rgb)
                    .any(|(i, t)| *i < -1e-5 * t.abs().max(1.0))
                {
                    return Err("canonical total is below primary emission".into());
                }
                direct.push(d)?;
                indirect.push(i.map(|v| v.max(0.0)))?;
                total.push(rgb)?;
            }
            probes.push(Probe {
                origin: ray.origin,
                direction: actual.direction,
                proposal,
                direct: direct.finish()?,
                indirect: indirect.finish()?,
                total_variance_of_mean: total.finish()?.variance_of_mean,
            });
        }
        let result = Capture {
            version: 1,
            integrator: "blade-canonical-point-ray-v1".into(),
            max_bounces,
            batches,
            paths_per_batch: 4,
            ray_epsilon: EPSILON,
            ray_distance: DISTANCE,
            radiance_ceiling: 1e6,
            probes,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn destroy(mut self, context: &gpu::Context) {
        self.target.destroy(context);
        self.surface.destroy(context);
        self.lobes.destroy(context);
        self.renderer.destroy(context);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn quad(y: f32, radiance: [f32; 3], color: f32) -> scene::Surface {
        blade_render::ProceduralGeometry {
            name: format!("quad-{y}"),
            vertices: [
                [-2.0, y, -2.0],
                [-2.0, y, 2.0],
                [2.0, y, 2.0],
                [2.0, y, -2.0],
            ]
            .map(|position| blade_render::Vertex {
                position,
                normal: 127 << 8,
                tangent: 127,
                bitangent_sign: 1.0,
                ..Default::default()
            })
            .to_vec(),
            indices: vec![0, 1, 2, 0, 2, 3],
            base_color_factor: [color, color, color, 1.0],
            emissive_factor: radiance,
            metalness: 0.0,
            roughness: 1.0,
        }
        .into()
    }
    #[test]
    fn probes_are_deterministic_and_do_not_change_with_emitter_intensity() {
        let cfg = scene::SceneConfig::default();
        let mut scene = scene::build(&cfg, 7);
        let a = queries(&scene, 32, 7);
        crate::field_capture::relight(&mut scene, Some(9));
        let b = queries(&scene, 32, 7);
        assert!(
            a.iter().zip(b).all(|((a, p), (b, q))| a.origin == b.origin
                && a.direction == b.direction
                && *p == q)
        );
        assert!(a.iter().any(|(_, p)| *p == Proposal::EmitterAimed));
        assert!(
            a.iter()
                .all(|(r, _)| (dot(r.direction, r.direction) - 1.0).abs() < 1e-5)
        );
    }
    #[test]
    #[ignore = "requires Vulkan ray queries"]
    fn incident_capture_distinguishes_sky_emission_occlusion_and_bounce() {
        let harness = Harness::new(None, None);
        let context = std::sync::Arc::clone(&harness.context);
        let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "incident-test",
            buffer_count: 2,
            manual_barriers: false,
        });
        encoder.start();
        let mut tracer = Tracer::new(&harness, &mut encoder);
        let sync = context.submit(&mut encoder);
        assert!(context.wait_for(&sync, 30000).unwrap());
        let palette = crate::TexturePalette::bake(&harness, 0);
        let light = [4.0, 2.0, 1.0];
        let mut source = vec![blade_render::Object::from(palette.build_model(
            &harness,
            "incident-source",
            vec![quad(0.0, light, 0.0)],
        ))];
        let rays = [
            (
                Ray {
                    origin: [0.0, 1.0, 0.0],
                    direction: [0.0, -1.0, 0.0],
                },
                Proposal::EmitterAimed,
            ),
            (
                Ray {
                    origin: [0.0, 1.0, 0.0],
                    direction: [0.0, 1.0, 0.0],
                },
                Proposal::UniformHemisphere,
            ),
            (
                Ray {
                    origin: [0.0, 3.0, 0.0],
                    direction: [0.0, -1.0, 0.0],
                },
                Proposal::EmitterAimed,
            ),
        ];
        // The field's first-surface labels come from the same GPU ray query,
        // not an independent geometry approximation or camera-Z conversion.
        for (ray, _) in &rays {
            let end = std::array::from_fn(|i| ray.origin[i] + ray.direction[i]);
            let camera = blade_render::Camera {
                pos: ray.origin.into(),
                rot: scene::look_at(ray.origin, end),
                fov_y: 1.0,
                depth: 200.0,
                fov: None,
            };
            let frame = render::capture(
                &mut tracer.renderer,
                &tracer.target,
                &context,
                &mut encoder,
                &harness.asset_hub,
                &mut source,
                &camera,
                render::Pass::Canonical {
                    frames: 1,
                    max_bounces: 2,
                    sample_offset: 0,
                },
                false,
                Some(&tracer.surface),
                Some(&tracer.lobes),
            );
            let labels = crate::field_capture::surface_labels(&frame, 1, 200.0).unwrap();
            if ray.direction[1] < 0.0 {
                assert!((labels.distance[0].unwrap() - ray.origin[1]).abs() < 1e-5);
                assert_eq!(labels.emission[0], light);
            } else {
                assert_eq!(labels.distance[0], None);
                assert_eq!(labels.emission[0], [0.0; 3]);
            }
        }
        let capture = tracer
            .capture(&harness, &mut encoder, &mut source, &rays, 2, 4)
            .unwrap();
        assert_eq!(capture.probes[0].direct.mean, light);
        assert_eq!(
            capture.probes[2].direct.mean, light,
            "radiance incorrectly attenuated by distance"
        );
        assert_eq!(capture.probes[1].direct.mean, [1.0; 3]);
        assert_eq!(capture.probes[1].indirect.mean, [0.0; 3]);
        assert_eq!(capture.probes[1].total_variance_of_mean, [0.0; 3]);
        let mut blocked = vec![blade_render::Object::from(palette.build_model(
            &harness,
            "incident-blocked",
            vec![quad(0.0, light, 0.0), quad(0.5, [0.0; 3], 0.5)],
        ))];
        let capture = tracer
            .capture(&harness, &mut encoder, &mut blocked, &rays[..1], 2, 16)
            .unwrap();
        assert_eq!(
            capture.probes[0].direct.mean, [0.0; 3],
            "emitter leaks through opaque blocker"
        );
        assert!(
            capture.probes[0].indirect.mean.iter().all(|v| *v > 0.05),
            "reflected illumination missing"
        );
        println!(
            "incident capture: sky/emission/occlusion/distance contracts pass; reflected={:?}",
            capture.probes[0].indirect.mean
        );
        tracer.destroy(&context);
        context.destroy_command_encoder(&mut encoder);
        harness.destroy();
    }
}
