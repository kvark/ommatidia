//! Static multiview RGB captures with target-only light-source annotations.
use crate::scene;
use ommatidia::{field, rng::Rng};

pub fn relight(surfaces: &mut [scene::Surface], seed: Option<u64>) {
    let Some(seed) = seed else { return };
    let mut rng = Rng::new(seed);
    for s in surfaces {
        if s.geometry.emissive_factor.iter().any(|v| *v > 0.0) {
            for v in &mut s.geometry.emissive_factor {
                *v *= 0.25 + 1.75 * rng.uniform();
            }
        }
    }
}
pub fn labels(
    surfaces: &[scene::Surface],
    seed: u64,
    lighting_seed: Option<u64>,
    spread: f32,
) -> field::SceneRecord {
    let mut emitters = Vec::new();
    let mut probes = Vec::new();
    for s in surfaces {
        let g = &s.geometry;
        let luminous = g.emissive_factor.iter().any(|v| *v > 0.0);
        if luminous {
            emitters.push(field::Emitter {
                name: g.name.clone(),
                vertices: g.vertices.iter().map(|v| v.position).collect(),
                triangles: g
                    .indices
                    .chunks_exact(3)
                    .map(|i| [i[0], i[1], i[2]])
                    .collect(),
                radiance: g.emissive_factor,
            });
        }
        // Surface centroids, not the emitter bounding volume. Include known dark surfaces.
        let triangles = g.indices.len() / 3;
        let stride = (triangles / 64).max(1);
        for t in (0..triangles).step_by(stride) {
            let points = g.indices[3 * t..3 * t + 3]
                .iter()
                .map(|i| g.vertices[*i as usize].position)
                .collect::<Vec<_>>();
            let position =
                std::array::from_fn(|c| (points[0][c] + points[1][c] + points[2][c]) / 3.0);
            probes.push(field::EmissionProbe {
                position,
                radiance: g.emissive_factor,
            });
        }
    }
    field::SceneRecord {
        scene_seed: seed,
        lighting_seed,
        bounds: field::Bounds {
            center: [0.0, 1.0, 0.0],
            radius: spread * 3.0,
        },
        lighting: field::Lighting {
            environment: [1.0; 3],
            emitters,
            probes,
        },
    }
}
pub fn camera(camera: blade_render::Camera) -> field::Camera {
    assert!(
        camera.fov.is_none(),
        "field capture uses centered, unjittered cameras"
    );
    let q = [camera.rot.v.x, camera.rot.v.y, camera.rot.v.z];
    let s = camera.rot.s;
    let rotate = |v: [f32; 3]| {
        let cross = |a: [f32; 3], b: [f32; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let a = cross(q, v).map(|v| 2.0 * v);
        let b = cross(q, a);
        std::array::from_fn(|i| v[i] + s * a[i] + b[i])
    };
    field::Camera {
        origin: [camera.pos.x, camera.pos.y, camera.pos.z],
        right: rotate([1.0, 0.0, 0.0]),
        up: rotate([0.0, 1.0, 0.0]),
        forward: rotate([0.0, 0.0, -1.0]),
        tan_half_fov_y: (0.5 * camera.fov_y).tan(),
    }
}
pub fn orbit(config: &scene::SceneConfig, frame: usize, count: usize) -> blade_render::Camera {
    let t = (frame as f32 + 0.5) / count as f32;
    let azimuth = if config.canopy {
        std::f32::consts::FRAC_PI_2 + t * std::f32::consts::PI
    } else {
        t * std::f32::consts::TAU
    };
    let position = [
        config.spread * 1.9 * azimuth.cos(),
        config.spread * 0.9,
        config.spread * 1.9 * azimuth.sin(),
    ];
    blade_render::Camera {
        pos: position.into(),
        rot: scene::look_at(position, [0.0, 1.0, 0.0]),
        fov_y: 0.8,
        depth: 200.0,
        fov: None,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relighting_preserves_geometry_and_labels_actual_emitters() {
        let config = scene::SceneConfig::default();
        let mut a = scene::build(&config, 3);
        let b = scene::build(&config, 3);
        relight(&mut a, Some(7));
        let mut changed = false;
        for (a, b) in a.iter().zip(&b) {
            assert_eq!(a.geometry.indices, b.geometry.indices);
            for (a, b) in a.geometry.vertices.iter().zip(&b.geometry.vertices) {
                assert_eq!(a.position, b.position);
            }
            changed |= a.geometry.emissive_factor != b.geometry.emissive_factor;
        }
        assert!(changed);
        let labels = labels(&a, 3, Some(7), config.spread);
        assert_eq!(labels.lighting.emitters.len(), config.light_count);
        assert!(
            labels
                .lighting
                .probes
                .iter()
                .any(|p| p.radiance == [0.0; 3])
        );
        for i in 0..6 {
            camera(orbit(&config, i, 6)).validate().unwrap();
        }
    }
}
