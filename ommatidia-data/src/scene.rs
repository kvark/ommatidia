//! Procedural scenes and camera poses to render them from.
//!
//! Building geometry in code rather than loading glTF keeps the generator
//! self-contained, and more importantly it makes the *variety* of the training
//! set a parameter. A fixed scene teaches the network that scene; randomised
//! material, layout, and viewpoint teach it the estimator's failure modes,
//! which is what actually transfers.

use ommatidia::rng::Rng;

const SPHERE_SEGMENTS: usize = 24;
const SPHERE_RINGS: usize = 12;

fn encode_normal(v: [f32; 3]) -> u32 {
    let quantize = |f: f32| ((f.clamp(-1.0, 1.0) * 127.0 + 0.5) as i8) as u8 as u32;
    quantize(v[0]) | (quantize(v[1]) << 8) | (quantize(v[2]) << 16)
}

/// A UV sphere with normals and tangents, wound counter-clockwise from
/// outside so the ray tracer's flat normals point away from the surface.
fn sphere(center: [f32; 3], radius: f32) -> (Vec<blade_render::Vertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity((SPHERE_SEGMENTS + 1) * (SPHERE_RINGS + 1));
    for ring in 0..=SPHERE_RINGS {
        let theta = std::f32::consts::PI * ring as f32 / SPHERE_RINGS as f32;
        let (sin_theta, cos_theta) = theta.sin_cos();
        for segment in 0..=SPHERE_SEGMENTS {
            let phi = std::f32::consts::TAU * segment as f32 / SPHERE_SEGMENTS as f32;
            let (sin_phi, cos_phi) = phi.sin_cos();
            let normal = [sin_theta * cos_phi, cos_theta, sin_theta * sin_phi];
            vertices.push(blade_render::Vertex {
                position: [
                    center[0] + radius * normal[0],
                    center[1] + radius * normal[1],
                    center[2] + radius * normal[2],
                ],
                bitangent_sign: 1.0,
                tex_coords: [
                    segment as f32 / SPHERE_SEGMENTS as f32,
                    ring as f32 / SPHERE_RINGS as f32,
                ],
                normal: encode_normal(normal),
                tangent: encode_normal([-sin_phi, 0.0, cos_phi]),
            });
        }
    }

    let mut indices = Vec::with_capacity(SPHERE_SEGMENTS * SPHERE_RINGS * 6);
    let stride = (SPHERE_SEGMENTS + 1) as u32;
    for ring in 0..SPHERE_RINGS as u32 {
        for segment in 0..SPHERE_SEGMENTS as u32 {
            let base = ring * stride + segment;
            indices.extend_from_slice(&[base, base + 1, base + stride]);
            indices.extend_from_slice(&[base + 1, base + stride + 1, base + stride]);
        }
    }
    (vertices, indices)
}

/// A horizontal quad centred on the origin, facing up.
fn ground(half_extent: f32, y: f32) -> (Vec<blade_render::Vertex>, Vec<u32>) {
    let normal = encode_normal([0.0, 1.0, 0.0]);
    let tangent = encode_normal([1.0, 0.0, 0.0]);
    let corners = [
        [-half_extent, y, -half_extent],
        [half_extent, y, -half_extent],
        [half_extent, y, half_extent],
        [-half_extent, y, half_extent],
    ];
    let vertices = corners
        .iter()
        .enumerate()
        .map(|(i, &position)| blade_render::Vertex {
            position,
            bitangent_sign: 1.0,
            tex_coords: [(i & 1) as f32, (i >> 1) as f32],
            normal,
            tangent,
        })
        .collect();
    // Counter-clockwise seen from above.
    (vertices, vec![0, 3, 2, 0, 2, 1])
}

/// How much variety to put into a generated scene.
pub struct SceneConfig {
    /// Shaded spheres scattered over the ground.
    pub sphere_count: usize,
    /// Emissive spheres acting as local lights.
    pub light_count: usize,
    /// Radius of the disc the spheres are scattered over.
    pub spread: f32,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            sphere_count: 12,
            light_count: 3,
            spread: 4.0,
        }
    }
}

/// Build one scene. The same `seed` always produces the same geometry.
pub fn build(config: &SceneConfig, seed: u64) -> Vec<blade_render::ProceduralGeometry> {
    let mut rng = Rng::new(seed);
    let mut geometries = Vec::with_capacity(config.sphere_count + config.light_count + 1);

    let (vertices, indices) = ground(config.spread * 3.0, 0.0);
    geometries.push(blade_render::ProceduralGeometry {
        name: "ground".into(),
        vertices,
        indices,
        // A mid-grey dielectric floor, which bounces light without dominating.
        base_color_factor: [0.5, 0.5, 0.5, 1.0],
        metalness: 0.0,
        roughness: 0.8,
        emissive_factor: [0.0; 3],
    });

    for i in 0..config.sphere_count {
        let angle = std::f32::consts::TAU * rng.uniform();
        let distance = config.spread * rng.uniform().sqrt();
        let radius = 0.3 + 0.5 * rng.uniform();
        let center = [distance * angle.cos(), radius, distance * angle.sin()];
        let (vertices, indices) = sphere(center, radius);
        geometries.push(blade_render::ProceduralGeometry {
            name: format!("sphere{i}"),
            vertices,
            indices,
            base_color_factor: [
                0.2 + 0.7 * rng.uniform(),
                0.2 + 0.7 * rng.uniform(),
                0.2 + 0.7 * rng.uniform(),
                1.0,
            ],
            // Materials cluster at the ends of the metalness range in
            // practice, and the mixed middle is not physical anyway.
            metalness: if rng.uniform() < 0.4 { 1.0 } else { 0.0 },
            // Kept off the mirror end: a near-zero roughness lobe is one the
            // real-time estimator cannot resolve at all, so those pixels teach
            // the network noise rather than structure.
            roughness: 0.15 + 0.75 * rng.uniform(),
            emissive_factor: [0.0; 3],
        });
    }

    for i in 0..config.light_count {
        let angle = std::f32::consts::TAU * rng.uniform();
        let distance = config.spread * (0.6 + 0.5 * rng.uniform());
        let radius = 0.2 + 0.3 * rng.uniform();
        let center = [
            distance * angle.cos(),
            2.0 + 2.0 * rng.uniform(),
            distance * angle.sin(),
        ];
        let (vertices, indices) = sphere(center, radius);
        // Bright enough to matter against the ambient dummy environment.
        let strength = 6.0 + 10.0 * rng.uniform();
        geometries.push(blade_render::ProceduralGeometry {
            name: format!("light{i}"),
            vertices,
            indices,
            base_color_factor: [0.0, 0.0, 0.0, 1.0],
            metalness: 0.0,
            roughness: 1.0,
            emissive_factor: [
                strength * (0.6 + 0.4 * rng.uniform()),
                strength * (0.6 + 0.4 * rng.uniform()),
                strength * (0.6 + 0.4 * rng.uniform()),
            ],
        });
    }

    geometries
}

/// A camera somewhere on a hemisphere around the scene, aimed at a point near
/// the origin.
pub fn camera(config: &SceneConfig, rng: &mut Rng) -> blade_render::Camera {
    let fov_y = 0.6 + 0.4 * rng.uniform();
    let azimuth = std::f32::consts::TAU * rng.uniform();
    // Kept off the horizon and off straight-down: both degenerate framings.
    let elevation = 0.15 + 0.5 * rng.uniform();
    let distance = config.spread * (1.4 + 0.8 * rng.uniform());

    let position = [
        distance * elevation.cos() * azimuth.cos(),
        distance * elevation.sin() + 0.5,
        distance * elevation.cos() * azimuth.sin(),
    ];
    // Aim a little off centre so the composition is not always identical.
    let target = [
        0.6 * config.spread * (rng.uniform() - 0.5),
        0.5 + rng.uniform(),
        0.6 * config.spread * (rng.uniform() - 0.5),
    ];

    blade_render::Camera {
        pos: position.into(),
        rot: look_at(position, target),
        fov_y,
        depth: 200.0,
        // Blade derives the horizontal field of view from the target extent
        // when this is None. The low and high resolution renders share an
        // aspect ratio, so they stay framed identically.
        fov: None,
    }
}

/// Rotation taking the camera's -Z axis onto `target - position`, with no roll.
///
/// Blade's convention is right-handed with X right, Y up, and Z towards the
/// camera, so the view direction is the negative Z axis of the rotation.
fn look_at(position: [f32; 3], target: [f32; 3]) -> mint::Quaternion<f32> {
    let forward = normalize([
        target[0] - position[0],
        target[1] - position[1],
        target[2] - position[2],
    ]);
    // The camera looks down -Z, so +Z is the reverse of the view direction.
    let z = [-forward[0], -forward[1], -forward[2]];
    let world_up = if z[1].abs() > 0.999 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let x = normalize(cross(world_up, z));
    let y = cross(z, x);
    matrix_to_quaternion([x, y, z])
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / length, v[1] / length, v[2] / length]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Shepperd's method: pick the branch with the largest denominator so the
/// square root never lands near zero.
fn matrix_to_quaternion(columns: [[f32; 3]; 3]) -> mint::Quaternion<f32> {
    let [x, y, z] = columns;
    let m = |row: usize, column: usize| columns[column][row];
    let trace = x[0] + y[1] + z[2];

    let (w, vx, vy, vz) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        (
            0.25 * s,
            (m(2, 1) - m(1, 2)) / s,
            (m(0, 2) - m(2, 0)) / s,
            (m(1, 0) - m(0, 1)) / s,
        )
    } else if x[0] > y[1] && x[0] > z[2] {
        let s = (1.0 + m(0, 0) - m(1, 1) - m(2, 2)).sqrt() * 2.0;
        (
            (m(2, 1) - m(1, 2)) / s,
            0.25 * s,
            (m(0, 1) + m(1, 0)) / s,
            (m(0, 2) + m(2, 0)) / s,
        )
    } else if y[1] > z[2] {
        let s = (1.0 + m(1, 1) - m(0, 0) - m(2, 2)).sqrt() * 2.0;
        (
            (m(0, 2) - m(2, 0)) / s,
            (m(0, 1) + m(1, 0)) / s,
            0.25 * s,
            (m(1, 2) + m(2, 1)) / s,
        )
    } else {
        let s = (1.0 + m(2, 2) - m(0, 0) - m(1, 1)).sqrt() * 2.0;
        (
            (m(1, 0) - m(0, 1)) / s,
            (m(0, 2) + m(2, 0)) / s,
            (m(1, 2) + m(2, 1)) / s,
            0.25 * s,
        )
    };

    mint::Quaternion {
        v: [vx, vy, vz].into(),
        s: w,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rotate(q: mint::Quaternion<f32>, v: [f32; 3]) -> [f32; 3] {
        let u = [q.v.x, q.v.y, q.v.z];
        let s = q.s;
        let dot_uv = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        let dot_uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
        let cross_uv = cross(u, v);
        let mut out = [0.0; 3];
        for i in 0..3 {
            out[i] = 2.0 * dot_uv * u[i] + (s * s - dot_uu) * v[i] + 2.0 * s * cross_uv[i];
        }
        out
    }

    #[test]
    fn look_at_points_negative_z_at_the_target() {
        let cases = [
            ([0.0f32, 0.0, 5.0], [0.0f32, 0.0, 0.0]),
            ([3.0, 4.0, 5.0], [0.0, 1.0, 0.0]),
            ([-2.0, 6.0, 1.0], [1.0, 0.0, -2.0]),
        ];
        for (position, target) in cases {
            let q = look_at(position, target);
            let view = rotate(q, [0.0, 0.0, -1.0]);
            let expected = normalize([
                target[0] - position[0],
                target[1] - position[1],
                target[2] - position[2],
            ]);
            for i in 0..3 {
                assert!(
                    (view[i] - expected[i]).abs() < 1e-4,
                    "from {position:?} to {target:?}: got {view:?}, want {expected:?}"
                );
            }
        }
    }

    #[test]
    fn look_at_keeps_the_horizon_level() {
        // With no roll, the camera's right axis stays in the world XZ plane.
        let q = look_at([3.0, 4.0, 5.0], [0.0, 0.0, 0.0]);
        let right = rotate(q, [1.0, 0.0, 0.0]);
        assert!(right[1].abs() < 1e-4, "camera rolled: right = {right:?}");
    }

    #[test]
    fn look_at_survives_looking_straight_down() {
        let q = look_at([0.0, 10.0, 0.0], [0.0, 0.0, 0.0]);
        let view = rotate(q, [0.0, 0.0, -1.0]);
        assert!(view.iter().all(|v| v.is_finite()), "{view:?}");
        assert!((view[1] + 1.0).abs() < 1e-3, "should look down: {view:?}");
    }

    #[test]
    fn scenes_are_deterministic_and_varied() {
        let config = SceneConfig::default();
        let a = build(&config, 1);
        let b = build(&config, 1);
        let c = build(&config, 2);
        assert_eq!(a.len(), config.sphere_count + config.light_count + 1);
        assert_eq!(a[1].roughness, b[1].roughness, "same seed, same scene");
        assert_ne!(a[1].roughness, c[1].roughness, "seeds should differ");
    }

    #[test]
    fn spheres_rest_on_the_ground() {
        let config = SceneConfig::default();
        for geometry in build(&config, 5)
            .iter()
            .filter(|g| g.name.starts_with("sphere"))
        {
            let lowest = geometry
                .vertices
                .iter()
                .map(|v| v.position[1])
                .fold(f32::INFINITY, f32::min);
            assert!(lowest > -0.01, "{} dips to {lowest}", geometry.name);
        }
    }
}
