//! Procedural scenes and camera poses to render them from.
//!
//! Building geometry in code rather than loading glTF keeps the generator
//! self-contained, and more importantly it makes the *variety* of the training
//! set a parameter. A fixed scene teaches the network that scene; randomised
//! material, layout, and viewpoint teach it the estimator's failure modes,
//! which is what actually transfers.

use crate::texture;
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

/// A flat rectangle at `center`, spanning `center ± u ± v`.
///
/// The normal is `u × v`, so the caller picks which way the surface faces by
/// the order it passes the two half-extents. Wound counter-clockwise seen from
/// that side, matching the spheres and boxes.
fn rect(center: [f32; 3], u: [f32; 3], v: [f32; 3]) -> (Vec<blade_render::Vertex>, Vec<u32>) {
    // One texture repeat per world unit, so a texture on a large surface has
    // the same feature size as one on a small surface. Without this a wall and
    // a floor tile carry patterns an order of magnitude apart.
    let repeats = |w: [f32; 3]| (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt().max(1.0);
    tiled_rect(center, u, v, [repeats(u), repeats(v)])
}

fn tiled_rect(
    center: [f32; 3],
    u: [f32; 3],
    v: [f32; 3],
    repeats: [f32; 2],
) -> (Vec<blade_render::Vertex>, Vec<u32>) {
    let face = normalize(cross(u, v));
    let normal = encode_normal(face);
    let tangent = encode_normal(normalize(u));
    let corners = [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
    let vertices = corners
        .iter()
        .map(|&(su, sv)| blade_render::Vertex {
            position: [
                center[0] + su * u[0] + sv * v[0],
                center[1] + su * u[1] + sv * v[1],
                center[2] + su * u[2] + sv * v[2],
            ],
            bitangent_sign: 1.0,
            tex_coords: [(su * 0.5 + 0.5) * repeats[0], (sv * 0.5 + 0.5) * repeats[1]],
            normal,
            tangent,
        })
        .collect();
    (vertices, vec![0, 1, 2, 0, 2, 3])
}

/// A horizontal quad centred on the origin, facing up.
fn ground(half_extent: f32, y: f32) -> (Vec<blade_render::Vertex>, Vec<u32>) {
    rect(
        [0.0, y, 0.0],
        [half_extent, 0.0, 0.0],
        [0.0, 0.0, -half_extent],
    )
}

/// An axis-aligned box, with a rotation about the vertical axis.
///
/// Worth having alongside the spheres because it is the case the upscaler
/// finds hardest and the spheres never present: a straight silhouette at an
/// arbitrary angle, which is exactly where a spatial upscaler produces
/// staircase artifacts, and a hard normal discontinuity at every edge.
fn box_shape(center: [f32; 3], half: [f32; 3], yaw: f32) -> (Vec<blade_render::Vertex>, Vec<u32>) {
    let (sin, cos) = yaw.sin_cos();
    let rotate = |v: [f32; 3]| [v[0] * cos - v[2] * sin, v[1], v[0] * sin + v[2] * cos];

    // Each face gets its own four vertices, so the normals stay hard.
    let faces: [([f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]),
        ([0.0, 0.0, -1.0], [1.0, 0.0, 0.0]),
    ];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (normal, tangent) in faces {
        let bitangent = cross(normal, tangent);
        let base = vertices.len() as u32;
        for (u, v) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let local = [
                (normal[0] + u * tangent[0] + v * bitangent[0]) * half[0],
                (normal[1] + u * tangent[1] + v * bitangent[1]) * half[1],
                (normal[2] + u * tangent[2] + v * bitangent[2]) * half[2],
            ];
            let world = rotate(local);
            vertices.push(blade_render::Vertex {
                position: [
                    center[0] + world[0],
                    center[1] + world[1],
                    center[2] + world[2],
                ],
                bitangent_sign: 1.0,
                tex_coords: [(u + 1.0) * 0.5, (v + 1.0) * 0.5],
                normal: encode_normal(rotate(normal)),
                tangent: encode_normal(rotate(tangent)),
            });
        }
        // Counter-clockwise seen from outside, matching the spheres.
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (vertices, indices)
}

/// One geometry and the texture, if any, its material should sample.
///
/// `ProceduralGeometry` carries only factors, so the texture is attached to the
/// model's material after `create_model` has built it. Pairing the choice with
/// the geometry here is what lets the generator move geometries between models
/// — splitting the movers out of the static one — without losing track of which
/// texture belonged to which surface.
pub struct Surface {
    pub geometry: blade_render::ProceduralGeometry,
    pub texture: Option<texture::Kind>,
}

impl From<blade_render::ProceduralGeometry> for Surface {
    fn from(geometry: blade_render::ProceduralGeometry) -> Self {
        Self {
            geometry,
            texture: None,
        }
    }
}

/// How much variety to put into a generated scene.
pub struct SceneConfig {
    /// Shaded spheres scattered over the ground.
    pub sphere_count: usize,
    /// Shaded boxes scattered over the ground.
    pub box_count: usize,
    /// Emissive spheres acting as local lights.
    pub light_count: usize,
    /// Radius of the disc the objects are scattered over.
    pub spread: f32,
    /// Put a canopy and two walls over part of the scene, so some of it is in
    /// shadow.
    ///
    ///
    /// Blade's fallback environment is a white 1x1 texture, which means an open
    /// scene is lit by a uniform furnace of radiance one from every direction.
    /// Nothing can be in shadow under that. Measured on the 4-spp validation
    /// set, none of its pixels fall below a displayed luminance of 0.10 and
    /// 87.8% of them sit inside a single quarter of the range — so the metrics
    /// have never been asked about the region where noise is most visible.
    ///
    /// Sealing the scene into a room is the obvious fix and the wrong one. The
    /// environment is the only light the estimator importance-samples, so
    /// walling it off leaves three small emissive spheres to be found by chance
    /// alone: a 4-spp input that is black with fireflies, and a 1024-frame
    /// reference that is still visibly noisy. Measured, bilinear upsampling of
    /// that input scores 8.5 dB, against 26.5 dB on the open scenes. Covering
    /// part of the scene instead keeps the sky as the light source and the
    /// sampling well conditioned, while giving the frame somewhere dark.
    pub canopy: bool,
    /// Subdivide the central ground into this many patches per axis, each with
    /// its own albedo.
    ///
    /// `ProceduralGeometry` carries a colour factor and no texture, so the only
    /// way to give a surface detail finer than an object is to make it out of
    /// more objects. Without it, albedo is constant across every surface, which
    /// is why demodulating by it — one of the larger wins available in a real
    /// renderer — measures as doing exactly nothing here.
    pub ground_patches: usize,
    /// Give surfaces a procedural base-colour texture.
    ///
    /// See `texture` for why this matters more than it sounds: without it every
    /// surface is one flat colour, so the ground truth inside an object is a
    /// smooth function the G-buffer already segments, and there is very little
    /// for any reconstruction to recover that a bilateral filter does not.
    pub textures: bool,
    /// Give some surfaces a tight specular lobe.
    ///
    /// The existing roughness floor of 0.15 was chosen to keep the real-time
    /// estimator's variance down, which is the right call for an estimator and
    /// the wrong one for a reference set. A sharp highlight is small, bright,
    /// and destroyed by exactly the over-smoothing that costs nothing in PSNR,
    /// so a set without any cannot see that failure.
    pub gloss: bool,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            sphere_count: 9,
            box_count: 5,
            light_count: 3,
            spread: 4.0,
            canopy: false,
            ground_patches: 0,
            textures: false,
            gloss: false,
        }
    }
}

/// Build one scene. The same `seed` always produces the same geometry.
pub fn build(config: &SceneConfig, seed: u64) -> Vec<Surface> {
    let mut rng = Rng::new(seed);
    let mut geometries =
        Vec::with_capacity(config.sphere_count + config.box_count + config.light_count + 1);
    // Drawn per surface rather than per scene, so one frame contains several
    // patterns and the network cannot learn the scene's texture as a constant.
    let pick_texture = |rng: &mut Rng, chance: f32| {
        (config.textures && rng.uniform() < chance).then(|| {
            texture::KINDS
                [(rng.uniform() * texture::KINDS.len() as f32) as usize % texture::KINDS.len()]
        })
    };
    // Most surfaces stay rough. A tight lobe is the interesting case precisely
    // because it is the rare one, and making it common would make the sparse
    // input mostly variance.
    let pick_roughness = |rng: &mut Rng| {
        if config.gloss && rng.uniform() < 0.3 {
            0.04 + 0.1 * rng.uniform()
        } else {
            0.15 + 0.75 * rng.uniform()
        }
    };

    let ground_tone = 0.25 + 0.45 * rng.uniform();
    let ground_color = |rng: &mut Rng| {
        [
            ground_tone,
            ground_tone * (0.85 + 0.3 * rng.uniform()),
            ground_tone * (0.85 + 0.3 * rng.uniform()),
            1.0,
        ]
    };
    let extent = config.spread * 3.0;
    if config.ground_patches == 0 {
        let (vertices, indices) = ground(extent, 0.0);
        let geometry = blade_render::ProceduralGeometry {
            name: "ground".into(),
            vertices,
            indices,
            // A dielectric floor that bounces light without dominating. Its
            // tone and roughness vary per scene, since a floor of one fixed
            // brightness would let the network learn the backdrop rather than
            // the geometry.
            base_color_factor: ground_color(&mut rng),
            metalness: 0.0,
            // A polished floor reflects the objects and the lights, which is
            // structure no amount of denoising the diffuse term recovers.
            roughness: if config.gloss && rng.uniform() < 0.4 {
                0.05 + 0.1 * rng.uniform()
            } else {
                0.4 + 0.55 * rng.uniform()
            },
            emissive_factor: [0.0; 3],
        };
        let texture = pick_texture(&mut rng, 0.9);
        geometries.push(Surface { geometry, texture });
    } else {
        geometries.extend(
            patched_ground(config, extent, ground_tone, &mut rng)
                .into_iter()
                .map(|geometry| {
                    let texture = pick_texture(&mut rng, 0.6);
                    Surface { geometry, texture }
                }),
        );
    }
    if config.canopy {
        geometries.extend(
            canopy_geometry(config, &mut rng)
                .into_iter()
                .map(|geometry| {
                    let texture = pick_texture(&mut rng, 0.7);
                    Surface { geometry, texture }
                }),
        );
    }

    for i in 0..config.sphere_count {
        let angle = std::f32::consts::TAU * rng.uniform();
        let distance = config.spread * rng.uniform().sqrt();
        let radius = 0.3 + 0.5 * rng.uniform();
        let center = [distance * angle.cos(), radius, distance * angle.sin()];
        let (vertices, indices) = sphere(center, radius);
        let geometry = blade_render::ProceduralGeometry {
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
            roughness: pick_roughness(&mut rng),
            emissive_factor: [0.0; 3],
        };
        let texture = pick_texture(&mut rng, 0.5);
        geometries.push(Surface { geometry, texture });
    }

    for i in 0..config.box_count {
        let angle = std::f32::consts::TAU * rng.uniform();
        let distance = config.spread * rng.uniform().sqrt();
        let half = [
            0.25 + 0.45 * rng.uniform(),
            0.3 + 0.9 * rng.uniform(),
            0.25 + 0.45 * rng.uniform(),
        ];
        // Resting on the ground, turned to an arbitrary angle so the edges do
        // not line up with the pixel grid.
        let center = [distance * angle.cos(), half[1], distance * angle.sin()];
        let (vertices, indices) = box_shape(center, half, std::f32::consts::TAU * rng.uniform());
        let geometry = blade_render::ProceduralGeometry {
            name: format!("box{i}"),
            vertices,
            indices,
            base_color_factor: [
                0.2 + 0.7 * rng.uniform(),
                0.2 + 0.7 * rng.uniform(),
                0.2 + 0.7 * rng.uniform(),
                1.0,
            ],
            metalness: if rng.uniform() < 0.4 { 1.0 } else { 0.0 },
            roughness: pick_roughness(&mut rng),
            emissive_factor: [0.0; 3],
        };
        let texture = pick_texture(&mut rng, 0.5);
        geometries.push(Surface { geometry, texture });
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
        // Bright enough to matter against the ambient dummy environment, and
        // brighter still where a canopy has taken most of that away, since
        // under it these are close to the only light left.
        let shaded = if config.canopy { 3.0 } else { 1.0 };
        let strength = shaded * (6.0 + 10.0 * rng.uniform());
        geometries.push(Surface::from(blade_render::ProceduralGeometry {
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
        }));
    }

    geometries
}

/// A canopy over one side of the scene, with two walls under it.
///
/// Deliberately partial: the sky stays visible over the rest of the frame, so
/// it goes on lighting the scene and being importance-sampled, while anything
/// under the slab sees almost none of it. That is the same arrangement a real
/// frame has when part of it is indoors, and it produces the contact shadows,
/// the falloff, and the dark corners that a uniform environment cannot.
fn canopy_geometry(config: &SceneConfig, rng: &mut Rng) -> Vec<blade_render::ProceduralGeometry> {
    /// A name and the three vectors `rect` needs.
    type Face = (&'static str, [f32; 3], [f32; 3], [f32; 3]);
    let half = config.spread;
    let height = config.spread * 0.85;
    // Offset along +X, so roughly half the scattered objects fall under it.
    let center_x = half * 0.75;
    let faces: [Face; 3] = [
        // Facing down, over the +X half.
        (
            "canopy",
            [center_x, height, 0.0],
            [half, 0.0, 0.0],
            [0.0, 0.0, half * 1.2],
        ),
        // The far wall, closing the shadowed side off from grazing sky.
        (
            "canopy-wall",
            [center_x + half, height * 0.5, 0.0],
            [0.0, 0.0, half * 1.2],
            [0.0, height * 0.5, 0.0],
        ),
        // One side wall, so the shadow has an edge that is not a straight
        // horizontal line across the frame.
        (
            "canopy-side",
            [center_x, height * 0.5, half * 1.2],
            [0.0, height * 0.5, 0.0],
            [half, 0.0, 0.0],
        ),
    ];
    faces
        .into_iter()
        .map(|(name, center, u, v)| {
            let (vertices, indices) = rect(center, u, v);
            // Saturated, because bounce off these surfaces is most of the light
            // reaching what they shade, and colour bleeding is a signal worth
            // having in the data.
            let tone = 0.35 + 0.4 * rng.uniform();
            blade_render::ProceduralGeometry {
                name: name.into(),
                vertices,
                indices,
                base_color_factor: [
                    tone * (0.5 + 0.7 * rng.uniform()),
                    tone * (0.5 + 0.7 * rng.uniform()),
                    tone * (0.5 + 0.7 * rng.uniform()),
                    1.0,
                ],
                metalness: 0.0,
                roughness: 0.6 + 0.35 * rng.uniform(),
                emissive_factor: [0.0; 3],
            }
        })
        .collect()
}

/// The ground as a patchwork, so albedo carries detail finer than an object.
///
/// The centre is divided into `ground_patches` squares per axis and each gets
/// its own colour; four rectangles fill the surround at one colour, since
/// nothing near the frame edge needs the resolution. The patch edges land at
/// arbitrary positions relative to the pixel grid, which is the case a
/// reconstruction filter finds hardest and this data has never contained.
fn patched_ground(
    config: &SceneConfig,
    extent: f32,
    tone: f32,
    rng: &mut Rng,
) -> Vec<blade_render::ProceduralGeometry> {
    let patches = config.ground_patches;
    let inner = (config.spread * 1.5).min(extent);
    let step = 2.0 * inner / patches as f32;
    let mut out = Vec::with_capacity(patches * patches + 4);
    for row in 0..patches {
        for column in 0..patches {
            let center = [
                -inner + step * (column as f32 + 0.5),
                0.0,
                -inner + step * (row as f32 + 0.5),
            ];
            let (vertices, indices) = rect(center, [step * 0.5, 0.0, 0.0], [0.0, 0.0, -step * 0.5]);
            // Wide enough that neighbouring patches are plainly different, so
            // the edge between them is a real high-frequency albedo feature.
            let shade = 0.35 + 1.3 * rng.uniform();
            out.push(blade_render::ProceduralGeometry {
                name: format!("ground{row}-{column}"),
                vertices,
                indices,
                base_color_factor: [
                    (tone * shade * (0.75 + 0.5 * rng.uniform())).min(0.95),
                    (tone * shade * (0.75 + 0.5 * rng.uniform())).min(0.95),
                    (tone * shade * (0.75 + 0.5 * rng.uniform())).min(0.95),
                    1.0,
                ],
                metalness: 0.0,
                roughness: 0.4 + 0.55 * rng.uniform(),
                emissive_factor: [0.0; 3],
            });
        }
    }

    // Four rectangles around the patchwork, meeting it exactly. Overlapping
    // coplanar surfaces would make the ray tracer pick between them per hit.
    let surround = (extent + inner) * 0.5;
    let band = (extent - inner) * 0.5;
    let surrounds: [([f32; 3], [f32; 3], [f32; 3]); 4] = [
        ([0.0, 0.0, -surround], [extent, 0.0, 0.0], [0.0, 0.0, -band]),
        ([0.0, 0.0, surround], [extent, 0.0, 0.0], [0.0, 0.0, -band]),
        ([-surround, 0.0, 0.0], [band, 0.0, 0.0], [0.0, 0.0, -inner]),
        ([surround, 0.0, 0.0], [band, 0.0, 0.0], [0.0, 0.0, -inner]),
    ];
    let roughness = 0.4 + 0.55 * rng.uniform();
    let color = [
        tone,
        tone * (0.85 + 0.3 * rng.uniform()),
        tone * (0.85 + 0.3 * rng.uniform()),
        1.0,
    ];
    for (index, (center, u, v)) in surrounds.into_iter().enumerate() {
        let (vertices, indices) = rect(center, u, v);
        out.push(blade_render::ProceduralGeometry {
            name: format!("ground-surround{index}"),
            vertices,
            indices,
            base_color_factor: color,
            metalness: 0.0,
            roughness,
            emissive_factor: [0.0; 3],
        });
    }
    out
}

/// Separate two shaded objects so they can receive independent transforms.
///
/// The ground and emissive geometry remain in the static model. Picking one
/// sphere and one box gives the motion gate both curved and hard silhouettes;
/// the seed varies which material from each family is animated.
pub fn split_moving_geometry(geometries: Vec<Surface>, seed: u64) -> (Vec<Surface>, Vec<Surface>) {
    let spheres: Vec<_> = geometries
        .iter()
        .enumerate()
        .filter(|(_, surface)| surface.geometry.name.starts_with("sphere"))
        .map(|(index, _)| index)
        .collect();
    let boxes: Vec<_> = geometries
        .iter()
        .enumerate()
        .filter(|(_, surface)| surface.geometry.name.starts_with("box"))
        .map(|(index, _)| index)
        .collect();
    let mut rng = Rng::new(seed ^ 0xE703_7ED1_A0B4_28DB);
    let choose = |indices: &[usize], value: f32| {
        (!indices.is_empty())
            .then(|| indices[(value * indices.len() as f32) as usize % indices.len()])
    };
    let selected = [
        choose(&spheres, rng.uniform()),
        choose(&boxes, rng.uniform()),
    ];

    let mut static_geometry = Vec::with_capacity(geometries.len());
    let mut moving_geometry = Vec::with_capacity(2);
    for (index, geometry) in geometries.into_iter().enumerate() {
        if selected.contains(&Some(index)) {
            moving_geometry.push(geometry);
        } else {
            static_geometry.push(geometry);
        }
    }
    (static_geometry, moving_geometry)
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

/// A deterministic, smoothly curving camera translation for one sequence.
///
/// The first frame is always the unmodified base pose. The dominant direction,
/// vertical drift, and bend differ per sequence, while the distance between
/// adjacent frames stays close to `step`. This exercises motion-vector signs,
/// disocclusions, and non-axis-aligned reprojection without making generation
/// depend on mutable RNG state from an earlier sequence.
pub fn camera_motion(seed: u64, frame: usize, step: f32) -> [f32; 3] {
    if frame == 0 || step == 0.0 {
        return [0.0; 3];
    }
    let mut rng = Rng::new(seed ^ 0xA24B_AED4_963E_E407);
    let angle = std::f32::consts::TAU * rng.uniform();
    let bend = 0.12 * (2.0 * rng.uniform() - 1.0);
    let vertical = 0.15 * (2.0 * rng.uniform() - 1.0);
    let t = frame as f32;
    let side = bend * t * (t - 1.0) * 0.5;
    let (sin, cos) = angle.sin_cos();
    [
        step * (t * cos - side * sin),
        step * vertical * t,
        step * (t * sin + side * cos),
    ]
}

/// Independent, smoothly curving XZ translation for one moving object.
///
/// Geometry starts in its generated position, so frame zero is identity. Each
/// object gets a distinct direction, speed, and bend while staying on the
/// ground plane. Keeping the path deterministic lets a failed quality gate be
/// reproduced from just the dataset seed.
pub fn object_motion(seed: u64, object: usize, frame: usize, step: f32) -> [f32; 3] {
    if frame == 0 || step == 0.0 {
        return [0.0; 3];
    }
    let mut rng = Rng::new(
        seed ^ (object as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x8EBC_6AF0_9C88_C6E3,
    );
    let angle = std::f32::consts::TAU * rng.uniform();
    let speed = 0.75 + 0.5 * rng.uniform();
    let bend = 0.18 * (2.0 * rng.uniform() - 1.0);
    let t = frame as f32;
    let forward = speed * t;
    let side = bend * t * (t - 1.0) * 0.5;
    let (sin, cos) = angle.sin_cos();
    [
        step * (forward * cos - side * sin),
        0.0,
        step * (forward * sin + side * cos),
    ]
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
    fn camera_motion_is_sequence_local_and_curved() {
        assert_eq!(camera_motion(7, 0, 0.05), [0.0; 3]);
        assert_eq!(camera_motion(7, 3, 0.05), camera_motion(7, 3, 0.05));
        assert_ne!(camera_motion(7, 3, 0.05), camera_motion(8, 3, 0.05));

        let a = camera_motion(7, 1, 0.05);
        let b = camera_motion(7, 2, 0.05);
        let cross_y = a[2] * b[0] - a[0] * b[2];
        assert!(
            cross_y.abs() > 1.0e-7,
            "trajectory should bend: {a:?}, {b:?}"
        );
    }

    #[test]
    fn object_motion_is_independent_and_sequence_local() {
        assert_eq!(object_motion(7, 0, 0, 0.1), [0.0; 3]);
        assert_eq!(object_motion(7, 0, 2, 0.1), object_motion(7, 0, 2, 0.1));
        assert_ne!(object_motion(7, 0, 2, 0.1), object_motion(7, 1, 2, 0.1));
        assert_eq!(object_motion(7, 1, 3, 0.1)[1], 0.0);
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
        assert_eq!(
            a.len(),
            config.sphere_count + config.box_count + config.light_count + 1
        );
        assert_eq!(
            a[1].geometry.roughness, b[1].geometry.roughness,
            "same seed, same scene"
        );
        assert_ne!(
            a[1].geometry.roughness, c[1].geometry.roughness,
            "seeds should differ"
        );
    }

    #[test]
    fn moving_geometry_is_one_sphere_and_one_box() {
        let config = SceneConfig::default();
        let (static_geometry, moving_geometry) = split_moving_geometry(build(&config, 5), 7);
        assert_eq!(
            static_geometry.len() + moving_geometry.len(),
            config.sphere_count + config.box_count + config.light_count + 1
        );
        assert_eq!(moving_geometry.len(), 2);
        assert!(
            moving_geometry
                .iter()
                .any(|surface| surface.geometry.name.starts_with("sphere"))
        );
        assert!(
            moving_geometry
                .iter()
                .any(|surface| surface.geometry.name.starts_with("box"))
        );
        assert!(
            static_geometry
                .iter()
                .any(|surface| surface.geometry.name == "ground")
        );
    }

    #[test]
    fn textures_reach_surfaces_but_never_the_lights() {
        let plain = SceneConfig::default();
        assert!(
            build(&plain, 5).iter().all(|s| s.texture.is_none()),
            "textures are off by default, so every published dataset still means what it meant"
        );

        let config = SceneConfig {
            textures: true,
            ..SceneConfig::default()
        };
        let scene = build(&config, 5);
        let textured = scene.iter().filter(|s| s.texture.is_some()).count();
        assert!(
            textured >= 3,
            "only {textured} of {} surfaces got a texture",
            scene.len()
        );
        // An emissive sphere is the light, not a surface being lit, and
        // multiplying its base colour by a pattern would do nothing anyway.
        assert!(
            scene
                .iter()
                .filter(|s| s.geometry.name.starts_with("light"))
                .all(|s| s.texture.is_none()),
            "a light was given a texture"
        );
        // One scene should not be one pattern repeated, or a network can learn
        // the texture as a constant rather than reconstructing it.
        let kinds: std::collections::BTreeSet<_> = scene
            .iter()
            .filter_map(|s| s.texture.map(|kind| format!("{kind:?}")))
            .collect();
        assert!(kinds.len() > 1, "the whole scene used {kinds:?}");
    }

    #[test]
    fn gloss_adds_a_tight_lobe_without_removing_the_rough_ones() {
        let config = SceneConfig {
            gloss: true,
            ..SceneConfig::default()
        };
        let scene = build(&config, 11);
        let shaded: Vec<f32> = scene
            .iter()
            .filter(|s| s.geometry.name.starts_with("sphere") || s.geometry.name.starts_with("box"))
            .map(|s| s.geometry.roughness)
            .collect();
        assert!(
            shaded.iter().any(|&r| r < 0.15),
            "no surface got a tight lobe: {shaded:?}"
        );
        assert!(
            shaded.iter().any(|&r| r > 0.5),
            "everything went glossy, which would make the input mostly variance: {shaded:?}"
        );
        // Without the flag the floor holds, because that is what keeps the
        // sparse estimator's variance down on the existing sets.
        let plain = build(&SceneConfig::default(), 11);
        assert!(
            plain
                .iter()
                .filter(|s| {
                    s.geometry.name.starts_with("sphere") || s.geometry.name.starts_with("box")
                })
                .all(|s| s.geometry.roughness >= 0.15),
            "the default roughness floor moved"
        );
    }

    #[test]
    fn shaded_objects_rest_on_the_ground() {
        let config = SceneConfig::default();
        for surface in build(&config, 5).iter().filter(|surface| {
            surface.geometry.name.starts_with("sphere") || surface.geometry.name.starts_with("box")
        }) {
            let geometry = &surface.geometry;
            let lowest = geometry
                .vertices
                .iter()
                .map(|v| v.position[1])
                .fold(f32::INFINITY, f32::min);
            assert!(lowest > -0.01, "{} dips to {lowest}", geometry.name);
        }
    }
}
