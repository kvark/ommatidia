// Read blade's G-buffer into a planar buffer the dataset writer can consume.
//
// Writing straight into a storage buffer rather than a texture means one
// readback instead of three, and the result is already channel-major, which is
// the layout `.omd` records use.
//
// The values here are the physical quantities the renderer produced. The
// transforms the network wants are applied on load, in `ommatidia::transform`,
// for the reason `ommatidia::dataset` documents.

struct Params {
    width: u32,
    height: u32,
    include_motion: u32,
    _pad: u32,
}

var<uniform> params: Params;
var t_depth: texture_2d<f32>;
var t_basis: texture_2d<f32>;
var t_diffuse_albedo: texture_2d<f32>;
var t_specular_f0: texture_2d<f32>;
var t_motion: texture_2d<f32>;
var<storage, read_write> planes: array<f32>;

// Distance recorded where no geometry was hit.
//
// Blade leaves the depth at zero for a miss, which would encode as the nearest
// possible surface rather than the sky. Anything past the far plane reads as
// zero once inverted, which is what the sky should look like.
const SKY_DEPTH: f32 = 1.0e6;
const MOTION_SCALE: f32 = 0.02;

// Matches `qrot` in blade's quaternion.inc.wgsl.
fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

@compute @workgroup_size(8, 8, 1)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let texel = vec2<i32>(i32(id.x), i32(id.y));
    let stride = params.width * params.height;
    let offset = id.y * params.width + id.x;

    let raw_depth = textureLoad(t_depth, texel, 0).x;
    let hit = raw_depth > 0.0;

    // The shading normal is the tangent frame applied to +Z, so it carries the
    // normal-mapped detail rather than only the triangle's orientation. A miss
    // gets a zero normal, which is not a direction any surface can have and so
    // marks the sky unambiguously.
    let basis = textureLoad(t_basis, texel, 0);
    let normal = select(vec3<f32>(0.0), qrot(basis, vec3<f32>(0.0, 0.0, 1.0)), hit);

    let albedo = textureLoad(t_diffuse_albedo, texel, 0).xyz;
    let specular = textureLoad(t_specular_f0, texel, 0);

    // Plane order follows `ommatidia::dataset::ALL_PLANES` with the colour
    // planes, which the post processing already produced, left out.
    planes[0u * stride + offset] = select(SKY_DEPTH, raw_depth, hit);
    planes[1u * stride + offset] = normal.x;
    planes[2u * stride + offset] = normal.y;
    planes[3u * stride + offset] = normal.z;
    planes[4u * stride + offset] = albedo.x;
    planes[5u * stride + offset] = albedo.y;
    planes[6u * stride + offset] = albedo.z;
    planes[7u * stride + offset] = specular.x;
    planes[8u * stride + offset] = specular.y;
    planes[9u * stride + offset] = specular.z;
    planes[10u * stride + offset] = specular.w;
    if params.include_motion != 0u {
        let motion = textureLoad(t_motion, texel, 0).xy / MOTION_SCALE;
        planes[11u * stride + offset] = motion.x;
        planes[12u * stride + offset] = motion.y;
    }
}
