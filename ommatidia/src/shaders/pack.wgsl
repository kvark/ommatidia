// Read the host's colour and G-buffer textures into the network's
// conditioning tensor.
//
// This is the GPU half of `ommatidia::batch::write_conditioning`, and the two
// have to agree exactly: the network was trained against what the CPU path
// produces, so any difference here shows up as a quality loss with nothing
// pointing at the cause. The plane order, the channel-major layout, and the
// transforms all follow `ommatidia::transform`.
//
// One invocation per input pixel, writing one value into every channel plane.

struct PackParams {
    // Input extent, which is also the tile the network was compiled for.
    width: u32,
    height: u32,
    // Number of conditioning channels, so the shader can be compiled once for
    // any plane set the model was configured with.
    channels: u32,
    // Which optional planes are present, matching `PlaneSet`'s bits.
    planes: u32,
    compose_blade_radiance: u32,
    decode_blade_gbuffer: u32,
    _pad0: u32,
    _pad1: u32,
}

// Bits of `ommatidia::dataset::Plane`, in storage order.
const PLANE_COLOR: u32 = 1u;
const PLANE_DEPTH: u32 = 2u;
const PLANE_NORMAL: u32 = 4u;
const PLANE_DIFFUSE_ALBEDO: u32 = 8u;
const PLANE_SPECULAR_F0: u32 = 16u;
const PLANE_ROUGHNESS: u32 = 32u;

var<uniform> params: PackParams;
var t_color: texture_2d<f32>;
var t_diffuse_radiance: texture_2d<f32>;
var t_specular_radiance: texture_2d<f32>;
var t_emissive: texture_2d<f32>;
var t_depth: texture_2d<f32>;
var t_normal: texture_2d<f32>;
var t_albedo: texture_2d<f32>;
var t_specular: texture_2d<f32>;
var<storage, read_write> cond: array<f32>;

const SKY_DEPTH: f32 = 1.0e6;

// Matches `qrot` in Blade's quaternion.inc.wgsl and the training-data probe.
fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

// Unbounded radiance into [0, 1). Mirrors `transform::compress`.
fn compress(x: f32) -> f32 {
    let v = max(x, 0.0);
    return v / (1.0 + v);
}

// View-space distance into (0, 1]. Mirrors `transform::encode_depth`.
fn encode_depth(d: f32) -> f32 {
    return 1.0 / (1.0 + max(d, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn pack(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let texel = vec2<i32>(i32(id.x), i32(id.y));
    let plane_stride = params.width * params.height;
    let offset = id.y * params.width + id.x;

    // `channel` walks forward as planes are emitted, so the layout matches
    // `PlaneSet::iter` without the shader needing the offsets precomputed.
    var channel = 0u;

    if (params.planes & PLANE_COLOR) != 0u {
        var color: vec3<f32>;
        if params.compose_blade_radiance != 0u {
            let albedo = textureLoad(t_albedo, texel, 0).xyz;
            let diffuse = textureLoad(t_diffuse_radiance, texel, 0).xyz;
            let specular = textureLoad(t_specular_radiance, texel, 0).xyz;
            let emissive = textureLoad(t_emissive, texel, 0).xyz;
            color = albedo * diffuse + specular + emissive;
        } else {
            color = textureLoad(t_color, texel, 0).xyz;
        }
        cond[(channel + 0u) * plane_stride + offset] = compress(color.x);
        cond[(channel + 1u) * plane_stride + offset] = compress(color.y);
        cond[(channel + 2u) * plane_stride + offset] = compress(color.z);
        channel += 3u;
    }
    if (params.planes & PLANE_DEPTH) != 0u {
        let raw_depth = textureLoad(t_depth, texel, 0).x;
        let depth = select(
            raw_depth,
            select(SKY_DEPTH, raw_depth, raw_depth > 0.0),
            params.decode_blade_gbuffer != 0u,
        );
        cond[channel * plane_stride + offset] = encode_depth(depth);
        channel += 1u;
    }
    if (params.planes & PLANE_NORMAL) != 0u {
        let encoded = textureLoad(t_normal, texel, 0);
        var normal = encoded.xyz;
        if params.decode_blade_gbuffer != 0u {
            let hit = textureLoad(t_depth, texel, 0).x > 0.0;
            normal = select(vec3<f32>(0.0), qrot(encoded, vec3<f32>(0.0, 0.0, 1.0)), hit);
        }
        cond[(channel + 0u) * plane_stride + offset] = normal.x;
        cond[(channel + 1u) * plane_stride + offset] = normal.y;
        cond[(channel + 2u) * plane_stride + offset] = normal.z;
        channel += 3u;
    }
    if (params.planes & PLANE_DIFFUSE_ALBEDO) != 0u {
        let albedo = textureLoad(t_albedo, texel, 0).xyz;
        cond[(channel + 0u) * plane_stride + offset] = albedo.x;
        cond[(channel + 1u) * plane_stride + offset] = albedo.y;
        cond[(channel + 2u) * plane_stride + offset] = albedo.z;
        channel += 3u;
    }
    // Blade packs the roughness into the alpha of the specular target, so both
    // planes come from one load.
    let specular = textureLoad(t_specular, texel, 0);
    if (params.planes & PLANE_SPECULAR_F0) != 0u {
        cond[(channel + 0u) * plane_stride + offset] = specular.x;
        cond[(channel + 1u) * plane_stride + offset] = specular.y;
        cond[(channel + 2u) * plane_stride + offset] = specular.z;
        channel += 3u;
    }
    if (params.planes & PLANE_ROUGHNESS) != 0u {
        cond[channel * plane_stride + offset] = specular.w;
        channel += 1u;
    }
}
