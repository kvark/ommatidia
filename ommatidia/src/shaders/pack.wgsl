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
    reconstruction_base: u32,
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
var<storage, read_write> base: array<f32>;

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

fn clamp_texel(texel: vec2<i32>) -> vec2<i32> {
    return clamp(texel, vec2<i32>(0), vec2<i32>(i32(params.width) - 1, i32(params.height) - 1));
}

fn load_color(texel_unclamped: vec2<i32>) -> vec3<f32> {
    let texel = clamp_texel(texel_unclamped);
    if params.compose_blade_radiance != 0u {
        let albedo = textureLoad(t_albedo, texel, 0).xyz;
        let diffuse = textureLoad(t_diffuse_radiance, texel, 0).xyz;
        let specular = textureLoad(t_specular_radiance, texel, 0).xyz;
        let emissive = textureLoad(t_emissive, texel, 0).xyz;
        return albedo * diffuse + specular + emissive;
    }
    return textureLoad(t_color, texel, 0).xyz;
}

fn load_depth(texel_unclamped: vec2<i32>) -> f32 {
    let raw = textureLoad(t_depth, clamp_texel(texel_unclamped), 0).x;
    let depth = select(
        raw,
        select(SKY_DEPTH, raw, raw > 0.0),
        params.decode_blade_gbuffer != 0u,
    );
    return encode_depth(depth);
}

fn load_normal(texel_unclamped: vec2<i32>) -> vec3<f32> {
    let texel = clamp_texel(texel_unclamped);
    let encoded = textureLoad(t_normal, texel, 0);
    if params.decode_blade_gbuffer != 0u {
        let hit = textureLoad(t_depth, texel, 0).x > 0.0;
        return select(vec3<f32>(0.0), qrot(encoded, vec3<f32>(0.0, 0.0, 1.0)), hit);
    }
    return encoded.xyz;
}

// Measured on held-out sparse-path data. Filtering once at input resolution
// and bilinearly reconstructing it is both better and four times cheaper than
// evaluating the same guide separately for every 2x output sample.
fn guided_color(center: vec2<i32>) -> vec3<f32> {
    let center_depth = load_depth(center);
    let center_normal = load_normal(center);
    let center_albedo = textureLoad(t_albedo, clamp_texel(center), 0).xyz;
    let center_normal_len2 = dot(center_normal, center_normal);
    var sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var dy = -6; dy <= 6; dy += 1) {
        for (var dx = -6; dx <= 6; dx += 1) {
            let texel = center + vec2<i32>(dx, dy);
            let distance2 = f32(dx * dx + dy * dy);
            var weight = exp(-distance2 / 18.0);
            let depth_delta = load_depth(texel) - center_depth;
            weight *= exp(-(depth_delta * depth_delta) / 0.005);

            let normal = load_normal(texel);
            let normal_len2 = dot(normal, normal);
            var normal_weight = 0.0;
            if center_normal_len2 < 0.25 {
                normal_weight = select(0.0, 1.0, normal_len2 < 0.25);
            } else if normal_len2 >= 0.25 {
                let cosine = max(dot(normal, center_normal) * inverseSqrt(normal_len2 * center_normal_len2), 0.0);
                normal_weight = pow(cosine, 32.0);
            }
            weight *= normal_weight;

            let albedo_delta = textureLoad(t_albedo, clamp_texel(texel), 0).xyz - center_albedo;
            weight *= exp(-dot(albedo_delta, albedo_delta) / 0.02);
            sum += weight * load_color(texel);
            weight_sum += weight;
        }
    }
    return sum / max(weight_sum, 1.0e-12);
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
        let color = load_color(texel);
        cond[(channel + 0u) * plane_stride + offset] = compress(color.x);
        cond[(channel + 1u) * plane_stride + offset] = compress(color.y);
        cond[(channel + 2u) * plane_stride + offset] = compress(color.z);
        channel += 3u;
    }
    if (params.planes & PLANE_DEPTH) != 0u {
        cond[channel * plane_stride + offset] = load_depth(texel);
        channel += 1u;
    }
    if (params.planes & PLANE_NORMAL) != 0u {
        let normal = load_normal(texel);
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

    var base_color = load_color(texel);
    if params.reconstruction_base == 2u {
        base_color = guided_color(texel);
    }
    base[0u * plane_stride + offset] = base_color.x;
    base[1u * plane_stride + offset] = base_color.y;
    base[2u * plane_stride + offset] = base_color.z;
}
