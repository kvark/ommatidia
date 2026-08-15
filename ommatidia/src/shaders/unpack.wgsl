// Scatter the network's sub-pixel output into the high resolution target.
//
// The GPU half of `ommatidia::batch::assemble`. Each invocation owns one
// *input* pixel and writes all `scale * scale` output texels that came from
// it. The residual remains a free reindex; its deterministic base is either
// the historical nearest reconstruction or texel-center-aligned bilinear.
//
// Channel `c * scale^2 + dy * scale + dx` holds sub-pixel `(dy, dx)` of colour
// channel `c`, matching the layout documented in `ommatidia::batch`.

struct UnpackParams {
    // Input extent.
    width: u32,
    height: u32,
    scale: u32,
    // The residual was trained scaled to unit variance; dividing it back out
    // is the last thing that happens before it is added to the base.
    inverse_gain: f32,
    reconstruction_base: u32,
    decode_blade_gbuffer: u32,
    decode_hr_blade_gbuffer: u32,
    _pad2: u32,
    guide_spatial_denominator: f32,
    guide_depth_denominator: f32,
    guide_normal_power: f32,
    guide_albedo_denominator: f32,
}

var<uniform> params: UnpackParams;
var<storage, read> base_pixels: array<f32>;
var<storage, read> residual: array<f32>;
var t_depth: texture_2d<f32>;
var t_normal: texture_2d<f32>;
var t_albedo: texture_2d<f32>;
var t_hr_depth: texture_2d<f32>;
var t_hr_normal: texture_2d<f32>;
var t_hr_albedo: texture_2d<f32>;
var output: texture_storage_2d<rgba16float, write>;

const SKY_DEPTH: f32 = 1.0e6;

// Mirrors `batch::GATHER_FALLBACK`. Below this the guided gather has rejected
// every tap and its normalisation is meaningless.
const GATHER_FALLBACK: f32 = 1.0e-4;

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

fn encode_depth(d: f32) -> f32 {
    return 1.0 / (1.0 + max(d, 0.0));
}

fn compress(x: f32) -> f32 {
    let v = max(x, 0.0);
    return v / (1.0 + v);
}

// Inverse of `compress`, held off 1.0 where it diverges. Mirrors
// `transform::decompress`.
fn decompress(y: f32) -> f32 {
    let v = clamp(y, 0.0, 1.0 - 1.0 / 4096.0);
    return v / (1.0 - v);
}

fn load_base(texel_unclamped: vec2<i32>) -> vec3<f32> {
    let upper = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
    let texel = clamp(texel_unclamped, vec2<i32>(0), upper);
    let stride = params.width * params.height;
    let offset = u32(texel.y) * params.width + u32(texel.x);
    return vec3<f32>(
        base_pixels[0u * stride + offset],
        base_pixels[1u * stride + offset],
        base_pixels[2u * stride + offset],
    );
}

fn clamp_low(texel: vec2<i32>) -> vec2<i32> {
    return clamp(
        texel,
        vec2<i32>(0),
        vec2<i32>(i32(params.width) - 1, i32(params.height) - 1),
    );
}

fn load_low_depth(texel_unclamped: vec2<i32>) -> f32 {
    let raw = textureLoad(t_depth, clamp_low(texel_unclamped), 0).x;
    let depth = select(
        raw,
        select(SKY_DEPTH, raw, raw > 0.0),
        params.decode_blade_gbuffer != 0u,
    );
    return encode_depth(depth);
}

fn load_low_normal(texel_unclamped: vec2<i32>) -> vec3<f32> {
    let texel = clamp_low(texel_unclamped);
    let encoded = textureLoad(t_normal, texel, 0);
    if params.decode_blade_gbuffer != 0u {
        let hit = textureLoad(t_depth, texel, 0).x > 0.0;
        return select(vec3<f32>(0.0), qrot(encoded, vec3<f32>(0.0, 0.0, 1.0)), hit);
    }
    return encoded.xyz;
}

fn load_hr_depth(texel: vec2<i32>) -> f32 {
    let raw = textureLoad(t_hr_depth, texel, 0).x;
    let depth = select(
        raw,
        select(SKY_DEPTH, raw, raw > 0.0),
        params.decode_hr_blade_gbuffer != 0u,
    );
    return encode_depth(depth);
}

fn load_hr_normal(texel: vec2<i32>) -> vec3<f32> {
    let encoded = textureLoad(t_hr_normal, texel, 0);
    if params.decode_hr_blade_gbuffer != 0u {
        let hit = textureLoad(t_hr_depth, texel, 0).x > 0.0;
        return select(vec3<f32>(0.0), qrot(encoded, vec3<f32>(0.0, 0.0, 1.0)), hit);
    }
    return encoded.xyz;
}

fn guide_similarity(
    center_depth: f32,
    center_normal: vec3<f32>,
    center_albedo: vec3<f32>,
    depth: f32,
    normal: vec3<f32>,
    albedo: vec3<f32>,
) -> f32 {
    let depth_delta = depth - center_depth;
    var weight = exp(-(depth_delta * depth_delta) / params.guide_depth_denominator);
    let center_normal_len2 = dot(center_normal, center_normal);
    let normal_len2 = dot(normal, normal);
    var normal_weight = 0.0;
    if center_normal_len2 < 0.25 {
        normal_weight = select(0.0, 1.0, normal_len2 < 0.25);
    } else if normal_len2 >= 0.25 {
        let cosine = max(
            dot(normal, center_normal) * inverseSqrt(normal_len2 * center_normal_len2),
            0.0,
        );
        normal_weight = pow(cosine, params.guide_normal_power);
    }
    let albedo_delta = albedo - center_albedo;
    return weight * normal_weight * exp(-dot(albedo_delta, albedo_delta) / params.guide_albedo_denominator);
}

fn high_resolution_guided_base(destination: vec2<u32>) -> vec3<f32> {
    let hr_texel = vec2<i32>(destination);
    let center_depth = load_hr_depth(hr_texel);
    let center_normal = load_hr_normal(hr_texel);
    let center_albedo = textureLoad(t_hr_albedo, hr_texel, 0).xyz;
    let position = (vec2<f32>(destination) + 0.5) / f32(params.scale) - 0.5;
    let lower = vec2<i32>(floor(position));
    var sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    // `guide_similarity` returns exactly zero for a tap whose normal faces away
    // from the centre's, and for a background centre next to geometry. At a
    // silhouette every tap can be rejected at once, so the guide-free gather is
    // carried alongside to fall back to. See `batch::GATHER_FALLBACK`.
    var spatial_sum = vec3<f32>(0.0);
    var spatial_weight_sum = 0.0;
    for (var dy = -2; dy <= 2; dy += 1) {
        for (var dx = -2; dx <= 2; dx += 1) {
            let texel = clamp_low(lower + vec2<i32>(dx, dy));
            let delta = vec2<f32>(texel) - position;
            let spatial = exp(-dot(delta, delta) / 4.5);
            let weight = spatial * guide_similarity(
                center_depth,
                center_normal,
                center_albedo,
                load_low_depth(texel),
                load_low_normal(texel),
                textureLoad(t_albedo, texel, 0).xyz,
            );
            let tap = load_base(texel);
            sum += weight * tap;
            spatial_sum += spatial * tap;
            weight_sum += weight;
            spatial_weight_sum += spatial;
        }
    }
    if weight_sum > GATHER_FALLBACK {
        return sum / weight_sum;
    }
    return spatial_sum / max(spatial_weight_sum, 1.0e-12);
}

fn reconstruction_base(destination: vec2<u32>, source: vec2<i32>) -> vec3<f32> {
    if params.reconstruction_base == 0u {
        return load_base(source);
    }
    if params.reconstruction_base == 3u {
        return high_resolution_guided_base(destination);
    }
    let position = (vec2<f32>(destination) + 0.5) / f32(params.scale) - 0.5;
    let lower_f = floor(position);
    let lower = vec2<i32>(lower_f);
    let fraction = position - lower_f;
    let top = mix(load_base(lower), load_base(lower + vec2<i32>(1, 0)), fraction.x);
    let bottom = mix(
        load_base(lower + vec2<i32>(0, 1)),
        load_base(lower + vec2<i32>(1, 1)),
        fraction.x,
    );
    return mix(top, bottom, fraction.y);
}

@compute @workgroup_size(8, 8, 1)
fn unpack(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let source = vec2<i32>(i32(id.x), i32(id.y));

    let plane_stride = params.width * params.height;
    let offset = id.y * params.width + id.x;
    let sub = params.scale * params.scale;

    for (var dy = 0u; dy < params.scale; dy += 1u) {
        for (var dx = 0u; dx < params.scale; dx += 1u) {
            let slot = dy * params.scale + dx;
            let destination = id.xy * params.scale + vec2<u32>(dx, dy);
            let low = reconstruction_base(destination, source);
            let base = vec3<f32>(compress(low.x), compress(low.y), compress(low.z));
            var color: vec3<f32>;
            for (var c = 0u; c < 3u; c += 1u) {
                let channel = c * sub + slot;
                let delta = residual[channel * plane_stride + offset] * params.inverse_gain;
                color[c] = decompress(base[c] + delta);
            }
            textureStore(output, vec2<i32>(destination), vec4<f32>(color, 1.0));
        }
    }
}
