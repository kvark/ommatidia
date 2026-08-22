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
    guide_spatial_denominator: f32,
    guide_depth_denominator: f32,
    guide_normal_power: f32,
    guide_albedo_denominator: f32,
    history_frames: u32,
    history_ready: u32,
    motion_scale: f32,
    rejection_depth_delta: f32,
    rejection_normal_cosine: f32,
    rejection_albedo_delta2: f32,
    _pad2a: u32,
    _pad2b: u32,
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
var t_motion: texture_2d<f32>;
var<storage, read> previous_low_history: array<f32>;
var<storage, read_write> current_low_history: array<f32>;
var<storage, read_write> cond: array<f32>;
var<storage, read_write> base: array<f32>;

const SKY_DEPTH: f32 = 1.0e6;
const SURFACE_SKY_ENCODED_DEPTH: f32 = 1.0 / (1.0 + 60000.0);

const HISTORY_COUNT: u32 = 3u;
const HISTORY_LUMINANCE: u32 = 4u;
const HISTORY_LUMINANCE_SQUARE: u32 = 5u;
const HISTORY_DEPTH: u32 = 6u;
const HISTORY_NORMAL: u32 = 7u;
const HISTORY_ALBEDO: u32 = 10u;

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
            var weight = exp(-distance2 / params.guide_spatial_denominator);
            let depth_delta = load_depth(texel) - center_depth;
            weight *= exp(-(depth_delta * depth_delta) / params.guide_depth_denominator);

            let normal = load_normal(texel);
            let normal_len2 = dot(normal, normal);
            var normal_weight = 0.0;
            if center_normal_len2 < 0.25 {
                normal_weight = select(0.0, 1.0, normal_len2 < 0.25);
            } else if normal_len2 >= 0.25 {
                let cosine = max(dot(normal, center_normal) * inverseSqrt(normal_len2 * center_normal_len2), 0.0);
                normal_weight = pow(cosine, params.guide_normal_power);
            }
            weight *= normal_weight;

            let albedo_delta = textureLoad(t_albedo, clamp_texel(texel), 0).xyz - center_albedo;
            weight *= exp(-dot(albedo_delta, albedo_delta) / params.guide_albedo_denominator);
            sum += weight * load_color(texel);
            weight_sum += weight;
        }
    }
    return sum / max(weight_sum, 1.0e-12);
}

fn write_planes(texel: vec2<i32>, plane_stride: u32, offset: u32, color: vec3<f32>) -> u32 {
    // `channel` walks forward as planes are emitted, so the layout matches
    // `PlaneSet::iter` without the shader needing precomputed offsets.
    var channel = 0u;
    if (params.planes & PLANE_COLOR) != 0u {
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
    // Blade packs roughness into the alpha of the specular target, so both
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
    return channel;
}

fn write_base(texel: vec2<i32>, plane_stride: u32, offset: u32, current: vec3<f32>) {
    // A kernel checkpoint gathers the samples themselves, so it wants the
    // colour untouched. Prefiltering here would put a second filter back into
    // the path that was built to remove it.
    var base_color = current;
    if params.reconstruction_base == 2u || params.reconstruction_base == 3u {
        base_color = guided_color(texel);
    }
    base[0u * plane_stride + offset] = base_color.x;
    base[1u * plane_stride + offset] = base_color.y;
    base[2u * plane_stride + offset] = base_color.z;
}

fn history_scalar(plane: u32, index: u32, stride: u32) -> f32 {
    return previous_low_history[plane * stride + index];
}

fn history_vec3(plane: u32, index: u32, stride: u32) -> vec3<f32> {
    return vec3<f32>(
        history_scalar(plane + 0u, index, stride),
        history_scalar(plane + 1u, index, stride),
        history_scalar(plane + 2u, index, stride),
    );
}

fn history_bilinear_scalar(plane: u32, position: vec2<f32>, stride: u32) -> f32 {
    let lower_f = floor(position);
    let lower = vec2<i32>(lower_f);
    let fraction = position - lower_f;
    let upper = vec2<i32>(i32(params.width) - 1, i32(params.height) - 1);
    let at = clamp(lower, vec2<i32>(0), upper);
    let bx = clamp(lower + vec2<i32>(1, 0), vec2<i32>(0), upper);
    let ay = clamp(lower + vec2<i32>(0, 1), vec2<i32>(0), upper);
    let by = clamp(lower + vec2<i32>(1, 1), vec2<i32>(0), upper);
    let index_at = u32(at.y) * params.width + u32(at.x);
    let index_bx = u32(bx.y) * params.width + u32(bx.x);
    let index_ay = u32(ay.y) * params.width + u32(ay.x);
    let index_by = u32(by.y) * params.width + u32(by.x);
    let top = mix(
        history_scalar(plane, index_at, stride),
        history_scalar(plane, index_bx, stride),
        fraction.x,
    );
    let bottom = mix(
        history_scalar(plane, index_ay, stride),
        history_scalar(plane, index_by, stride),
        fraction.x,
    );
    return mix(top, bottom, fraction.y);
}

fn history_bilinear_vec3(plane: u32, position: vec2<f32>, stride: u32) -> vec3<f32> {
    return vec3<f32>(
        history_bilinear_scalar(plane + 0u, position, stride),
        history_bilinear_scalar(plane + 1u, position, stride),
        history_bilinear_scalar(plane + 2u, position, stride),
    );
}

fn surfaces_match(
    current_depth: f32,
    current_normal: vec3<f32>,
    current_albedo: vec3<f32>,
    previous_depth: f32,
    previous_normal: vec3<f32>,
    previous_albedo: vec3<f32>,
) -> bool {
    let current_sky = current_depth <= SURFACE_SKY_ENCODED_DEPTH;
    let previous_sky = previous_depth <= SURFACE_SKY_ENCODED_DEPTH;
    if current_sky || previous_sky {
        return current_sky == previous_sky;
    }
    if abs(current_depth - previous_depth) > params.rejection_depth_delta {
        return false;
    }
    let normal_denominator = sqrt(
        max(dot(current_normal, current_normal) * dot(previous_normal, previous_normal), 1.0e-12),
    );
    let cosine = dot(current_normal, previous_normal) / normal_denominator;
    let albedo_delta = current_albedo - previous_albedo;
    return cosine > params.rejection_normal_cosine
        && dot(albedo_delta, albedo_delta) < params.rejection_albedo_delta2;
}

@compute @workgroup_size(8, 8, 1)
fn pack(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let texel = vec2<i32>(i32(id.x), i32(id.y));
    let plane_stride = params.width * params.height;
    let offset = id.y * params.width + id.x;
    let current = load_color(texel);
    _ = write_planes(texel, plane_stride, offset, current);
    write_base(texel, plane_stride, offset, current);
}

// Sequence-aware half of `batch::write_temporal_conditioning`. The sparse
// sample history lives at input resolution and is distinct from the previous
// reconstructed output mixed by `unpack_temporal`.
@compute @workgroup_size(8, 8, 1)
fn pack_temporal(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let texel = vec2<i32>(i32(id.x), i32(id.y));
    let stride = params.width * params.height;
    let offset = id.y * params.width + id.x;
    let current_color = load_color(texel);
    let current_depth = load_depth(texel);
    let current_normal = load_normal(texel);
    let current_albedo = textureLoad(t_albedo, texel, 0).xyz;
    let motion = textureLoad(t_motion, texel, 0).xy * params.motion_scale;
    let position = vec2<f32>(texel) + motion;
    let inside = position.x >= 0.0
        && position.y >= 0.0
        && position.x <= f32(params.width - 1u)
        && position.y <= f32(params.height - 1u);

    // CPU accumulation validates at the nearest surface and then bilinearly
    // samples history values. `floor(p + 0.5)` matches Rust's positive-coordinate
    // rounding without relying on the shader language's tie rule.
    let rounded = clamp(
        vec2<i32>(floor(position + vec2<f32>(0.5))),
        vec2<i32>(0),
        vec2<i32>(i32(params.width) - 1, i32(params.height) - 1),
    );
    let previous_index = u32(rounded.y) * params.width + u32(rounded.x);
    let valid = params.history_ready != 0u
        && inside
        && surfaces_match(
            current_depth,
            current_normal,
            current_albedo,
            history_scalar(HISTORY_DEPTH, previous_index, stride),
            history_vec3(HISTORY_NORMAL, previous_index, stride),
            history_vec3(HISTORY_ALBEDO, previous_index, stride),
        );

    var previous_count = 0.0;
    var previous_color = vec3<f32>(0.0);
    var previous_luminance = 0.0;
    var previous_luminance_square = 0.0;
    if valid {
        previous_count = min(
            history_bilinear_scalar(HISTORY_COUNT, position, stride),
            f32(params.history_frames - 1u),
        );
        previous_color = history_bilinear_vec3(0u, position, stride);
        previous_luminance = history_bilinear_scalar(HISTORY_LUMINANCE, position, stride);
        previous_luminance_square = history_bilinear_scalar(
            HISTORY_LUMINANCE_SQUARE,
            position,
            stride,
        );
    }

    let count = previous_count + 1.0;
    let accumulated = (current_color + previous_count * previous_color) / count;
    let compressed = vec3<f32>(
        compress(current_color.x),
        compress(current_color.y),
        compress(current_color.z),
    );
    let luminance = dot(compressed, vec3<f32>(0.2126, 0.7152, 0.0722));
    let mean_luminance = (luminance + previous_count * previous_luminance) / count;
    let mean_luminance_square = (
        luminance * luminance + previous_count * previous_luminance_square
    ) / count;

    current_low_history[0u * stride + offset] = accumulated.x;
    current_low_history[1u * stride + offset] = accumulated.y;
    current_low_history[2u * stride + offset] = accumulated.z;
    current_low_history[HISTORY_COUNT * stride + offset] = count;
    current_low_history[HISTORY_LUMINANCE * stride + offset] = mean_luminance;
    current_low_history[HISTORY_LUMINANCE_SQUARE * stride + offset] = mean_luminance_square;
    current_low_history[HISTORY_DEPTH * stride + offset] = current_depth;
    current_low_history[(HISTORY_NORMAL + 0u) * stride + offset] = current_normal.x;
    current_low_history[(HISTORY_NORMAL + 1u) * stride + offset] = current_normal.y;
    current_low_history[(HISTORY_NORMAL + 2u) * stride + offset] = current_normal.z;
    current_low_history[(HISTORY_ALBEDO + 0u) * stride + offset] = current_albedo.x;
    current_low_history[(HISTORY_ALBEDO + 1u) * stride + offset] = current_albedo.y;
    current_low_history[(HISTORY_ALBEDO + 2u) * stride + offset] = current_albedo.z;

    var channel = write_planes(texel, stride, offset, accumulated);
    cond[(channel + 0u) * stride + offset] = compressed.x;
    cond[(channel + 1u) * stride + offset] = compressed.y;
    cond[(channel + 2u) * stride + offset] = compressed.z;
    cond[(channel + 3u) * stride + offset] = count / f32(params.history_frames);
    channel += 4u;
    if channel < params.channels {
        cond[channel * stride + offset] = sqrt(max(
            mean_luminance_square - mean_luminance * mean_luminance,
            0.0,
        ));
    }
    write_base(texel, stride, offset, current_color);
}
