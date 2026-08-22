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
    kernel_radius: u32,
    demodulate: u32,
    // How far demodulation may rescale a pixel. See
    // `ModelConfig::demodulation_offset`.
    demodulation_offset: f32,
    guide_spatial_denominator: f32,
    guide_depth_denominator: f32,
    guide_normal_power: f32,
    guide_albedo_denominator: f32,
    history_ready: u32,
    _pad1: u32,
    motion_scale: f32,
    rejection_depth_delta: f32,
    rejection_normal_cosine: f32,
    rejection_albedo_delta2: f32,
    _pad2a: u32,
    _pad2b: u32,
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
var t_motion: texture_2d<f32>;
var t_history_output: texture_2d<f32>;
var t_history_surface0: texture_2d<f32>;
var t_history_surface1: texture_2d<f32>;
var history_output: texture_storage_2d<rgba16float, write>;
var history_surface0: texture_storage_2d<rgba16float, write>;
var history_surface1: texture_storage_2d<rgba8unorm, write>;
var output: texture_storage_2d<rgba16float, write>;

const SKY_DEPTH: f32 = 1.0e6;
const SURFACE_SKY_ENCODED_DEPTH: f32 = 1.0 / (1.0 + 60000.0);

// Mirrors `batch::GATHER_FALLBACK`. Below this the guided gather has rejected
// every tap and its normalisation is meaningless.
const GATHER_FALLBACK: f32 = 1.0e-4;

// Mirrors `batch::KERNEL_FLOOR`. Softplus weights are strictly positive, so
// this only guards against every one of them underflowing in f32.
const KERNEL_FLOOR: f32 = 1.0e-20;


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

// The whole reconstruction, for a kernel checkpoint: one weighted pass over
// the input samples, with no filtered image in between. `residual` holds the
// network's gather weights, channel `slot * taps + tap`, and the tap order is
// `ModelConfig::tap_offset` — dy outer, dx inner, both from -radius.
//
// The CPU half is `batch::assemble_kernel`. They have to agree texel for
// texel, so the loop bounds, the ordering, and the floor are all mirrored
// rather than reimplemented.
fn gather_kernel(source: vec2<i32>, slot: u32, plane_stride: u32, offset: u32) -> vec3<f32> {
    let radius = i32(params.kernel_radius);
    let taps = u32((2 * radius + 1) * (2 * radius + 1));
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    var tap = 0u;
    for (var dy = -radius; dy <= radius; dy += 1) {
        for (var dx = -radius; dx <= radius; dx += 1) {
            let weight = residual[(slot * taps + tap) * plane_stride + offset];
            let texel = clamp_low(source + vec2<i32>(dx, dy));
            var color = load_base(texel);
            if params.demodulate != 0u {
                color /= textureLoad(t_albedo, texel, 0).xyz + params.demodulation_offset;
            }
            sum += weight * vec3<f32>(compress(color.x), compress(color.y), compress(color.z));
            total += weight;
            tap += 1u;
        }
    }
    return sum / max(total, KERNEL_FLOOR);
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

struct Surface {
    depth: f32,
    normal: vec3<f32>,
    albedo: vec3<f32>,
}

fn current_surface(destination: vec2<u32>) -> Surface {
    let texel = vec2<i32>(destination);
    return Surface(
        load_hr_depth(texel),
        load_hr_normal(texel),
        textureLoad(t_hr_albedo, texel, 0).xyz,
    );
}

fn previous_surface(texel: vec2<i32>) -> Surface {
    let first = textureLoad(t_history_surface0, texel, 0);
    return Surface(
        first.x,
        first.yzw,
        textureLoad(t_history_surface1, texel, 0).xyz,
    );
}

fn surfaces_match(current: Surface, previous: Surface) -> bool {
    let current_sky = current.depth <= SURFACE_SKY_ENCODED_DEPTH;
    let previous_sky = previous.depth <= SURFACE_SKY_ENCODED_DEPTH;
    if current_sky || previous_sky {
        return current_sky == previous_sky;
    }
    if abs(current.depth - previous.depth) > params.rejection_depth_delta {
        return false;
    }
    let normal_denominator = sqrt(max(
        dot(current.normal, current.normal) * dot(previous.normal, previous.normal),
        1.0e-12,
    ));
    let cosine = dot(current.normal, previous.normal) / normal_denominator;
    let albedo_delta = current.albedo - previous.albedo;
    return cosine > params.rejection_normal_cosine
        && dot(albedo_delta, albedo_delta) < params.rejection_albedo_delta2;
}

// Motion-reproject the previous compressed reconstruction, dropping bilinear
// taps whose stored primary surface no longer matches. RGB is the accepted
// history and alpha is explicit validity; rejected storage never means black.
fn reproject_history(destination: vec2<u32>, source: vec2<i32>) -> vec4<f32> {
    if params.history_ready == 0u {
        return vec4<f32>(0.0);
    }
    let motion = textureLoad(t_motion, source, 0).xy * params.motion_scale;
    let position = vec2<f32>(destination) + motion * f32(params.scale);
    let output_extent = vec2<u32>(params.width, params.height) * params.scale;
    if position.x < 0.0
        || position.y < 0.0
        || position.x > f32(output_extent.x - 1u)
        || position.y > f32(output_extent.y - 1u)
    {
        return vec4<f32>(0.0);
    }
    let lower_f = floor(position);
    let lower = vec2<i32>(lower_f);
    let fraction = position - lower_f;
    let upper = vec2<i32>(output_extent) - vec2<i32>(1);
    let surface = current_surface(destination);
    var color = vec3<f32>(0.0);
    var total = 0.0;
    for (var dy = 0; dy <= 1; dy += 1) {
        let wy = select(1.0 - fraction.y, fraction.y, dy != 0);
        for (var dx = 0; dx <= 1; dx += 1) {
            let wx = select(1.0 - fraction.x, fraction.x, dx != 0);
            let weight = wx * wy;
            if weight == 0.0 {
                continue;
            }
            let texel = clamp(lower + vec2<i32>(dx, dy), vec2<i32>(0), upper);
            if !surfaces_match(surface, previous_surface(texel)) {
                continue;
            }
            color += weight * textureLoad(t_history_output, texel, 0).xyz;
            total += weight;
        }
    }
    if total == 0.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(color / total, 1.0);
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

    if params.reconstruction_base == 4u {
        for (var dy = 0u; dy < params.scale; dy += 1u) {
            for (var dx = 0u; dx < params.scale; dx += 1u) {
                let gathered = gather_kernel(source, dy * params.scale + dx, plane_stride, offset);
                var color = vec3<f32>(
                    decompress(gathered.x),
                    decompress(gathered.y),
                    decompress(gathered.z),
                );
                let destination = id.xy * params.scale + vec2<u32>(dx, dy);
                if params.demodulate != 0u {
                    // The exact output-resolution albedo, which is what puts
                    // the texture back at a resolution the gather never had to
                    // reconstruct it at.
                    color *= textureLoad(t_hr_albedo, vec2<i32>(destination), 0).xyz
                        + params.demodulation_offset;
                }
                textureStore(output, vec2<i32>(destination), vec4<f32>(color, 1.0));
            }
        }
        return;
    }

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

@compute @workgroup_size(8, 8, 1)
fn unpack_temporal(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let source = vec2<i32>(i32(id.x), i32(id.y));
    let plane_stride = params.width * params.height;
    let offset = id.y * params.width + id.x;
    let taps = u32((2 * i32(params.kernel_radius) + 1) * (2 * i32(params.kernel_radius) + 1));
    let slots = params.scale * params.scale;

    for (var dy = 0u; dy < params.scale; dy += 1u) {
        for (var dx = 0u; dx < params.scale; dx += 1u) {
            let slot = dy * params.scale + dx;
            let destination = id.xy * params.scale + vec2<u32>(dx, dy);
            let gathered = gather_kernel(source, slot, plane_stride, offset);
            let previous = reproject_history(destination, source);
            let mixture = residual[(slots * taps + slot) * plane_stride + offset];
            let gate = previous.w * mixture / (mixture + 1.0);
            let compressed = mix(gathered, previous.xyz, gate);

            // Store exactly the compressed, demodulated representation the CPU
            // evaluator feeds back. The caller receives linear radiance after
            // the current frame's exact albedo is restored.
            textureStore(history_output, vec2<i32>(destination), vec4<f32>(compressed, 1.0));
            let surface = current_surface(destination);
            textureStore(
                history_surface0,
                vec2<i32>(destination),
                vec4<f32>(surface.depth, surface.normal),
            );
            textureStore(
                history_surface1,
                vec2<i32>(destination),
                vec4<f32>(surface.albedo, 1.0),
            );

            var color = vec3<f32>(
                decompress(compressed.x),
                decompress(compressed.y),
                decompress(compressed.z),
            );
            if params.demodulate != 0u {
                color *= surface.albedo + params.demodulation_offset;
            }
            textureStore(output, vec2<i32>(destination), vec4<f32>(color, 1.0));
        }
    }
}
