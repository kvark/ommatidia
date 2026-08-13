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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<uniform> params: UnpackParams;
var<storage, read> base_pixels: array<f32>;
var<storage, read> residual: array<f32>;
var output: texture_storage_2d<rgba16float, write>;

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

fn reconstruction_base(destination: vec2<u32>, source: vec2<i32>) -> vec3<f32> {
    if params.reconstruction_base == 0u {
        return load_base(source);
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
