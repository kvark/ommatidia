// Scatter the network's sub-pixel output into the high resolution target.
//
// The GPU half of `ommatidia::batch::assemble`. Each invocation owns one
// *input* pixel and writes all `scale * scale` output texels that came from
// it, which is what makes the reindex free: the sub-pixel channels of one
// input pixel are exactly the block of output texels around it, so there is no
// gather and no interpolation anywhere in the upscale.
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
    compose_blade_radiance: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<uniform> params: UnpackParams;
var t_color: texture_2d<f32>;
var t_diffuse_radiance: texture_2d<f32>;
var t_specular_radiance: texture_2d<f32>;
var t_emissive: texture_2d<f32>;
var t_albedo: texture_2d<f32>;
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

@compute @workgroup_size(8, 8, 1)
fn unpack(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let texel = vec2<i32>(i32(id.x), i32(id.y));
    var low: vec3<f32>;
    if params.compose_blade_radiance != 0u {
        let albedo = textureLoad(t_albedo, texel, 0).xyz;
        let diffuse = textureLoad(t_diffuse_radiance, texel, 0).xyz;
        let specular = textureLoad(t_specular_radiance, texel, 0).xyz;
        let emissive = textureLoad(t_emissive, texel, 0).xyz;
        low = albedo * diffuse + specular + emissive;
    } else {
        low = textureLoad(t_color, texel, 0).xyz;
    }
    // The base is the same for every sub-pixel of this input pixel: nearest
    // neighbour, which is what the residual was defined against.
    let base = vec3<f32>(compress(low.x), compress(low.y), compress(low.z));

    let plane_stride = params.width * params.height;
    let offset = id.y * params.width + id.x;
    let sub = params.scale * params.scale;

    for (var dy = 0u; dy < params.scale; dy += 1u) {
        for (var dx = 0u; dx < params.scale; dx += 1u) {
            let slot = dy * params.scale + dx;
            var color: vec3<f32>;
            for (var c = 0u; c < 3u; c += 1u) {
                let channel = c * sub + slot;
                let delta = residual[channel * plane_stride + offset] * params.inverse_gain;
                color[c] = decompress(base[c] + delta);
            }
            let destination = vec2<i32>(
                i32(id.x * params.scale + dx),
                i32(id.y * params.scale + dy),
            );
            textureStore(output, destination, vec4<f32>(color, 1.0));
        }
    }
}
