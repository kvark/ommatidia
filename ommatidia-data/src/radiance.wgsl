struct Params {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

var<uniform> params: Params;
var t_diffuse: texture_2d<f32>;
var t_specular: texture_2d<f32>;
var t_emissive: texture_2d<f32>;
var<storage, read_write> planes: array<f32>;

@compute @workgroup_size(8, 8)
fn probe(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.width || global_id.y >= params.height) {
        return;
    }
    let tc = vec2<i32>(global_id.xy);
    let diffuse = textureLoad(t_diffuse, tc, 0);
    let specular = textureLoad(t_specular, tc, 0);
    let emissive = textureLoad(t_emissive, tc, 0);
    let count = max(diffuse.w, 1.0);
    let index = global_id.y * params.width + global_id.x;
    let texels = params.width * params.height;
    for (var component = 0u; component < 3u; component += 1u) {
        planes[(component + 0u) * texels + index] = diffuse[component] / count;
        planes[(component + 3u) * texels + index] = specular[component] / count;
        planes[(component + 6u) * texels + index] = emissive[component] / count;
    }
}
