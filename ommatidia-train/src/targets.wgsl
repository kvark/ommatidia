struct Params { w:u32, h:u32, scale:u32, pad:u32 }
struct Surface { normal_depth:vec4<f32>, albedo_roughness:vec4<f32>, motion:vec4<f32>, specular_f0:vec4<f32>, emission:vec4<f32> }
var<uniform> params:Params;
var<storage> surfaces:array<Surface>;
var<storage> rgb:array<f32>;
var<storage,read_write> rgb_target:array<f32>;
var<storage,read_write> albedo:array<f32>;
var<storage,read_write> emission:array<f32>;
@compute @workgroup_size(8,8)
fn pack_targets(@builtin(global_invocation_id) id:vec3<u32>) {
    let extent=vec2(params.w,params.h)*params.scale;
    let p=id.xy;if any(p>=extent) {return;}
    let i=p.y*extent.x+p.x;
    let slot=(p.y%params.scale)*params.scale+p.x%params.scale;
    for(var c=0u;c<3u;c++) {
        let j=((c*params.scale*params.scale+slot)*params.h+p.y/params.scale)*params.w+p.x/params.scale;
        rgb_target[j]=rgb[3u*i+c];
        albedo[j]=surfaces[i].albedo_roughness[c];
        emission[j]=surfaces[i].emission[c];
    }
}
