// Only layout conversion, geometric bilinear tap construction and physical RGB.
// All denoising, history selection and latent recurrence live in the graph.
struct Params { w:u32, h:u32, scale:u32, state_channels:u32, ready:u32, exposure:f32, jitter:vec2<f32> }
struct Ray { diffuse:vec4<f32>, specular:vec4<f32>, normal_depth:vec4<f32> }
struct Surface { normal_depth:vec4<f32>, albedo_roughness:vec4<f32>, motion:vec4<f32>, specular_f0:vec4<f32>, emission:vec4<f32> }
var<uniform> params:Params;
var<storage> rays:array<Ray>;
var<storage> surfaces:array<Surface>;
var<storage> previous:array<f32>;
var<storage,read_write> features:array<f32>;
// Linear samples, [lobe RGB channel, LR pixel, 5x5 row-major tap].
var<storage,read_write> samples:array<f32>;
var<storage,read_write> history:array<f32>;
var<storage,read_write> metadata:array<f32>;
var<storage,read_write> valid:array<f32>;
var<storage,read_write> exposure:array<f32>;
var<storage,read_write> warp0:array<u32>;
var<storage,read_write> warp1:array<u32>;
var<storage,read_write> warp2:array<u32>;
var<storage,read_write> warp3:array<u32>;
var<storage,read_write> coeff0:array<f32>;
var<storage,read_write> coeff1:array<f32>;
var<storage,read_write> coeff2:array<f32>;
var<storage,read_write> coeff3:array<f32>;
var<storage> image:array<f32>;
var<storage,read_write> next:array<f32>;
var<storage,read_write> output:array<vec4<f32>>;
fn extent()->vec2<u32> {return vec2(params.w,params.h)*params.scale;}
fn idx(c:u32,p:vec2<u32>)->u32 {
    let slot=(p.y%params.scale)*params.scale+p.x%params.scale;
    return ((c*params.scale*params.scale+slot)*params.h+p.y/params.scale)*params.w+p.x/params.scale;
}
fn enc(v:f32,e:f32)->f32 {let x=v*e;return x/(1.0+x);}

@compute @workgroup_size(8,8)
fn pack(@builtin(global_invocation_id) id:vec3<u32>) {
    let p=id.xy;if any(p>=extent()) {return;}
    let lr=params.w*params.h;
    let s=surfaces[p.y*extent().x+p.x];
    if all(p==vec2(0u)) {exposure[0]=params.exposure;}
    if all(p%vec2(params.scale)==vec2(0u)) {
        let i=(p.y/params.scale)*params.w+p.x/params.scale;
        let r=rays[i];
        for(var tap=0u;tap<25u;tap++) {
            let q=clamp(vec2<i32>(p/params.scale)+vec2(i32(tap%5u)-2,i32(tap/5u)-2),
                        vec2(0),vec2<i32>(i32(params.w)-1,i32(params.h)-1));
            let neighbor=rays[u32(q.y)*params.w+u32(q.x)];
            for(var c=0u;c<3u;c++) {
                samples[(c*lr+i)*25u+tap]=neighbor.diffuse[c]*params.exposure;
                samples[((c+3u)*lr+i)*25u+tap]=neighbor.specular[c]*params.exposure;
            }
        }
        for(var c=0u;c<3u;c++) {
            features[c*lr+i]=enc(r.diffuse[c],params.exposure);
            features[(c+3u)*lr+i]=enc(r.specular[c],params.exposure);
            features[(c+6u)*lr+i]=r.normal_depth[c];
        }
        features[9u*lr+i]=enc(r.normal_depth.w,1.0);
        features[10u*lr+i]=params.jitter.x;
        features[11u*lr+i]=params.jitter.y;
    }
    for(var c=0u;c<4u;c++) {
        var nd=s.normal_depth[c];if c==3u {nd=enc(nd,1.0);}
        features[12u*lr+idx(c,p)]=nd;
        metadata[idx(c,p)]=nd;
        features[12u*lr+idx(c+4u,p)]=s.albedo_roughness[c];
    }
    for(var c=0u;c<3u;c++) {
        features[12u*lr+idx(c+8u,p)]=s.specular_f0[c];
        metadata[idx(c+4u,p)]=s.albedo_roughness[c];
    }
    for(var c=0u;c<2u;c++) {
        // No previous reconstruction exists on a hard reset. Large first-frame
        // renderer motion must not condition the cold spatial prediction.
        features[12u*lr+idx(c+11u,p)]=select(0.0,s.motion[c],(params.ready&1u)!=0u);
        features[12u*lr+idx(c+13u,p)]=f32(p[c]%params.scale)/f32(params.scale)+0.5/f32(params.scale)-0.5-params.jitter[c];
    }
    let q=vec2<f32>(p)+s.motion.xy;
    var positions:array<vec2<u32>,4>;
    var weights=vec4(0.0);
    if (params.ready&1u)!=0u && all(q>vec2(-1.0)) && all(q<vec2<f32>(extent())) {
        let base=vec2<i32>(floor(q));let f=q-vec2<f32>(base);
        for(var k=0u;k<4u;k++) {
            let at=base+vec2<i32>(i32(k%2u),i32(k/2u));
            if all(at>=vec2(0)) && all(at<vec2<i32>(extent())) {
                positions[k]=vec2<u32>(at);
                weights[k]=select(f.x,1.0-f.x,k%2u==0u)*select(f.y,1.0-f.y,k/2u==0u);
            }
        }
    }
    let total=weights.x+weights.y+weights.z+weights.w;
    if total>0.0 {weights/=total;}
    valid[idx(0u,p)]=select(0.0,1.0,total>0.0);
    for(var c=0u;c<params.state_channels;c++) {
        let i=idx(c,p);
        // Avoid even reading undefined device memory on the first frame/reset.
        // Bit 1 suppresses writes for later unroll slots, whose history lives
        // inside the graph. Bit 0 means the detached cursor state is valid.
        if (params.ready&2u)==0u {
            history[i]=0.0;if (params.ready&1u)!=0u {history[i]=previous[i];}
        }
        warp0[i]=idx(c,positions[0]);warp1[i]=idx(c,positions[1]);
        warp2[i]=idx(c,positions[2]);warp3[i]=idx(c,positions[3]);
        coeff0[i]=weights.x;coeff1[i]=weights.y;coeff2[i]=weights.z;coeff3[i]=weights.w;
    }
}

@compute @workgroup_size(8,8)
fn resolve(@builtin(global_invocation_id) id:vec3<u32>) {
    let p=id.xy;if any(p>=extent()) {return;}
    let i=p.y*extent().x+p.x;let s=surfaces[i];
    let d=vec3(image[idx(0u,p)],image[idx(1u,p)],image[idx(2u,p)]);
    let sp=vec3(image[idx(3u,p)],image[idx(4u,p)],image[idx(5u,p)]);
    for(var c=0u;c<params.state_channels;c++) {let j=idx(c,p);next[j]=image[j];}
    output[i]=vec4(d*s.albedo_roughness.xyz+sp+s.emission.xyz,1.0);
}
