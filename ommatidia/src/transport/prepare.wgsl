struct Params { w:u32, h:u32, scale:u32, level:u32, ready:u32, exposure:f32, diffuse_frames:f32, specular_frames:f32, jitter:vec2<f32>, pad:vec2<u32> }
struct Ray { diffuse:vec4<f32>, specular:vec4<f32>, normal_depth:vec4<f32>, albedo_roughness:vec4<f32> }
struct Surface { normal_depth:vec4<f32>, albedo_roughness:vec4<f32>, motion:vec4<f32>, specular_motion:vec4<f32>, emission:vec4<f32> }
struct State { diffuse:vec4<f32>, specular:vec4<f32>, moments:vec4<f32>, normal_depth:vec4<f32>, albedo_roughness:vec4<f32> }
var<uniform> params:Params;
var<storage> rays:array<Ray>;
var<storage> surfaces:array<Surface>;
var<storage> previous:array<State>;
var<storage,read_write> candidates:array<f32>;
var<storage,read_write> features:array<f32>;
var<storage,read_write> history:array<f32>;
var<storage,read_write> prior:array<f32>;
var<storage,read_write> moments:array<vec4<f32>>;
var<storage,read_write> ages:array<vec2<f32>>;
var<storage> image:array<f32>;
var<storage,read_write> next:array<State>;
var<storage,read_write> output:array<vec4<f32>>;
fn extent()->vec2<u32> {return vec2(params.w,params.h)*params.scale;}
fn count()->u32 {return params.w*params.h*params.scale*params.scale;}
fn idx(c:u32,p:vec2<u32>)->u32 {let slot=(p.y%params.scale)*params.scale+p.x%params.scale;return ((c*params.scale*params.scale+slot)*params.h+p.y/params.scale)*params.w+p.x/params.scale;}
fn geom(a:vec4<f32>,b:vec4<f32>)->f32 {
    if a.w>=60000.0 || b.w>=60000.0 {return select(0.0,1.0,a.w>=60000.0 && b.w>=60000.0);}
    let normal=max(dot(a.xyz,b.xyz),0.0);let d=abs(a.w-b.w)/(0.01+0.02*abs(a.w));
    return pow(normal,16.0)*exp(-d);
}
fn same(s:Surface,p:State)->bool {var expected=s.normal_depth;if s.motion.z>0.0 {expected.w=s.motion.z;}
    let d=s.albedo_roughness.xyz-p.albedo_roughness.xyz;return geom(expected,p.normal_depth)>0.1 && dot(d,d)<0.04;}
fn enc(v:f32)->f32 {let x=max(v,0.0)*params.exposure;return x/(1.0+x);}
fn read_rgb(level:u32,lobe:u32,p:vec2<u32>)->vec3<f32> {let o=level*6u*count();return vec3(candidates[o+idx(lobe*3u,p)],candidates[o+idx(lobe*3u+1u,p)],candidates[o+idx(lobe*3u+2u,p)]);}
fn lum(v:vec3<f32>)->f32 {return dot(v,vec3(0.2126,0.7152,0.0722));}
fn bounded(p:vec2<i32>)->vec2<u32> {return vec2<u32>(clamp(p,vec2(0),vec2<i32>(extent())-vec2(1)));}
@compute @workgroup_size(8,8)
fn seed(@builtin(global_invocation_id) id:vec3<u32>) {
    let p=id.xy;if any(p>=extent()) {return;}
    let s=surfaces[p.y*extent().x+p.x];let q=(vec2<f32>(p)+vec2(0.5))/f32(params.scale)-vec2(0.5)-params.jitter;
    var d=vec3(0.0);var spec=vec3(0.0);var total=0.0;
    for(var y=-1;y<=1;y++) {for(var x=-1;x<=1;x++) {
        let low=vec2<u32>(clamp(vec2<i32>(floor(q+vec2(0.5)))+vec2(x,y),vec2(0),vec2<i32>(i32(params.w)-1,i32(params.h)-1)));
        let r=rays[low.y*params.w+low.x];let delta=vec2<f32>(low)-q;
        let w=geom(s.normal_depth,r.normal_depth)*exp(-2.0*dot(delta,delta));d+=w*r.diffuse.xyz;spec+=w*r.specular.xyz;total+=w;
    }}
    if total<=1e-12 {let low=vec2<u32>(clamp(vec2<i32>(floor(q+vec2(0.5))),vec2(0),vec2<i32>(i32(params.w)-1,i32(params.h)-1)));let r=rays[low.y*params.w+low.x];d=r.diffuse.xyz;spec=r.specular.xyz;total=1.0;}
    for(var c=0u;c<3u;c++) {candidates[idx(c,p)]=max(d[c],0.0)/total;candidates[idx(3u+c,p)]=max(spec[c],0.0)/total;}
}
@compute @workgroup_size(8,8)
fn atrous(@builtin(global_invocation_id) id:vec3<u32>) {
    let p=id.xy;if any(p>=extent()) {return;}
    let s=surfaces[p.y*extent().x+p.x];let step=1 << (params.level-1u);
    for(var l=0u;l<2u;l++) {var sum=vec3(0.0);var total=0.0;
        for(var y=-1;y<=1;y++) {for(var x=-1;x<=1;x++) {
            let q=bounded(vec2<i32>(p)+vec2(x,y)*step);let t=surfaces[q.y*extent().x+q.x];
            var w=geom(s.normal_depth,t.normal_depth);if l==1u {w*=exp(-abs(s.albedo_roughness.w-t.albedo_roughness.w)*16.0);}
            w*=select(1.0,2.0,x==0)*select(1.0,2.0,y==0);sum+=w*read_rgb(params.level-1u,l,q);total+=w;
        }}
        for(var c=0u;c<3u;c++) {candidates[params.level*6u*count()+idx(l*3u+c,p)]=sum[c]/max(total,1e-12);}
    }
}
@compute @workgroup_size(8,8)
fn pack(@builtin(global_invocation_id) id:vec3<u32>) {
    let p=id.xy;if any(p>=extent()) {return;}let i=p.y*extent().x+p.x;let s=surfaces[i];
    var out_moments=vec4(0.0);var out_ages=vec2(1.0);
    for(var l=0u;l<2u;l++) {
        var motion=s.motion;if l==1u && s.specular_motion.z>0.5 {motion=s.specular_motion;}
        let q=vec2<f32>(p)+motion.xy;var old=vec3(0.0);var age=0.0;var m=vec2(0.0);var coverage=0.0;
        if params.ready!=0u && all(q>=vec2(0.0)) && all(q<=vec2<f32>(extent()-vec2(1u))) {
            let t=fract(q);
            for(var k=0u;k<4u;k++) {let at=min(vec2<u32>(floor(q))+vec2(k%2u,k/2u),extent()-vec2(1u));let st=previous[at.y*extent().x+at.x];
                var h=st.diffuse;var mm=st.moments.xy;if l==1u {h=st.specular;mm=st.moments.zw;}
                if h.w<=0.0 || !same(s,st) {continue;}
                let w=select(t.x,1.0-t.x,k%2u==0u)*select(t.y,1.0-t.y,k/2u==0u);old+=w*h.xyz;age+=w*h.w;m+=w*mm;coverage+=w;
            }
        }
        if coverage>1e-6 {old/=coverage;age/=coverage;m/=coverage;}
        let value=lum(read_rgb(0u,l,p));var variance=0.0;var samples=0.0;
        for(var y=-1;y<=1;y++) {for(var x=-1;x<=1;x++) {let at=bounded(vec2<i32>(p)+vec2(x,y)*i32(params.scale));let w=geom(s.normal_depth,surfaces[at.y*extent().x+at.x].normal_depth);let delta=lum(read_rgb(0u,l,at))-value;variance+=w*delta*delta;samples+=w;}}
        variance/=max(samples,1e-6);let broad=lum(read_rgb(4u,l,p));let delta=broad-lum(old);
        let v=variance+max(m.y-m.x*m.x,0.0)/max(age,1.0)+0.01*(1.0+broad*broad);
        let reactive=max(clamp(s.motion.w,0.0,1.0),clamp((delta*delta/v-4.0)/16.0,0.0,1.0));
        let max_age=select(params.diffuse_frames,params.specular_frames,l==1u);let retained=min(age,max_age-1.0)*coverage*(1.0-reactive);let h=retained/(1.0+retained);
        out_ages[l]=1.0+retained;out_moments[2u*l]=(1.0-h)*value+h*m.x;out_moments[2u*l+1u]=(1.0-h)*value*value+h*m.y;
        for(var c=0u;c<3u;c++) {history[idx(l*3u+c,p)]=old[c];}
        var weights=array<f32,5>(0.02,0.06,0.12,0.3,0.5);var total=0.0;
        for(var k=0u;k<5u;k++) {if l==1u {weights[k]*=exp(-f32(k)*(1.0-s.albedo_roughness.w)*2.0);}total+=weights[k];}
        for(var k=0u;k<5u;k++) {prior[k*2u*count()+idx(l,p)]=(1.0-h)*weights[k]/total;}
        prior[5u*2u*count()+idx(l,p)]=h;
        features[idx(38u+l,p)]=enc(sqrt(variance));features[idx(40u+l,p)]=out_ages[l]/max_age;features[idx(42u+l,p)]=select(0.0,1.0,coverage>1e-6);
    }
    ages[i]=out_ages;moments[i]=out_moments;
    for(var k=0u;k<5u;k++) {for(var c=0u;c<6u;c++) {features[idx(k*6u+c,p)]=enc(candidates[k*6u*count()+idx(c,p)]);}}
    for(var c=0u;c<3u;c++) {features[idx(30u+c,p)]=s.normal_depth[c];features[idx(34u+c,p)]=s.albedo_roughness[c];}
    features[idx(33u,p)]=1.0/(1.0+max(s.normal_depth.w,0.0));features[idx(37u,p)]=s.albedo_roughness.w;
}
@compute @workgroup_size(8,8)
fn resolve(@builtin(global_invocation_id) id:vec3<u32>) {
    let p=id.xy;if any(p>=extent()) {return;}let i=p.y*extent().x+p.x;let s=surfaces[i];
    let d=vec3(image[idx(0u,p)],image[idx(1u,p)],image[idx(2u,p)]);let sp=vec3(image[idx(3u,p)],image[idx(4u,p)],image[idx(5u,p)]);
    next[i]=State(vec4(d,ages[i].x),vec4(sp,ages[i].y),moments[i],s.normal_depth,s.albedo_roughness);
    output[i]=vec4(d*s.albedo_roughness.xyz+sp+s.emission.xyz,1.0);
}
