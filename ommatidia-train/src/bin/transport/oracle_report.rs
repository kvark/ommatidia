//! Target-dependent offline diagnostic; never replaces production state.
use super::*;
use ommatidia::transport::{oracle,CANDIDATES,SCALES};
pub(super) fn frame(native:&native::Native,frame:&Frame,target:&Target,config:Config,out:&Path,prefix:&str)->Result<serde_json::Value> {
    let input=native.read_candidates();let solution=oracle::reconstruct(&input,&target.lobes)?;let n=frame.surfaces.len();
    let mut learned=vec![0.0;6*n];native.session.read_output_by_index(0,&mut learned);
    let value=|k:usize,j:usize|if k==SCALES {input.history[j]}else{input.spatial[k*6*n+j]};
    let mut fixed=vec![0.0;6*n];
    for l in 0..2 {for i in 0..n {
        let sum=(0..CANDIDATES).map(|k|input.prior[k*2*n+l*n+i]).sum::<f32>().max(1e-12);
        for c in 0..3 {let j=(3*l+c)*n+i;fixed[j]=(0..CANDIDATES).map(|k|input.prior[k*2*n+l*n+i]*value(k,j)/sum).sum();}
    }}
    let mse=|image:&[f32]|->[f64;2]{std::array::from_fn(|l|(0..3*n).map(|j|(image[l*3*n+j]as f64-target.lobes[l*3*n+j]as f64).powi(2)).sum::<f64>()/(3*n)as f64)};
    let signed=|image:&[f32]|->[f64;6]{std::array::from_fn(|c|(0..n).map(|i|image[c*n+i]as f64-target.lobes[c*n+i]as f64).sum::<f64>()/n as f64)};
    let shares=|weights:&[f32]|->Vec<[f64;CANDIDATES]>{(0..2).map(|l|std::array::from_fn(|k|(0..n).map(|i|weights[k*2*n+l*n+i]as f64).sum::<f64>()/n as f64)).collect()};
    let individual:Vec<_>=(0..CANDIDATES).map(|k|{
        let mut error=[0.0f64;2];let mut count=[0usize;2];
        for l in 0..2 {for i in 0..n {if input.prior[k*2*n+l*n+i]<=0.0 {continue;}count[l]+=1;
            for c in 0..3 {let j=(3*l+c)*n+i;error[l]+=(value(k,j)as f64-target.lobes[j]as f64).powi(2);}
        }}
        serde_json::json!({"available_pixels":count,"linear_lobe_mse":std::array::from_fn::<_,2,_>(|l|if count[l]>0 {Some(error[l]/(3*count[l])as f64)}else{None})})
    }).collect();
    let extent=frame.low.map(|v|v*config.scale);let width=extent[0]as usize;let mut rgb=vec![0.0;3*n];
    for i in 0..n {for c in 0..3 {
        let d=solution.lobes[config.index(frame.low,c,i%width,i/width)];let s=solution.lobes[config.index(frame.low,3+c,i%width,i/width)];
        rgb[3*i+c]=d*frame.surfaces[i].albedo_roughness[c]+s+frame.surfaces[i].emission[c];
    }}
    save_png(&out.join(format!("{prefix}-oracle.png")),&rgb,extent)?;
    Ok(serde_json::json!({"fixed_state":"actual learned-history GPU inputs; never oracle feedback",
        "objective":"independent linear RGB lobe squared distance; not final RGB PSNR",
        "fixed_lobe_mse":mse(&fixed),"learned_lobe_mse":mse(&learned),"oracle_lobe_mse":solution.lobe_mse,
        "max_dual_gap":solution.max_dual_gap,"signed_lobe_rgb_error":signed(&learned),
        "mean_prior":shares(&input.prior),"mean_learned_share":shares(&input.selected),"mean_oracle_share":shares(&solution.weights),
        "candidates":individual,"oracle_display_psnr":-10.0*(metrics::error(&rgb,&target.rgb)as f64).max(1e-20).log10()}))
}
