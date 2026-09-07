//! Offline, target-dependent selector diagnostic. Never feeds recurrent state.
//! Projects each RGB lobe onto the convex closure of its available candidates.
//! A conditional linear-lobe error bound, not a recurrent or RGB-PSNR bound.
use super::{CANDIDATES, SCALES};

#[derive(Debug)]
pub struct Projection {
    pub point: [f32; 3],
    pub weights: Vec<f32>,
    pub squared_error: f64,
    /// First-order optimality certificate for one-half squared distance.
    pub dual_gap: f64,
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 { (0..3).map(|i| a[i] * b[i]).sum() }
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { std::array::from_fn(|i| a[i] - b[i]) }
fn solve(mut a: [[f64; 4]; 3], n: usize) -> Option<[f64; 3]> {
    let scale=a.iter().take(n).flat_map(|r|r.iter().take(n)).map(|v|v.abs()).fold(0.0f64,f64::max);
    for c in 0..n {
        let pivot=(c..n).max_by(|i,j|a[*i][c].abs().total_cmp(&a[*j][c].abs()))?;
        if a[pivot][c].abs()<=1e-12*scale {return None;}
        a.swap(c,pivot);let d=a[c][c];
        for j in c..=n {a[c][j]/=d;}
        for i in 0..n {if i==c {continue;} let d=a[i][c];for j in c..=n {a[i][j]-=d*a[c][j];}}
    }
    Some(std::array::from_fn(|i|if i<n {a[i][n]}else{0.0}))
}
/// Exhaust faces of up to four vertices. Singular faces are covered by their
/// lower-dimensional subsets. Report a dual gap rather than assume exactness.
pub fn project(candidates:&[[f32;3]],target:[f32;3])->Result<Projection,String> {
    if candidates.is_empty() || candidates.len()>CANDIDATES || candidates.iter().flatten().chain(&target).any(|v|!v.is_finite()) {
        return Err("oracle needs one to six finite RGB candidates and a finite target".into());
    }
    let points:Vec<_>=candidates.iter().map(|p|p.map(f64::from)).collect();let target=target.map(f64::from);
    let mut best=points[0];let mut error=dot(sub(best,target),sub(best,target));
    let mut weights=vec![0.0;points.len()];weights[0]=1.0;
    for mask in 1usize..(1<<points.len()) {
        let count=mask.count_ones()as usize;if count>4 {continue;}
        let ids:Vec<_>=(0..points.len()).filter(|i|mask&(1<<i)!=0).collect();let origin=points[ids[0]];let n=count-1;
        let basis:Vec<_>=ids[1..].iter().map(|i|sub(points[*i],origin)).collect();let mut a=[[0.0;4];3];
        for i in 0..n {for j in 0..n {a[i][j]=dot(basis[i],basis[j]);}a[i][n]=dot(basis[i],sub(target,origin));}
        let Some(x)=solve(a,n)else{continue;};
        let mut w=vec![1.0-x[..n].iter().sum::<f64>()];w.extend_from_slice(&x[..n]);
        if w.iter().any(|v|*v < -1e-9 || !v.is_finite()) {continue;}
        w.iter_mut().for_each(|v|*v=v.max(0.0));let sum=w.iter().sum::<f64>();w.iter_mut().for_each(|v|*v/=sum);
        let point=std::array::from_fn(|c|ids.iter().zip(&w).map(|(i,w)|w*points[*i][c]).sum());
        let e=dot(sub(point,target),sub(point,target));
        if e<error {best=point;error=e;weights.fill(0.0);for(i,w)in ids.iter().zip(w){weights[*i]=w as f32;}}
    }
    let gradient=sub(best,target);let gap=points.iter().map(|p|dot(gradient,sub(best,*p))).fold(0.0f64,f64::max);
    Ok(Projection{point:best.map(|v|v as f32),weights,squared_error:error,dual_gap:gap})
}
/// Snapshot of actual native GPU inputs and normalized selector weights.
pub struct Candidates {
    pub spatial:Vec<f32>,pub history:Vec<f32>,pub prior:Vec<f32>,pub selected:Vec<f32>,
}
pub struct Reconstruction {
    pub lobes:Vec<f32>,pub weights:Vec<f32>,pub lobe_mse:[f64;2],pub max_dual_gap:f64,
}
pub fn reconstruct(input:&Candidates,target:&[f32])->Result<Reconstruction,String> {
    let n=input.history.len()/6;
    if n==0 || input.history.len()!=6*n || input.spatial.len()!=SCALES*6*n || input.prior.len()!=CANDIDATES*2*n || target.len()!=6*n {
        return Err("invalid oracle candidate layout".into());
    }
    let mut result=Reconstruction{lobes:vec![0.0;6*n],weights:vec![0.0;CANDIDATES*2*n],lobe_mse:[0.0;2],max_dual_gap:0.0};
    for l in 0..2 {for i in 0..n {
        let ids:Vec<_>=(0..CANDIDATES).filter(|k|input.prior[k*2*n+l*n+i]>0.0).collect();
        let points:Vec<_>=ids.iter().map(|k|std::array::from_fn(|c|{let j=(3*l+c)*n+i;if *k==SCALES {input.history[j]}else{input.spatial[k*6*n+j]}})).collect();
        let truth=std::array::from_fn(|c|target[(3*l+c)*n+i]);let p=project(&points,truth)?;
        for c in 0..3 {result.lobes[(3*l+c)*n+i]=p.point[c];}
        for(k,w)in ids.into_iter().zip(p.weights){result.weights[k*2*n+l*n+i]=w;}
        result.lobe_mse[l]+=p.squared_error/(3*n)as f64;result.max_dual_gap=result.max_dual_gap.max(p.dual_gap);
    }}
    Ok(result)
}
