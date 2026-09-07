use ommatidia::transport::{oracle,CANDIDATES,SCALES};
include!("fixtures/transport.rs");
#[test]
fn exact_rgb_hull_projection_and_certificate(){
    let cases=[
        (vec![[0.0;3],[2.0,0.0,0.0]],[1.0,1.0,0.0],[1.0,0.0,0.0]),
        (vec![[0.0;3],[2.0,0.0,0.0],[0.0,2.0,0.0]],[0.5,0.5,1.0],[0.5,0.5,0.0]),
        (vec![[0.0;3],[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,2.0]],[0.2,0.3,0.4],[0.2,0.3,0.4]),
        (vec![[1.0;3],[1.0;3],[2.0;3]],[4.0;3],[2.0;3]),
        (vec![[0.0;3],[10000.0;3]],[5000.0;3],[5000.0;3]),
    ];
    for(points,target,expected)in cases{
        let p=oracle::project(&points,target).unwrap();
        for c in 0..3 {assert!((p.point[c]-expected[c]).abs()<1e-4);}
        assert!((p.weights.iter().sum::<f32>()-1.0).abs()<1e-6);assert!(p.weights.iter().all(|w|*w>=0.0));assert!(p.dual_gap<1e-7,"gap {}",p.dual_gap);
    }
    assert!(oracle::project(&[],[0.0;3]).is_err());assert!(oracle::project(&[[f32::NAN;3]],[0.0;3]).is_err());
}
#[test]
fn unavailable_history_is_not_an_oracle_candidate(){
    let mut input=oracle::Candidates{spatial:vec![2.0;SCALES*6],history:vec![0.0;6],prior:vec![1.0;CANDIDATES*2],selected:Vec::new()};
    input.prior[SCALES*2..].fill(0.0);let p=oracle::reconstruct(&input,&[0.0;6]).unwrap();
    assert_eq!(p.lobes,vec![2.0;6]);assert_eq!(&p.weights[SCALES*2..],&[0.0;2]);
    input.prior[SCALES*2..].fill(1.0);assert_eq!(oracle::reconstruct(&input,&[0.0;6]).unwrap().lobes,vec![0.0;6]);
}
#[test]
#[ignore="requires Vulkan or Metal"]
fn actual_native_candidates_match_selection_and_are_not_mutated(){
    use ommatidia::transport::native;
    let config=Config::default();let mut native=native::Native::new(ommatidia::gpu::create_context(None,false),config,[8,8]).unwrap();
    let(frame,target)=fixture(config,0,11);native.process(&frame).unwrap();let before=native.read_state();let input=native.read_candidates();
    let solution=oracle::reconstruct(&input,&target.lobes).unwrap();let mut image=vec![0.0;target.lobes.len()];native.session.read_output_by_index(0,&mut image);let n=frame.surfaces.len();
    for l in 0..2 {
        let error=(0..3*n).map(|j|(image[l*3*n+j]as f64-target.lobes[l*3*n+j]as f64).powi(2)).sum::<f64>()/(3*n)as f64;assert!(solution.lobe_mse[l]<=error+1e-5);
        for i in 0..n {for c in 0..3 {let j=(l*3+c)*n+i;
            let rebuilt=(0..CANDIDATES).map(|k|input.selected[k*2*n+l*n+i]*if k==SCALES {input.history[j]}else{input.spatial[k*6*n+j]}).sum::<f32>();
            assert!((rebuilt-image[j]).abs()<1e-5*(1.0+image[j].abs()));
        }}
    }
    for(a,b)in before.iter().zip(native.read_state()){assert_eq!(a.diffuse,b.diffuse);assert_eq!(a.specular,b.specular);}
}
