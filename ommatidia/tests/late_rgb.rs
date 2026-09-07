use ommatidia::field::{self,data::{Prepared,RenderShape,Targets},graph,*};
use std::sync::Arc;
fn config()->Config{Config{extent:[8,8],views:2,channels:4,hidden:16,position_frequencies:1,view_fusion:ViewFusion::LateRgb,..Default::default()}}
fn observations(c:&Config)->Observations{Observations{bounds:Bounds{center:[0.0;3],radius:1.0},views:(0..c.views).map(|i|View{
    camera:Camera{origin:[i as f32*0.1,0.0,3.0],right:[1.0,0.0,0.0],up:[0.0,1.0,0.0],forward:[0.0,0.0,-1.0],tan_half_fov_y:0.5},rgb:vec![if i==0{2.0}else{6.0};8*8*3],
}).collect()}}
#[test]
fn common_parameters_keep_order_and_views_share_weights(){
    let c=config();let late=graph::build_points(&c,2).unwrap();let mut old=c.clone();old.view_fusion=ViewFusion::Moments;
    let old=graph::build_points(&old,2).unwrap();for(a,b)in old.params.iter().zip(&late.params){assert_eq!(a.name,b.name);assert_eq!(a.len,b.len);}
    assert!(late.params.len()>old.params.len());let mut three=c;three.views=3;let other=graph::build_points(&three,2).unwrap();assert_eq!(late.params.len(),other.params.len());
    assert!(late.graph.nodes().iter().all(|n|match &n.op{meganeura::graph::Op::Input{name}=>!name.starts_with("target."),_=>true}));
}
#[test]
#[ignore="requires Vulkan or Metal"]
fn linear_sources_direction_isolation_and_missing_coverage(){
    let c=config();let obs=observations(&c);let m=graph::build_points(&c,3).unwrap();let mut s=ommatidia::gpu::inference_session(&m.graph,ommatidia::gpu::create_context(None,false));m.initialize(&mut s,7);
    s.set_parameter("field.source.gate.bias",&[20.0]);s.set_parameter("field.emission.bias",&[-20.0;3]);s.set_parameter("field.emission.weight",&vec![0.0;c.hidden as usize*3]);
    let queries=[Query{position:[0.0;3],direction:[0.0,0.0,-1.0]},Query{position:[0.0;3],direction:[1.0,0.0,0.0]},Query{position:[0.0,0.0,4.0],direction:[0.0,0.0,-1.0]}];
    Prepared::new(&obs,&c,&queries).unwrap().feed(&mut s);s.step();s.wait();let sigma=s.read_output(3);assert!((sigma[0]-sigma[1]).abs()<1e-6);
    let mut rgb=vec![0.0;9];s.read_output_by_index(1,&mut rgb);for v in &rgb[..6]{assert!((*v-4.0).abs()<1e-4,"linear source blend {v}");}assert!(rgb.iter().all(|v|v.is_finite()&&*v>=0.0));
    let mut swapped=obs.clone();swapped.views.reverse();Prepared::new(&swapped,&c,&queries).unwrap().feed(&mut s);s.step();s.wait();let mut reordered=vec![0.0;9];s.read_output_by_index(1,&mut reordered);assert!(rgb.iter().zip(&reordered).all(|(a,b)|(a-b).abs()<1e-5));
    let mut changed=obs;for v in &mut changed.views{v.rgb.fill(1000.0);}Prepared::new(&changed,&c,&queries).unwrap().feed(&mut s);s.step();s.wait();s.read_output_by_index(1,&mut reordered);assert!(rgb[6..].iter().zip(&reordered[6..]).all(|(a,b)|(a-b).abs()<1e-5));
}
#[test]
#[ignore="requires Vulkan or Metal"]
fn late_rgb_training_updates_and_reloads_view_parameters(){
    let c=config();let obs=observations(&c);let shape=RenderShape{rays:2,steps:4,probes:0};let m=graph::build_render(&c,shape,true).unwrap();let context=ommatidia::gpu::create_context(None,false);let mut s=ommatidia::gpu::training_session(&m.graph,context.clone());m.initialize(&mut s,7);
    let rays=[obs.views[0].camera.ray([3.0,3.0],c.extent),obs.views[1].camera.ray([4.0,4.0],c.extent)];let(q,dt)=field::data::ray_queries(obs.bounds,&rays,shape.steps).unwrap();Prepared::new(&obs,&c,&q).unwrap().feed(&mut s);s.set_input("ray.deltas",&dt);
    Targets{rgb:vec![1.0;6],emission:vec![0.0;q.len()*3],emission_mask:vec![0.0;q.len()*3],environment:[0.1;3],environment_mask:[1.0;3]}.feed(&mut s);
    let mut first=0.0;let mut last=0.0;for step in 0..32{s.set_adam(0.001,0.9,0.999,1e-8);s.step();s.wait();last=s.read_loss();if step==0{first=last;}assert!(last.is_finite());}assert!(last<first,"{first} -> {last}");
    let mut weights=vec![0.0;c.hidden as usize];s.read_param("field.source.score.out.weight",&mut weights);assert!(weights.iter().any(|v|v.abs()>1e-6));
    let path=std::env::temp_dir().join(format!("late-rgb-{}.safetensors",std::process::id()));s.save_checkpoint(&path).unwrap();let inference=graph::build_render(&c,shape,false).unwrap();let mut restored=ommatidia::gpu::inference_session(&inference.graph,Arc::clone(&context));restored.load_checkpoint(&path).unwrap();Prepared::new(&obs,&c,&q).unwrap().feed(&mut restored);restored.set_input("ray.deltas",&dt);restored.step();restored.wait();assert!(restored.read_output(6).iter().all(|v|v.is_finite()));std::fs::remove_file(path).unwrap();println!("late RGB learning loss {first} -> {last}");
}
