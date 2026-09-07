use ommatidia::transport::{Config,Frame,Ray,Surface,Target};
fn fixture(config:Config,step:usize,seed:u64)->(Frame,Target) {
    let low=[8,8];let width=(low[0]*config.scale) as usize;let n=width*width;
    let jitter=if step%2==0 {[-0.25,0.25]} else {[0.25,-0.25]};
    let mut rng=ommatidia::rng::Rng::new(seed+step as u64);
    let truth=|x:f32,y:f32| {
        let foreground=x>=4.0+step as f32 && x<8.0+step as f32 && y>3.0 && y<12.0;
        let mut s=Surface{normal_depth:[0.0,0.0,1.0,if foreground {0.5} else {1.0}],albedo_roughness:[if foreground {0.8} else {0.4},0.6,0.3,0.6],..Surface::default()};
        s.motion[0]=if foreground {-1.0} else {0.0};
        s.specular_motion=[-0.5,0.0,1.0,0.0];
        s.emission=[0.02,0.03,0.01,0.0];
        let diffuse=0.3+0.6*x/width as f32+0.1*(step as f32*0.2).sin();
        let spec=0.8*(-((x-11.0-step as f32*0.5).powi(2)+(y-8.0).powi(2))/8.0).exp();
        (s,[diffuse,0.9*diffuse,1.1*diffuse],[spec,0.8*spec,0.5*spec])
    };
    let mut frame=Frame{low,jitter,rays:Vec::new(),surfaces:Vec::new()};let mut target=Target{lobes:vec![0.0;6*n],rgb:vec![0.0;3*n]};
    for y in 0..width {for x in 0..width {let (s,d,sp)=truth(x as f32+0.5,y as f32+0.5);frame.surfaces.push(s);
        for c in 0..3 {target.lobes[config.index(low,c,x,y)]=d[c];target.lobes[config.index(low,3+c,x,y)]=sp[c];target.rgb[(y*width+x)*3+c]=d[c]*s.albedo_roughness[c]+sp[c]+s.emission[c];}
    }}
    for y in 0..low[1] {for x in 0..low[0] {let (s,d,sp)=truth((x as f32+0.5+jitter[0])*config.scale as f32,(y as f32+0.5+jitter[1])*config.scale as f32);
        let noise=-rng.uniform().max(1e-6).ln();let noise2=-rng.uniform().max(1e-6).ln();
        frame.rays.push(Ray{diffuse:[d[0]*noise,d[1]*noise,d[2]*noise,0.0],specular:[sp[0]*noise2,sp[1]*noise2,sp[2]*noise2,0.0],normal_depth:s.normal_depth,albedo_roughness:s.albedo_roughness});
    }}
    (frame,target)
}
