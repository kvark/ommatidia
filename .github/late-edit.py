from pathlib import Path

def edit(path, old, new):
 p=Path(path);s=p.read_text();assert s.count(old)==1,(path,old[:80],s.count(old));p.write_text(s.replace(old,new))

edit('ommatidia/src/field/mod.rs','    pub exposure: f32,','    pub exposure: f32,\n    #[serde(default)]\n    pub view_fusion: ViewFusion,')
edit('ommatidia/src/field/mod.rs','            exposure: 1.0,','            exposure: 1.0,\n            view_fusion: ViewFusion::Moments,')
edit('ommatidia/src/field/mod.rs','#[derive(Clone, Debug, Serialize, Deserialize)]\n#[serde(deny_unknown_fields)]\npub struct Config','''/// Missing fields preserve historical moment-pooled checkpoints. LateRgb adds
/// appearance parameters and requires a newly trained matching checkpoint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ViewFusion {
    #[default]
    Moments,
    LateRgb,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config''')
edit('ommatidia/src/field/data.rs','#[derive(Clone, Debug, PartialEq)]\npub struct Prepared','''#[derive(Clone, Debug, PartialEq)]
pub struct SourceEvidence {
    pub rgb: Vec<f32>,
    pub direction: Vec<f32>,
    pub valid: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Prepared''')
edit('ommatidia/src/field/data.rs','    pub images: Vec<Vec<f32>>,','    pub images: Vec<Vec<f32>>,\n    pub sources: Vec<SourceEvidence>,')
edit('ommatidia/src/field/data.rs','            images: Vec::new(),','            images: Vec::new(),\n            sources: Vec::new(),')
edit('ommatidia/src/field/data.rs','            let mut indices: [Vec<u32>; 4]','''            let mut source = SourceEvidence {
                rgb: if config.view_fusion == ViewFusion::LateRgb { view.rgb.clone() } else { Vec::new() },
                direction: vec![0.0; 3*q], valid: vec![0.0; q],
            };
            let mut indices: [Vec<u32>; 4]''')
edit('ommatidia/src/field/data.rs','                result.coverage[i] += 1.0;','''                result.coverage[i] += 1.0;
                source.valid[i] = 1.0;
                let direction=unit(std::array::from_fn(|c|query.position[c]-view.camera.origin[c]));
                source.direction[3*i..3*i+3].copy_from_slice(&direction);''')
edit('ommatidia/src/field/data.rs','            result.images.push(image);','''            if config.view_fusion == ViewFusion::LateRgb { result.sources.push(source); }
            result.images.push(image);''')
edit('ommatidia/src/field/data.rs','            session.set_input(&format!("view{v}.rgb_rays"), &self.images[v]);','''            session.set_input(&format!("view{v}.rgb_rays"), &self.images[v]);
            if let Some(source)=self.sources.get(v) {
                session.set_input(&format!("view{v}.linear_rgb"), &source.rgb);
                session.set_input(&format!("view{v}.source_direction"), &source.direction);
                session.set_input(&format!("view{v}.valid"), &source.valid);
            }''')
edit('ommatidia/src/field/graph.rs','    let mut global = None;','    let mut global = None;\n    let mut views = Vec::new();')
edit('ommatidia/src/field/graph.rs','        let mut projected = None;','''        let mut projected = None;
        let mut color = None;
        let rgb = (c.view_fusion == super::ViewFusion::LateRgb)
            .then(||b.g.input(&format!("view{v}.linear_rgb"), &[n,3]));''')
edit('ommatidia/src/field/graph.rs','            let weight = b.g.broadcast_inner(weight, ch);','''            let scalar_weight = weight;
            let weight = b.g.broadcast_inner(weight, ch);''')
edit('ommatidia/src/field/graph.rs','            projected = add(&mut b.g, projected, feature);','''            projected = add(&mut b.g, projected, feature);
            if let Some(table)=rgb {
                let tap=b.g.embedding(indices,table);
                let weight=b.g.broadcast_inner(scalar_weight,3);
                let tap=b.g.mul(tap,weight);
                color=add(&mut b.g,color,tap);
            }''')
edit('ommatidia/src/field/graph.rs','        let projected = projected.unwrap();','''        let projected = projected.unwrap();
        if let Some(rgb)=color {
            views.push(ViewEvidence {
                features:projected, rgb,
                direction:b.g.input(&format!("view{v}.source_direction"),&[q,3]),
                valid:b.g.input(&format!("view{v}.valid"),&[q,1]),
            });
        }''')
edit('ommatidia/src/field/graph.rs','    let radiance = b.g.add(scattered, emission);','    let mut radiance = b.g.add(scattered, emission);')
edit('ommatidia/src/field/graph.rs','    let environment = b.g.softplus(environment, 1.0);','''    let environment = b.g.softplus(environment, 1.0);
    // Append parameters after all historical parameters so paired common
    // encoder/geometry/fallback initializations stay identical at the same seed.
    let scattered = if views.is_empty() { scattered } else {
        source_fusion(b,c,q,&views,[mean,var,latent,direction,emission,scattered])
    };
    if !views.is_empty() { radiance=b.g.add(scattered,emission); }''')
edit('ommatidia/src/field/graph.rs','struct Field {','''struct ViewEvidence {
    features: NodeId,
    rgb: NodeId,
    direction: NodeId,
    valid: NodeId,
}

/// Late appearance fusion, not an occlusion oracle. Keep each source until a
/// shared scoring MLP chooses its contribution. No target-direction-dependent
/// geometry, and missing source coverage falls back exactly.
fn source_fusion(b: &mut Builder, c: &Config, q: usize, views: &[ViewEvidence], inputs: [NodeId;6]) -> NodeId {
    let [mean,var,latent,direction,emission,fallback]=inputs;
    let ch=c.channels as usize;
    let mut rgb_sum=None;
    let mut weight_sum=None;
    for view in views {
        let mut feature=columns(&mut b.g,view.features,mean,q,ch,ch);
        feature=columns(&mut b.g,feature,var,q,2*ch,ch);
        feature=columns(&mut b.g,feature,latent,q,3*ch,c.hidden as usize);
        let exposed=scaled(&mut b.g,view.rgb,c.exposure);
        let one=constant(&mut b.g,exposed,1.0);
        let den=b.g.add(exposed,one);
        let encoded=b.g.div(exposed,den);
        let dim=3*ch+c.hidden as usize;
        feature=columns(&mut b.g,feature,encoded,q,dim,3);
        feature=columns(&mut b.g,feature,view.direction,q,dim+3,3);
        feature=columns(&mut b.g,feature,direction,q,dim+6,3);
        let score=b.linear(feature,"field.source.score.in",(dim+9) as u32,c.hidden);
        let score=b.g.silu(score);
        let score=b.linear(score,"field.source.score.out",c.hidden,1);
        let score=b.g.softplus(score,1.0);
        let weight=b.g.mul(score,view.valid);
        weight_sum=add(&mut b.g,weight_sum,weight);
        let weight=b.g.broadcast_inner(weight,3);
        let part=b.g.mul(view.rgb,weight);
        rgb_sum=add(&mut b.g,rgb_sum,part);
    }
    for p in &mut b.params {
        if p.name=="field.source.score.out.weight" { p.kind=crate::model::InitKind::Zeros; }
    }
    let weights=weight_sum.unwrap();
    let eps=constant(&mut b.g,weights,1e-8);
    let den=b.g.add(weights,eps);
    let support=b.g.div(weights,den);
    let den=b.g.broadcast_inner(den,3);
    let color=b.g.div(rgb_sum.unwrap(),den);
    // Source RGB already contains emission. Subtract before mixing scattered
    // light, since volume rendering adds the supervised emission exactly once.
    let negative=b.g.neg(emission);
    let color=b.g.add(color,negative);
    let color=b.g.relu(color);
    let gate_input=columns(&mut b.g,latent,direction,q,c.hidden as usize,3);
    let gate=b.linear(gate_input,"field.source.gate",c.hidden+3,1);
    bias(b,"field.source.gate",-2.0);
    for p in &mut b.params {
        if p.name=="field.source.gate.weight" { p.kind=crate::model::InitKind::Zeros; }
    }
    let gate=b.g.sigmoid(gate);
    let gate=b.g.mul(gate,support);
    let gate=b.g.broadcast_inner(gate,3);
    let one=constant(&mut b.g,gate,1.0);
    let negative=b.g.neg(gate);
    let rest=b.g.add(one,negative);
    let fallback=b.g.mul(fallback,rest);
    let source=b.g.mul(color,gate);
    b.g.add(fallback,source)
}

struct Field {''')
edit('ommatidia-train/src/bin/field.rs','    let mut stratified = false;','    let mut stratified = false;\n    let mut eval_checkpoint = None::<PathBuf>;')
edit('ommatidia-train/src/bin/field.rs','            "--eval-data" => eval = Some(PathBuf::from(v)),','''            "--eval-data" => eval = Some(PathBuf::from(v)),
            "--eval-checkpoint" => eval_checkpoint = Some(PathBuf::from(v)),
            "--view-fusion" => c.view_fusion = match v.as_str() {
                "moments"=>field::ViewFusion::Moments,
                "late-rgb"=>field::ViewFusion::LateRgb,
                _=>return Err("view fusion must be moments or late-rgb".into()),
            },''')
edit('ommatidia-train/src/bin/field.rs',r'--rate F --stratified\n',r'--rate F --stratified\n  --view-fusion moments|late-rgb --eval-checkpoint PATH\n')
edit('ommatidia-train/src/bin/field.rs','    c.validate()?;','''    if let Some(path)=&eval_checkpoint {
        c=serde_json::from_slice(&std::fs::read(path.with_file_name("model.field.json"))?)?;
        steps=0;
    }
    c.validate()?;''')
edit('ommatidia-train/src/bin/field.rs','data_files.is_empty() || steps == 0 ||','data_files.is_empty() || (steps == 0 && eval_checkpoint.is_none()) ||')
edit('ommatidia-train/src/bin/field.rs','    let model = if surface_weight.is_some() {','''    let model = if eval_checkpoint.is_some() { graph::build_diagnostics(&c,shape)?
    } else if surface_weight.is_some() {''')
edit('ommatidia-train/src/bin/field.rs','    let mut session = ommatidia::gpu::training_session(&model.graph, Arc::clone(&context));','''    let mut session = if eval_checkpoint.is_some() {
        ommatidia::gpu::inference_session(&model.graph, Arc::clone(&context))
    } else { ommatidia::gpu::training_session(&model.graph, Arc::clone(&context)) };''')
edit('ommatidia-train/src/bin/field.rs','    session.save_checkpoint(&out.join("model.safetensors"))?;','''    if eval_checkpoint.is_none() { session.save_checkpoint(&out.join("model.safetensors"))?; }
    let checkpoint=eval_checkpoint.clone().unwrap_or_else(||out.join("model.safetensors"));''')
p=Path('ommatidia-train/src/bin/field.rs');s=p.read_text();assert s.count('load_checkpoint(&out.join("model.safetensors"))?')==2
s=s.replace('load_checkpoint(&out.join("model.safetensors"))?', 'load_checkpoint(&checkpoint)?')
s=s.replace('"backend":backend,"quality_only":true,','"backend":backend,"quality_only":true,"eval_checkpoint":eval_checkpoint,"view_fusion":c.view_fusion,');p.write_text(s)
edit('ommatidia/src/transport/mod.rs','pub mod olat;','pub mod olat;\npub mod oracle;')
edit('ommatidia/src/transport/native.rs','    /// Only call after waiting for the resolve submission.','''    /// Offline readback after a completed resolve. No targets accepted; no
    /// history or parameters mutated. Reads actual GPU-prepared candidates.
    pub fn read_candidates(&self) -> super::oracle::Candidates {
        let n=(self.low[0]*self.low[1]*self.config.scale.pow(2)) as usize;
        let read=|name: &str,len| {
            let buffer=self.session.plan().input_buffers.iter().find(|(n,_)|n==name).unwrap().1;
            let mut values=vec![0.0;len];self.session.read_buffer(buffer,&mut values);values
        };
        let mut selected=vec![0.0;CANDIDATES*2*n];self.session.read_output_by_index(1,&mut selected);
        super::oracle::Candidates {
            spatial:read("f0.candidates",SCALES*6*n),history:read("f0.history",6*n),
            prior:read("f0.prior",CANDIDATES*2*n),selected,
        }
    }
    /// Only call after waiting for the resolve submission.''')
edit('ommatidia-train/src/bin/transport.rs','struct Corpus {','#[path = "transport/oracle_report.rs"]\nmod oracle_report;\n\nstruct Corpus {')
edit('ommatidia-train/src/bin/transport.rs','    out: &Path,\n) -> Result<serde_json::Value>', '    out: &Path,\n    candidate_oracle: bool,\n) -> Result<serde_json::Value>')
edit('ommatidia-train/src/bin/transport.rs','    let mut scores = [Score::default(), Score::default()];','    let mut oracle_frames=Vec::new();\n    let mut scores = [Score::default(), Score::default()];')
edit('ommatidia-train/src/bin/transport.rs','        for (name, image) in [','''        if candidate_oracle {
            let mut report=oracle_report::frame(learned,frame,target,config,out,&prefix)?;
            report["sequence"]=serde_json::json!(index/corpus.length);
            report["frame"]=serde_json::json!(index%corpus.length);oracle_frames.push(report);
        }
        for (name, image) in [''')
edit('ommatidia-train/src/bin/transport.rs','    scores.iter_mut().for_each(Score::finish);','''    if candidate_oracle {
        std::fs::write(out.join("candidate-oracle.json"),serde_json::to_vec_pretty(&oracle_frames)?)?;
    }
    scores.iter_mut().for_each(Score::finish);''')
edit('ommatidia-train/src/bin/transport.rs','    let mut eval_only = false;', '    let mut eval_only = false;\n    let mut candidate_oracle = false;')
edit('ommatidia-train/src/bin/transport.rs','        if arg == "--fixed-exposure-loss" {','        if arg == "--candidate-oracle" { candidate_oracle=true; continue; }\n        if arg == "--fixed-exposure-loss" {')
edit('ommatidia-train/src/bin/transport.rs','[--eval-only] [--fixed-exposure-loss]','[--eval-only] [--candidate-oracle] [--fixed-exposure-loss]')
edit('ommatidia-train/src/bin/transport.rs','evaluate(&holdout, config, &mut learned, &mut baseline, &out)?','evaluate(&holdout, config, &mut learned, &mut baseline, &out, candidate_oracle)?')
