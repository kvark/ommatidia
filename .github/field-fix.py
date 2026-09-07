from pathlib import Path

p = Path('ommatidia/src/field/graph.rs')
s = p.read_text()
s = s.replace('    let h=g.concat(a,b,rows as u32,ca as u32,cb as u32,1);', '''    // Channel operators use flat NCHW storage; retain explicit matrix boundaries
    // so their flat backward scatter reshapes before joining other gradients.
    let a=g.reshape(a,&[rows*ca]); let b=g.reshape(b,&[rows*cb]);
    let h=g.concat(a,b,rows as u32,ca as u32,cb as u32,1);''')
s = s.replace('    let total=g.node(x).ty.num_elements()/channels;', '''    let total=g.node(x).ty.num_elements()/channels;
    let x=g.reshape(x,&[total*channels]);''')
p.write_text(s)

p = Path('ommatidia-train/src/bin/field.rs')
s = p.read_text().replace('if let Some(held)=&held {if held.iter().any(|a|train.iter().any(|b|a.record.scene_seed==b.record.scene_seed)) {', 'if held.as_ref().is_some_and(|held|held.iter().any(|a|train.iter().any(|b|a.record.scene_seed==b.record.scene_seed))) {')
s = s.replace('return Err("--eval-data must contain unseen scene seeds, including across lighting variants".into());}}', 'return Err("--eval-data must contain unseen scene seeds, including across lighting variants".into());}')
p.write_text(s)

p = Path('ommatidia-data/src/main.rs')
s = p.read_text().replace('  --sequence-frames N       consecutive frames per scene [1]', '''  --sequence-frames N       consecutive frames per scene [1]
  --field-views N           static posed RGB orbit for field reconstruction, N>=3
  --scene-labels            export static procedural cameras and emitter labels
  --lighting-seed N         vary only source emission; requires scene labels''')
p.write_text(s)
