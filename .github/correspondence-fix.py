from pathlib import Path

p=Path('ommatidia/tests/stereo.rs')
s=p.read_text().replace('serde_json::to_string','ron::to_string').replace('serde_json::from_str','ron::from_str')
p.write_text(s)
p=Path('ommatidia/src/field/stereo.rs')
s=p.read_text()
old='''                    if let Some(pixel)=peer.camera.project(position,c.extent) {
                        if projection.set(i,pixel,c.extent) { sweep.support[i]+=1.0; }
                    }'''
new='''                    if let Some(pixel)=peer.camera.project(position,c.extent)
                        && projection.set(i,pixel,c.extent) { sweep.support[i]+=1.0; }'''
assert old in s
p.write_text(s.replace(old,new))
