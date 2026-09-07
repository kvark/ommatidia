from pathlib import Path
import subprocess

p=Path('Cargo.toml');s=p.read_text()
if 'serde_json =' not in s: s=s.replace('ron = "0.8"','ron = "0.8"\nserde_json = "1"')
p.write_text(s)
p=Path('ommatidia/src/transport/mod.rs');p.write_text(p.read_text().replace('|p, c, i|','|p: Plane, c: usize, i: usize|'))
p=Path('ommatidia/src/transport/prepare.wgsl');p.write_text(p.read_text().replace('fn filter(', 'fn atrous('))
p=Path('ommatidia/src/transport/native.rs');p.write_text(p.read_text().replace('"filter"', '"atrous"'))
p=Path('ommatidia/src/transport/graph.rs');s=p.read_text()
s=s.replace('fn compress(g:', '''fn filled(g:&mut Graph,x:NodeId,value:f32)->NodeId {
    let shape=g.node(x).ty.shape.clone();
    g.constant(vec![value;shape.iter().product()],&shape)
}
fn compress(g:''')
s=s.replace('self.g.scalar(0.1)', 'filled(&mut self.g,a,0.1)')
s=s.replace('let scale=g.scalar(e)', 'let scale=filled(g,x,e)')
s=s.replace('let one=g.scalar(1.0)', 'let one=filled(g,v,1.0)')
s=s.replace('let eps=b.g.scalar(1e-12)', 'let eps=filled(&mut b.g,sum,1e-12)')
p.write_text(s)
p=Path('ommatidia/src/transport/cpu.rs');s=p.read_text()
s=s.replace('for c in 0..2 {moments[c]+=w*old.moments[2*lobe+c];}', 'for (c,v) in moments.iter_mut().enumerate() {*v+=w*old.moments[2*lobe+c];}')
s=s.replace('for c in 0..3 {history[c]/=coverage;}', 'for v in &mut history {*v/=coverage;}')
s=s.replace('for c in 0..2 {moments[c]/=coverage;}', 'for v in &mut moments {*v/=coverage;}')
s=s.replace('for c in 0..3 {\n                let j=index(lobe*3+c,x,y);p.history[j]=history[c];', 'for (c,v) in history.iter().enumerate() {\n                let j=index(lobe*3+c,x,y);p.history[j]=*v;')
s=s.replace('for k in 0..SCALES {p.prior[k*2*n+index(lobe,x,y)]=(1.0-h)*prior[k]/total;}', 'for (k,v) in prior.iter().enumerate() {p.prior[k*2*n+index(lobe,x,y)]=(1.0-h)*v/total;}')
p.write_text(s)
p=Path('ommatidia-data/src/main.rs');s=p.read_text();a=s.index('fn translation_transform(');z=s.index('impl ActiveSequence',a);p.write_text(s[:a]+s[z:])
# Keep published inference compatibility, not two competing training programs.
for name in ['main.rs','batcher.rs','eval.rs']:
    Path('ommatidia-train/src',name).unlink(missing_ok=True)
p=Path('ommatidia-train/Cargo.toml');p.write_text(p.read_text().replace('default-run = "ommatidia-train"','default-run = "transport"'))
# Workflow updates are performed separately with the authorized GitHub tool;
# the Actions contents token intentionally cannot alter workflow definitions.
subprocess.run(['git','restore','--source=HEAD','--staged','--worktree','.github/workflows/check.yml'],check=True)
