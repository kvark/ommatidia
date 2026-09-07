from pathlib import Path
p=Path('Cargo.toml')
s=p.read_text()
if 'serde_json =' not in s:
    s=s.replace('ron = "0.8"','ron = "0.8"\nserde_json = "1"')
p.write_text(s)
p=Path('ommatidia/src/transport/mod.rs')
s=p.read_text().replace('|p, c, i|', '|p: Plane, c: usize, i: usize|')
p.write_text(s)
