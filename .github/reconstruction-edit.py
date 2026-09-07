from pathlib import Path
p=Path('ommatidia-train/src/bin/transport.rs')
s=p.read_text().replace('type Result<T> =', 'type EvaluationHistory = ([Vec<f32>; 2], Vec<f32>, Vec<ommatidia::temporal::Surface>);\ntype Result<T> =')
s=s.replace('Option<([Vec<f32>; 2], Vec<f32>, Vec<ommatidia::temporal::Surface>)>', 'Option<EvaluationHistory>')
p.write_text(s)
p=Path('ommatidia-train/Cargo.toml');s=p.read_text()
for dep in ['bytemuck','half','log']:
    s=s.replace(f'{dep}.workspace = true\n','')
p.write_text(s)
