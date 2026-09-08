from pathlib import Path

def change(s, old, new):
    assert s.count(old) == 1, (old[:100], s.count(old))
    return s.replace(old, new)

p = Path('ommatidia/src/transport/noise.rs')
s = p.read_text()
s = change(s, 'impl Regression {', 'impl Regression {\n    pub fn samples(&self) -> usize { self.count }')
s = change(s, 'let y = risk.ln_1p();', '// Predict expected linear squared risk, not expected log-risk.\n        let y = risk;')
s = change(s, 'r.add(x, (0.3 + 2.0 * x[1]).exp_m1()).unwrap();', 'r.add(x, 0.3 + 2.0 * x[1]).unwrap();')
p.write_text(s)
p = Path('ommatidia-train/src/bin/noise-risk.rs');s = p.read_text()
s = change(s, 'struct Capture {', '''const METHODS: [&str; 6] = ["learned", "fixed-prior", "observable-risk", "cross-noise-risk", "single-oracle", "convex-oracle"];
struct Capture {''')
s = change(s, 'fn write_png(', '''fn prior(s: &oracle::Candidates, p: &[[f32; 3]; CANDIDATES], l: usize, i: usize, n: usize) -> [f32; 3] {
    let total: f32 = (0..CANDIDATES).map(|k| s.prior[k*2*n+l*n+i]).sum();
    std::array::from_fn(|c| (0..CANDIDATES).map(|k| s.prior[k*2*n+l*n+i]*p[k][c]).sum::<f32>() / total)
}
fn write_png(''')
s = change(s, 'let mut snapshots = Vec::new();', 'let mut snapshots = Vec::new();\n        let mut native_images = Vec::new();\n        let mut native_parity = 0.0f64;\n        let mut max_dual_gap = 0.0f64;')
s = change(s, 'let mut frames = Vec::new();\n            for', 'let mut frames = Vec::new();\n            let mut native_frames = Vec::new();\n            for')
s = change(s, 'runtime.process(f)?;', 'native_frames.push(runtime.process(f)?);')
s = change(s, 'snapshots.push(frames);', 'snapshots.push(frames);\n            native_images.push(native_frames);')
s = change(s, 'let weights: Vec<_> = regressions.iter().map(|r| r.fit(1e-3).ok()).collect();', '''let weights: Vec<_> = regressions.iter().map(|r| {
            if r.samples() == 0 { Ok(None) } else { r.fit(1e-3).map(Some) }
        }).collect::<std::result::Result<_, String>>()?;''')
s = change(s, 'let mut errors = [[0.0f64; 5]; 2];', 'let mut errors = [[0.0f64; METHODS.len()]; 2];')
s = change(s, 'vec![vec![vec![0.0f32; 6 * n]; 5]; captures.len() - fit]', 'vec![vec![vec![0.0f32; 6 * n]; METHODS.len()]; captures.len() - fit]')
s = change(s, 'let predicted_choice = ids', 'max_dual_gap = max_dual_gap.max(optimum.dual_gap);\n                        let predicted_choice = ids')
s = change(s, 'selected(s, p, l, i, n),', 'selected(s, p, l, i, n),\n                            prior(s, p, l, i, n),')
a=s.index('                    let name = [');z=s.index('                    let psnr = ',a)
s=s[:a]+'''                    if m == 0 {
                        for (a,b) in rgb.iter().zip(&native_images[r+fit][f]) {
                            native_parity = native_parity.max((*a as f64-*b as f64).abs()/(1.0+(*b as f64).abs()));
                        }
                        if native_parity > 1e-5 { return Err(format!("candidate recomposition differs from native output: {native_parity}").into()); }
                    }
                    let name = METHODS[m];
'''+s[z:]
a=s.index('        for (m, name) in [');z=s.index('            methods.insert(',a)
s=s[:a]+'''        for (m, name) in METHODS.iter().enumerate() {
'''+s[z:]
s=change(s, '"spatial_bias2_population_variance_mse":risk_by_scale,"frames":per_frame', '"native_recomposition_max_relative_difference":native_parity,"max_oracle_dual_gap":max_dual_gap,"spatial_bias2_population_variance_mse":risk_by_scale,"frames":per_frame')
s=change(s, '"ridge":1e-3,', '"ridge":1e-3,"risk_target":"linear_lobe_mse",')
p.write_text(s)
