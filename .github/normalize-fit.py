from pathlib import Path

def edit(name, old, new):
    p=Path(name);s=p.read_text();assert s.count(old)==1,(name,old[:80],s.count(old));p.write_text(s.replace(old,new))

edit('ommatidia/src/transport/mod.rs', 'pub const FEATURES: usize = 44;', '''pub const FEATURES: usize = 44;
/// Positive selector multipliers must survive activation underflow.
pub const MIN_MULTIPLIER: f32 = 1e-8;''')
p=Path('ommatidia/src/transport/cpu.rs');s=p.read_text();assert s.count('.max(1e-8)')==2;s=s.replace('.max(1e-8)', '.max(MIN_MULTIPLIER)');p.write_text(s)
edit('ommatidia/src/transport/graph.rs', '        let multiplier = b.g.softplus(logits, 1.0);', '''        let multiplier = b.g.softplus(logits, 1.0);
        // Match the scalar reference. Softplus can round to zero for negative
        // logits; dividing zero weights by an added epsilon invents black.
        let floor = filled(&mut b.g, multiplier, MIN_MULTIPLIER);
        let negative_floor = b.g.neg(floor);
        let above_floor = b.g.add(multiplier, negative_floor);
        let above_floor = b.g.relu(above_floor);
        let multiplier = b.g.add(above_floor, floor);''')
edit('ommatidia/src/transport/graph.rs', '''        let eps = filled(&mut b.g, sum, 1e-12);
        sum = b.g.add(sum, eps);''', '''        let eps = filled(&mut b.g, sum, 1e-12);
        let negative_eps = b.g.neg(eps);
        let above_eps = b.g.add(sum, negative_eps);
        let above_eps = b.g.relu(above_eps);
        sum = b.g.add(above_eps, eps);''')
p=Path('ommatidia-train/src/bin/fit/selector.rs');s=p.read_text()
s=s.replace('let total = 1e-12\n                    + (0..CANDIDATES)', 'let total = (0..CANDIDATES)')
s=s.replace('let sp = z.max(0.0) + (-z.abs()).exp().ln_1p();', 'let sp = (z.max(0.0) + (-z.abs()).exp().ln_1p()).max(ommatidia::transport::MIN_MULTIPLIER as f64);')
s=s.replace('let dp = self.candidates.prior[j] as f64 / (1.0 + (-z).exp()) / den[l * n + i];', '''let active = z.max(0.0) + (-z.abs()).exp().ln_1p() > ommatidia::transport::MIN_MULTIPLIER as f64;
                    let dp = if active { self.candidates.prior[j] as f64 / (1.0 + (-z).exp()) / den[l * n + i] } else { 0.0 };''')
needle='        let z = output(&inf, 3, 2 * n * CANDIDATES);'
assert s.count(needle)==1
s=s.replace(needle,needle+'''
        let mass: Vec<f64> = (0..2*n).map(|i| (0..CANDIDATES).map(|k| weights[k*2*n+i] as f64).sum()).collect();
        let min_mass = mass.iter().copied().fold(f64::INFINITY, f64::min);
        let max_mass = mass.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if mass.iter().any(|v| !v.is_finite() || (v-1.0).abs()>2e-6) {
            return Err(format!("selector left the candidate hull: mass {min_mass}..{max_mass}").into());
        }
        report["normalization"] = json!({"minimum":min_mass,"maximum":max_mass});
        write(o.out.join(name).join("prediction.json"), &json!({"logits":z,"weights":weights,"lobes":image}))?;''')
p.write_text(s)
