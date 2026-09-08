from pathlib import Path
p=Path('ommatidia-train/src/bin/fit.rs')
s=p.read_text().replace('use ommatidia::{gpu, neural::Network};', 'use ommatidia::{gpu, transport::graph::Network};')
p.write_text(s)
p=Path('ommatidia-train/src/bin/fit/selector.rs')
s=p.read_text().replace('transport::{self, Config,', 'transport::{Config,')
p.write_text(s)
