from pathlib import Path
p=Path('ommatidia-train/src/bin/fit.rs')
s=p.read_text().replace('use ommatidia::{gpu, neural::Network};', 'use ommatidia::{gpu, transport::graph::Network};')
s=s.replace('save_checkpoint(dir.join(', 'save_checkpoint(&dir.join(').replace('load_checkpoint(dir.join(', 'load_checkpoint(&dir.join(')
p.write_text(s)
p=Path('ommatidia-train/src/bin/fit/selector.rs')
s=p.read_text().replace('transport::{self, Config,', 'transport::{Config,')
p.write_text(s)
