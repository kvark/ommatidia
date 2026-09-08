from pathlib import Path
p = Path('ommatidia-train/src/bin/noise-risk.rs')
s = p.read_text()
assert s.count('let score = |k|') == 1 or s.count('let score=|k|') == 1
s = s.replace('let score = |k|', 'let score = |k: usize|').replace('let score=|k|', 'let score=|k: usize|')
p.write_text(s)
