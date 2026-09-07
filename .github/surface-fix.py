from pathlib import Path
p = Path('ommatidia-data/src/field_capture.rs')
s = p.read_text()
start = s.index('#[cfg(test)]')
end = s.index('/// Read f32 geometric distances', start)
p.write_text(s[:start] + s[end:] + '\n' + s[start:end])
