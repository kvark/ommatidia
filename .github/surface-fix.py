from pathlib import Path
p = Path('ommatidia-data/src/field_capture.rs')
s = p.read_text()
start = s.index('#[cfg(test)]')
end = s.index('///', s.index('pub fn surface_labels')) if False else s.index('/// Extract centred first-surface', start) if '/// Extract centred first-surface' in s[start:] else -1
# The newly appended helper starts after the test module; move the whole helper.
fn = s.index('pub fn surface_labels', start)
end = s.rfind('\n///', start, fn)
if end == -1:
    end = fn
else:
    end += 1
p.write_text(s[:start] + s[end:] + '\n' + s[start:end])
