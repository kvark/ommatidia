from pathlib import Path

p = Path('ommatidia/tests/field.rs')
s = p.read_text().replace('serde_json::to_string', 'ron::to_string').replace('serde_json::from_str', 'ron::from_str').replace('serde_json::to_value', 'ron::to_string').replace('!obs.to_string().contains("incident")', '!obs.contains("incident")')
p.write_text(s)
