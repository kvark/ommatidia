from pathlib import Path

p = Path('ommatidia/tests/field.rs')
s = p.read_text().replace('serde_json::to_string', 'ron::to_string').replace('serde_json::from_str', 'ron::from_str').replace('serde_json::to_value(observations(&config()))', 'ron::to_string(&observations(&config()))').replace('!obs.to_string().contains("incident")', '!obs.contains("incident")')
s = s.replace('for c in 0..3 {expected[c]+=trans*(1.0-keep)*emission[3*i+c];}', 'for (c, value) in expected.iter_mut().enumerate() {*value+=trans*(1.0-keep)*emission[3*i+c];}')
s = s.replace('for c in 0..3 {expected[c]+=trans*env[c];assert!((expected[c]-direct[3*r+c]).abs()<1e-5);}', 'for (c, value) in expected.iter_mut().enumerate() {*value+=trans*env[c];assert!((*value-direct[3*r+c]).abs()<1e-5);}')
p.write_text(s)
