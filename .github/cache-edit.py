from pathlib import Path
import sys

def edit(name, old, new):
    p=Path(name); s=p.read_text(); assert s.count(old)==1, (name, old[:80])
    p.write_text(s.replace(old,new))

if sys.argv[1] == 'isolate':
    edit('ommatidia-data/src/main.rs',
         '        let cache = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/data-assets");',
         '''        let cache = std::env::var_os("OMMATIDIA_ASSET_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/data-assets"));''')
elif sys.argv[1] == 'fix':
    edit('ommatidia-data/src/main.rs',
         '''                let name = format!("{kind:?}{variant}.png").to_lowercase();
                let bytes = texture::bake(kind, seed ^ variant.wrapping_mul(0x9E37_79B9_7F4A_7C15));''',
         '''                let bytes = texture::bake(kind, seed ^ variant.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let name = texture::cache_name(&bytes);''')
    edit('ommatidia-data/src/texture.rs', '#[cfg(test)]', '''/// Blade's disk cache keys inline assets by name/metadata, not source bytes.
/// Content-dependent names prevent earlier captures from replacing this palette.
/// This non-cryptographic key is local cache identity, not dataset provenance.
pub fn cache_name(bytes: &[u8]) -> String {
    use std::hash::{Hash, Hasher};
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hash);
    format!("procedural-{:016x}.png", hash.finish())
}

#[cfg(test)]''')
    edit('ommatidia-data/src/texture.rs', '    use super::*;', '''    use super::*;

    #[test]
    fn inline_cache_identity_tracks_source_content() {
        let a = bake(Kind::Noise, 7);
        let b = bake(Kind::Noise, 20260907);
        assert_eq!(cache_name(&a), cache_name(&bake(Kind::Noise, 7)));
        assert_ne!(cache_name(&a), cache_name(&b));
        assert!(cache_name(&a).ends_with(".png"));
    }''')
else:
    raise ValueError('unknown phase')
