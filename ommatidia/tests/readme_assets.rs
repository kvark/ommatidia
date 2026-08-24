use std::{fs::File, io::BufReader, path::PathBuf};

#[test]
fn primary_readme_comparisons_share_the_output_extent() {
    let docs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../docs/comparison-suite");
    // The workspace owns these documentation assets; a standalone packaged
    // crate intentionally does not.
    if !docs.exists() {
        return;
    }
    let scenes = ["canopy-shadow", "local-light", "hard-shadow"];
    let methods = [
        "bilinear.png",
        "ommatidium.png",
        "oidn-input-high.png",
        "restir-svgf.png",
        "canonical.png",
    ];

    for scene in scenes {
        for method in methods {
            let path = docs.join(scene).join(method);
            let decoder = png::Decoder::new(BufReader::new(File::open(&path).unwrap()));
            let reader = decoder.read_info().unwrap();
            assert_eq!(
                (reader.info().width, reader.info().height),
                (256, 256),
                "{} must match the README comparison extent",
                path.display()
            );
        }
    }
}
