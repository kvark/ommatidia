use std::{
    fs::{self, File},
    io::BufReader,
    path::PathBuf,
};

#[test]
fn readme_comparisons_share_the_output_extent_and_display_size() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let docs = workspace.join("docs");
    // The workspace owns these documentation assets; a standalone packaged
    // crate intentionally does not.
    if !docs.exists() {
        return;
    }
    let readme = fs::read_to_string(workspace.join("README.md")).unwrap();
    let check = |relative: &str| {
        let path = docs.join(relative);
        let decoder = png::Decoder::new(BufReader::new(File::open(&path).unwrap()));
        let reader = decoder.read_info().unwrap();
        assert_eq!(
            (reader.info().width, reader.info().height),
            (256, 256),
            "{} must match the README comparison extent",
            path.display()
        );

        let needle = format!(r#"<img src="docs/{relative}""#);
        let tag = readme
            .split_once(&needle)
            .unwrap_or_else(|| panic!("README has no explicit image tag for docs/{relative}"))
            .1
            .split_once('>')
            .expect("README image tag is not closed")
            .0;
        assert!(
            tag.contains(r#"width="256""#) && tag.contains(r#"height="256""#),
            "README must display docs/{relative} at 256x256"
        );
    };

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
            check(&format!("comparison-suite/{scene}/{method}"));
        }
    }

    for relative in [
        "temporal-low-frequency/hr-guided.png",
        "temporal-low-frequency/split-guided.png",
        "temporal-low-frequency/predicted.png",
        "temporal-low-frequency/reference.png",
        "lobe-scale-oracle/fixed.png",
        "lobe-scale-oracle/oracle-b8.png",
        "lobe-scale-oracle/reference.png",
        "direct-lobe-residual/fixed.png",
        "direct-lobe-residual/predicted.png",
        "direct-lobe-residual/reference.png",
    ] {
        check(relative);
    }
}
