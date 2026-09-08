//! Recovery must not overwrite the original run's metadata, even via `..`.
#[test]
fn recovery_rejects_its_source_directory_before_writing() {
    let root = std::env::temp_dir().join(format!("field-recovery-{}", std::process::id()));
    std::fs::create_dir_all(root.join("child")).unwrap();
    for name in ["model.safetensors", "model.field.json", "loss.csv"] {
        std::fs::write(root.join(name), b"original run").unwrap();
    }
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_field"))
        .arg("--data")
        .arg("not-read.omd")
        .arg("--eval-checkpoint")
        .arg(root.join("model.safetensors"))
        .arg("--out")
        .arg(root.join("child/.."))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("separate output directory"));
    for name in ["model.safetensors", "model.field.json", "loss.csv"] {
        assert_eq!(std::fs::read(root.join(name)).unwrap(), b"original run");
    }
    std::fs::remove_dir_all(root).unwrap();
}
