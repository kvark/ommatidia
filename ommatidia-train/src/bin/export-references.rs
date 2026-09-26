//! Export references for crop selection without loading or running a model.
use ommatidia::transport::{Config, Target};
use ommatidia_train::{Result, open_capture, save_linear, save_png};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut data = Vec::new();
    let mut out = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!(
                "export-references --data FILE [--data FILE ...] --out NEW_DIRECTORY\nExport the first, middle and last reference of every sequence. No model, checkpoint or GPU is used."
            );
            return Ok(());
        }
        let value = PathBuf::from(args.next().ok_or("missing option value")?);
        match flag.as_str() {
            "--data" => data.push(value),
            "--out" => out = Some(value),
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    if data.is_empty() {
        return Err("--data required".into());
    }
    let out = out.ok_or("--out required")?;
    std::fs::create_dir(&out)?;
    let mut sequence = 0usize;
    for path in data {
        let (mut reader, _) = open_capture(&path)?;
        let layout = *reader.layout();
        let config = Config {
            scale: layout.scale,
            ..Config::default()
        };
        let length = reader.sequence_length();
        let mut frames = vec![0, length / 2 - 1, length - 1];
        frames.dedup();
        for start in (0..reader.len()).step_by(length) {
            for &frame in &frames {
                let target = Target::from_sample(&reader.sample(start + frame)?, layout, config)?;
                let prefix = format!("{sequence:03}-{frame:03}-reference");
                save_png(
                    &out.join(format!("{prefix}.png")),
                    &target.rgb,
                    [layout.hr_width(), layout.hr_height()],
                )?;
                save_linear(&out.join(format!("{prefix}.rgbf32")), &target.rgb)?;
            }
            sequence += 1;
        }
    }
    Ok(())
}
