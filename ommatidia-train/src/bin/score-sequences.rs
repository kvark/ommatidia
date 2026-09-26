//! Re-score saved full-precision outputs, exposing per-sequence temporal failures.
//! No inference, checkpoint selection, image filtering, or GPU work is performed.
use ommatidia::{
    metrics, temporal,
    transport::{Config, Frame},
};
use ommatidia_train::{Result, open_capture};
use std::{
    io::Write,
    path::{Path, PathBuf},
};

type History = ([Vec<f32>; 2], Vec<f32>, Vec<temporal::Surface>);

fn decode(bytes: &[u8], values: usize) -> Result<Vec<f32>> {
    if bytes.len() != values * 4 {
        return Err("wrong full-precision image length".into());
    }
    let image: Vec<_> = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    if image.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err("nonfinite or negative radiance".into());
    }
    Ok(image)
}

fn load(directory: &Path, prefix: &str, role: &str, values: usize) -> Result<Vec<f32>> {
    decode(
        &std::fs::read(directory.join(format!("{prefix}-{role}.rgbf32")))?,
        values,
    )
}

#[derive(Default)]
struct Score {
    psnr: f64,
    temporal: f64,
    broad: f64,
    frames: usize,
    pairs: usize,
}
impl Score {
    fn report(&self) -> serde_json::Value {
        serde_json::json!({"frames": self.frames, "temporal_pairs": self.pairs,
            "psnr": self.psnr / self.frames as f64,
            "temporal_mse": (self.pairs > 0).then(|| self.temporal / self.pairs as f64),
            "temporal_block8_mse": (self.pairs > 0).then(|| self.broad / self.pairs as f64)})
    }
}

fn main() -> Result<()> {
    let mut benchmark = None;
    let mut before = None;
    let mut after = None;
    let mut out = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!(
                "score-sequences --benchmark FILE --before SAVED_OUTPUTS --after SAVED_OUTPUTS --out NEW_DIRECTORY\nReuses the evaluator's motion-compensated metric on every saved RGB f32 frame. An additional 8x8 block diagnostic separates broad fluctuations from grain; it is not a perceptual video pass. Record benchmark, ordered datasets and evaluation manifests as inputs."
            );
            return Ok(());
        }
        let value = PathBuf::from(args.next().ok_or("missing option value")?);
        match flag.as_str() {
            "--benchmark" => benchmark = Some(value),
            "--before" => before = Some(value),
            "--after" => after = Some(value),
            "--out" => out = Some(value),
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    let benchmark: serde_json::Value =
        serde_json::from_slice(&std::fs::read(benchmark.ok_or("--benchmark required")?)?)?;
    let before = before.ok_or("--before required")?;
    let after = after.ok_or("--after required")?;
    let out = out.ok_or("--out required")?;
    std::fs::create_dir(&out)?;
    let mut rows = std::fs::File::create(out.join("frames.csv"))?;
    writeln!(
        rows,
        "case,sequence,frame,before_psnr,after_psnr,before_temporal_mse,after_temporal_mse,before_temporal_block8_mse,after_temporal_block8_mse,temporal_values"
    )?;
    let mut summaries = Vec::new();
    let mut sequence = 0usize;
    for case in benchmark["datasets"].as_array().ok_or("missing datasets")? {
        let name = case["name"].as_str().ok_or("missing case name")?;
        let (mut reader, _) = open_capture(Path::new(
            case["path"].as_str().ok_or("missing dataset path")?,
        ))?;
        let layout = *reader.layout();
        let config = Config {
            scale: layout.scale,
            ..Config::default()
        };
        let length = reader.sequence_length();
        if benchmark["schema"] != 1
            || benchmark["sequence_length"] != length
            || case["sequences"] != reader.len() / length
            || benchmark["extent"]
                != serde_json::json!([
                    layout.lr_width * config.scale,
                    layout.lr_height * config.scale
                ])
        {
            return Err("benchmark and capture extents/sequences differ".into());
        }
        for start in (0..reader.len()).step_by(length) {
            let mut previous: Option<History> = None;
            let mut scores: [Score; 2] = Default::default();
            for index in 0..length {
                let frame = Frame::from_sample(&reader.sample(start + index)?, layout, config)?;
                let prefix = format!("{sequence:03}-{index:03}");
                let values = frame.surfaces.len() * 3;
                let reference = load(&before, &prefix, "reference", values)?;
                if reference != load(&after, &prefix, "reference", values)? {
                    return Err(format!("reference mismatch at {prefix}").into());
                }
                let images = [
                    load(&before, &prefix, "learned", values)?,
                    load(&after, &prefix, "learned", values)?,
                ];
                let surfaces: Vec<_> = frame
                    .surfaces
                    .iter()
                    .map(|s| temporal::Surface {
                        depth: s.normal_depth[3],
                        normal: s.normal_depth[..3].try_into().unwrap(),
                        albedo: s.albedo_roughness[..3].try_into().unwrap(),
                    })
                    .collect();
                let motion: Vec<_> = frame
                    .surfaces
                    .iter()
                    .flat_map(|s| s.motion[..2].iter().copied())
                    .collect();
                let psnr = images
                    .each_ref()
                    .map(|v| -10.0 * metrics::error(v, &reference).max(1e-20).log10() as f64);
                let mut temporal = [None; 2];
                let mut broad = [None; 2];
                if let Some((old, old_reference, old_surfaces)) = &previous {
                    let warp = temporal::Reprojection {
                        motion: &motion,
                        current: &surfaces,
                        previous: old_surfaces,
                        rejection: Default::default(),
                    };
                    for k in 0..2 {
                        temporal[k] = metrics::temporal_error(
                            [&images[k], &old[k]],
                            [&reference, old_reference],
                            warp,
                            None,
                            frame.low.map(|v| v as usize),
                            config.scale as usize,
                        );
                        broad[k] = metrics::temporal_low_frequency_error(
                            [&images[k], &old[k]],
                            [&reference, old_reference],
                            warp,
                            frame.low.map(|v| v as usize),
                            config.scale as usize,
                            8,
                        );
                    }
                }
                let cell = |v: Option<metrics::TemporalError>| {
                    v.map(|v| format!("{:.12}", v.mean())).unwrap_or_default()
                };
                writeln!(
                    rows,
                    "{name},{sequence},{index},{:.6},{:.6},{},{},{},{},{}",
                    psnr[0],
                    psnr[1],
                    cell(temporal[0]),
                    cell(temporal[1]),
                    cell(broad[0]),
                    cell(broad[1]),
                    temporal[0].map_or(0, |v| v.values)
                )?;
                for k in 0..2 {
                    scores[k].frames += 1;
                    scores[k].psnr += psnr[k];
                    if let Some(t) = temporal[k] {
                        scores[k].pairs += 1;
                        scores[k].temporal += t.mean();
                        scores[k].broad += broad[k]
                            .ok_or("missing block score for valid reprojection")?
                            .mean();
                    }
                }
                previous = Some((images, reference, surfaces));
            }
            summaries.push(serde_json::json!({"case": name, "sequence": sequence, "before": scores[0].report(), "after": scores[1].report()}));
            sequence += 1;
        }
    }
    rows.flush()?;
    let report = serde_json::json!({"metric_space": "fixed x/(1+x), before display conversion",
        "aggregation": "frame-mean, matching evaluator; resets excluded from temporal scores",
        "broad_diagnostic": "additional motion-compensated 8x8 block-change residual, not a predeclared spatial crop or perceptual video pass",
        "sequences": summaries});
    std::fs::write(
        out.join("quality.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn linear_reader_rejects_wrong_length_and_invalid_radiance() {
        assert_eq!(
            decode(&[1.25f32.to_le_bytes(), 0.0f32.to_le_bytes()].concat(), 2).unwrap(),
            [1.25, 0.0]
        );
        assert!(decode(&[0; 3], 1).is_err());
        for v in [-1.0f32, f32::NAN, f32::INFINITY] {
            assert!(decode(&v.to_le_bytes(), 1).is_err());
        }
    }
}
