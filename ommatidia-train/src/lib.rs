//! Capture checks and output formats shared by the evaluation tools.
use ommatidia::dataset::{InputSource, Reader};
use std::{io::Write, path::Path};

pub mod checkpoint;
pub mod corpus;
pub mod evaluation;
pub mod sampler;
pub mod training;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn validate_capture(provenance: &serde_json::Value) -> Result<()> {
    if provenance["matching_path_depth"] != true
        || provenance["input_estimator"] != "independent-paths"
    {
        return Err("verified, matched independent-path captures are required".into());
    }
    if let Some(value) = provenance
        .get("minimum_catalog_visible_fraction")
        .filter(|v| !v.is_null())
    {
        let coverage = value
            .as_f64()
            .ok_or("invalid catalog coverage provenance")?;
        if !(0.01..=1.0).contains(&coverage) {
            return Err("catalog coverage below 1%; fix the asset, camera or driver".into());
        }
    }
    Ok(())
}

pub fn open_capture(path: &Path) -> Result<(Reader, serde_json::Value)> {
    let provenance =
        serde_json::from_slice(&std::fs::read(path.with_extension("transport.json"))?)?;
    validate_capture(&provenance)?;
    let reader = Reader::open(path)?;
    if reader.layout().lr_source != InputSource::PathTrace {
        return Err("independently path-traced inputs are required".into());
    }
    if provenance["records"].as_u64() != Some(reader.len() as u64) {
        return Err("provenance record count differs from dataset".into());
    }
    // A one-frame sequence is a valid cold-start fitting/evaluation probe.
    // Consumers requiring recurrence still check their unroll or benchmark length.
    if reader.is_empty() {
        return Err("a nonempty capture is required".into());
    }
    Ok((reader, provenance))
}

pub fn save_png(path: &Path, rgb: &[f32], extent: [u32; 2]) -> Result<()> {
    let bytes: Vec<_> = rgb
        .iter()
        .map(|&v| (ommatidia::transform::display(v) * 255.0).round() as u8)
        .collect();
    let mut encoder = png::Encoder::new(std::fs::File::create(path)?, extent[0], extent[1]);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&bytes)?;
    Ok(())
}

pub fn save_linear(path: &Path, rgb: &[f32]) -> Result<()> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    for value in rgb {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_frame_capture_is_valid_but_empty_or_mismatched_capture_is_not() {
        use ommatidia::dataset::{Layout, Plane, PlaneSet, Sample, Writer};
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "ommatidia-single-frame-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("capture.omd");
        let sidecar = path.with_extension("transport.json");
        let layout = Layout {
            lr_width: 4,
            lr_height: 4,
            scale: 2,
            lr_source: InputSource::PathTrace,
            lr_planes: PlaneSet::new().with(Plane::Color),
            hr_planes: PlaneSet::new().with(Plane::Color),
        };
        let mut writer = Writer::create(&path, layout).unwrap();
        writer
            .write(&Sample {
                lr: vec![Default::default(); layout.lr_len()],
                hr: vec![Default::default(); layout.hr_len()],
            })
            .unwrap();
        writer.finish().unwrap();
        let write_provenance = |records| {
            std::fs::write(
                &sidecar,
                serde_json::to_vec(&serde_json::json!({
                    "matching_path_depth": true,
                    "input_estimator": "independent-paths",
                    "records": records,
                }))
                .unwrap(),
            )
            .unwrap();
        };
        write_provenance(1);
        let (reader, _) = open_capture(&path).unwrap();
        assert_eq!(reader.len(), 1);
        assert_eq!(reader.sequence_length(), 1);
        drop(reader);
        write_provenance(2);
        assert!(open_capture(&path).is_err());
        Writer::create(&path, layout).unwrap().finish().unwrap();
        write_provenance(0);
        assert!(open_capture(&path).is_err());
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(sidecar).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn capture_quality_rejects_invalid_provenance_and_invisible_assets() {
        assert!(validate_capture(&serde_json::json!({})).is_err());
        let mut p =
            serde_json::json!({"matching_path_depth":true,"input_estimator":"independent-paths"});
        assert!(validate_capture(&p).is_ok());
        for coverage in [
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!(1.1),
            serde_json::json!("unknown"),
        ] {
            p["minimum_catalog_visible_fraction"] = coverage;
            assert!(validate_capture(&p).is_err());
        }
        p["minimum_catalog_visible_fraction"] = serde_json::json!(0.05);
        assert!(validate_capture(&p).is_ok());
    }
}
