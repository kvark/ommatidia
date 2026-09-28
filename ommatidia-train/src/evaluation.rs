//! Evaluation protocol and archived-control validation, independent of the model.
use crate::Result;
use std::{
    io::Write,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

pub fn age(frame: usize, reset_every: Option<NonZeroUsize>) -> usize {
    reset_every.map_or(frame, |interval| frame % interval.get())
}

pub fn bucket(age: usize) -> &'static str {
    match age {
        0 => "cold",
        1..=7 => "early",
        8..=15 => "settling",
        _ => "warm",
    }
}

pub fn decode_linear(bytes: &[u8], values: usize) -> Result<Vec<f32>> {
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

fn validate_rows(
    csv: &str,
    frames: usize,
    length: usize,
    interval: Option<NonZeroUsize>,
) -> Result<()> {
    let mut lines = csv.lines();
    let header: Vec<_> = lines
        .next()
        .ok_or("empty control CSV")?
        .split(',')
        .collect();
    let columns = ["sequence", "frame", "frames_since_reset"].map(|key| {
        header
            .iter()
            .position(|v| *v == key)
            .ok_or("control CSV lacks frame identity")
    });
    let [sequence, frame, since_reset] = [columns[0]?, columns[1]?, columns[2]?];
    for index in 0..frames {
        let row: Vec<_> = lines
            .next()
            .ok_or("control CSV is incomplete")?
            .split(',')
            .collect();
        let value = |column: usize| -> Result<usize> {
            Ok(row.get(column).ok_or("short control CSV row")?.parse()?)
        };
        if value(sequence)? != index / length
            || value(frame)? != index % length
            || value(since_reset)? != age(index % length, interval)
        {
            return Err("control CSV frame order/reset protocol differs".into());
        }
    }
    if lines.next().is_some() {
        return Err("control CSV has extra frames".into());
    }
    Ok(())
}

pub struct ControlRun {
    directory: PathBuf,
}
impl ControlRun {
    pub fn open(directory: &Path, expected: &serde_json::Value) -> Result<Self> {
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("quality.json"))?)?;
        // Provenance is an ordered list of captures, not an unordered set of seeds.
        for key in [
            "capture",
            "extent",
            "sequence_length",
            "frames",
            "reset_every",
        ] {
            if report.get(key).is_none() || report[key] != expected[key] {
                return Err(format!("control run {key} differs or is missing").into());
            }
        }
        if report["save_linear"] != true {
            return Err("control run requires --save-linear".into());
        }
        let interval = expected["reset_every"]
            .as_u64()
            .and_then(|v| NonZeroUsize::new(v as usize));
        validate_rows(
            &std::fs::read_to_string(directory.join("frames.csv"))?,
            expected["frames"].as_u64().ok_or("missing frame count")? as usize,
            expected["sequence_length"]
                .as_u64()
                .ok_or("missing sequence length")? as usize,
            interval,
        )?;
        Ok(Self {
            directory: directory.to_owned(),
        })
    }

    fn reference_bytes(&self, prefix: &str, reference: &[f32]) -> Result<Vec<u8>> {
        let saved = std::fs::read(self.directory.join(format!("{prefix}-reference.rgbf32")))?;
        let bytes: Vec<_> = reference.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Float equality would incorrectly accept -0.0 vs +0.0 and lose byte identity.
        if saved != bytes {
            return Err(format!("control reference differs byte for byte at {prefix}").into());
        }
        Ok(bytes)
    }

    /// Preserve byte-identical immutable references without duplicating their
    /// storage. Cross-filesystem/unsupported links fall back to a new copy.
    /// Neither path overwrites an existing destination or changes the control.
    pub fn save_reference(
        &self,
        prefix: &str,
        reference: &[f32],
        destination: &Path,
    ) -> Result<bool> {
        let bytes = self.reference_bytes(prefix, reference)?;
        if std::fs::hard_link(
            self.directory.join(format!("{prefix}-reference.rgbf32")),
            destination,
        )
        .is_ok()
        {
            return Ok(true);
        }
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)?;
        output.write_all(&bytes)?;
        Ok(false)
    }

    pub fn load(&self, prefix: &str, reference: &[f32]) -> Result<Vec<f32>> {
        self.reference_bytes(prefix, reference)?;
        decode_linear(
            &std::fs::read(self.directory.join(format!("{prefix}-learned.rgbf32")))?,
            reference.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_age_and_buckets_have_no_implicit_warm_frames() {
        let interval = NonZeroUsize::new(16);
        for frame in 0..64 {
            assert_eq!(age(frame, None), frame);
            assert_eq!(age(frame, interval), frame % 16);
            assert_ne!(bucket(age(frame, interval)), "warm");
        }
        assert_eq!(
            [0, 1, 7, 8, 15, 16].map(bucket),
            ["cold", "early", "early", "settling", "settling", "warm"]
        );
    }

    #[test]
    fn control_rows_reject_missing_extra_reordered_and_wrong_resets() {
        let valid = "sequence,frame,frames_since_reset\n0,0,0\n0,1,1\n1,0,0\n1,1,1\n";
        assert!(validate_rows(valid, 4, 2, None).is_ok());
        assert!(validate_rows(valid, 3, 2, None).is_err());
        assert!(validate_rows(valid, 5, 2, None).is_err());
        assert!(validate_rows(valid, 4, 2, NonZeroUsize::new(1)).is_err());
        assert!(validate_rows(&valid.replace("0,1,1", "1,0,0"), 4, 2, None).is_err());
    }

    #[test]
    fn linear_images_require_finite_nonnegative_values_and_exact_length() {
        for invalid in [f32::NAN, f32::INFINITY, -1.0] {
            assert!(decode_linear(&invalid.to_le_bytes(), 1).is_err());
        }
        assert!(decode_linear(&[0; 3], 1).is_err());
        assert_eq!(decode_linear(&1.0f32.to_le_bytes(), 1).unwrap(), [1.0]);
    }

    #[test]
    fn control_checks_order_protocol_and_reference_bytes() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("ommatidia-control-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let report = serde_json::json!({"capture":{"captures":[{"seed":1},{"seed":2}]},
            "extent":[1,1],"sequence_length":1,"frames":1,"reset_every":null,"save_linear":true});
        std::fs::write(
            dir.join("quality.json"),
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("frames.csv"),
            "sequence,frame,frames_since_reset\n0,0,0\n",
        )
        .unwrap();
        crate::save_linear(&dir.join("000-000-reference.rgbf32"), &[0.0, 1.0, 2.0]).unwrap();
        crate::save_linear(&dir.join("000-000-learned.rgbf32"), &[3.0, 4.0, 5.0]).unwrap();
        let control = ControlRun::open(&dir, &report).unwrap();
        assert_eq!(
            control.load("000-000", &[0.0, 1.0, 2.0]).unwrap(),
            [3.0, 4.0, 5.0]
        );
        assert!(control.load("000-000", &[-0.0, 1.0, 2.0]).is_err());
        let shared = dir.join("shared-reference.rgbf32");
        assert!(
            control
                .save_reference("000-000", &[-0.0, 1.0, 2.0], &shared)
                .is_err()
        );
        assert!(!shared.exists());
        control
            .save_reference("000-000", &[0.0, 1.0, 2.0], &shared)
            .unwrap();
        let expected = std::fs::read(dir.join("000-000-reference.rgbf32")).unwrap();
        assert_eq!(std::fs::read(&shared).unwrap(), expected);
        assert!(
            control
                .save_reference("000-000", &[0.0, 1.0, 2.0], &shared)
                .is_err()
        );
        assert_eq!(std::fs::read(&shared).unwrap(), expected);
        std::fs::remove_file(shared).unwrap();
        assert_eq!(
            std::fs::read(dir.join("000-000-reference.rgbf32")).unwrap(),
            expected
        );
        let mut changed = report.clone();
        changed["capture"]["captures"]
            .as_array_mut()
            .unwrap()
            .reverse();
        assert!(ControlRun::open(&dir, &changed).is_err());
        changed = report;
        changed["reset_every"] = serde_json::json!(16);
        assert!(ControlRun::open(&dir, &changed).is_err());
        for name in [
            "quality.json",
            "frames.csv",
            "000-000-reference.rgbf32",
            "000-000-learned.rgbf32",
        ] {
            std::fs::remove_file(dir.join(name)).unwrap();
        }
        std::fs::remove_dir(dir).unwrap();
    }
}
