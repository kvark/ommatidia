//! CPU-only admission check using the actual mapped training loader; no model evaluation.
use ommatidia::transport::Config;
use ommatidia_train::{Result, corpus::Corpus};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct FileIdentity {
    sha256: String,
}

#[derive(Deserialize)]
struct Dataset {
    path: PathBuf,
    sha256: String,
    bytes: usize,
    provenance: FileIdentity,
}

#[derive(Deserialize)]
struct Manifest {
    scenes: usize,
    frames: usize,
    datasets: Vec<Dataset>,
    pass: bool,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let path = args.next().ok_or("usage: check-corpus <corpus.json>")?;
    if args.next().is_some() {
        return Err("usage: check-corpus <corpus.json>".into());
    }
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(path)?)?;
    if !manifest.pass {
        return Err("corpus manifest did not pass admission".into());
    }
    let config = Config::default();
    let paths: Vec<_> = manifest.datasets.iter().map(|d| d.path.clone()).collect();
    let corpus = Corpus::open(&paths, config)?;
    if corpus.sequences.len() != manifest.scenes || corpus.len() != manifest.frames {
        return Err("manifest sequence/frame count mismatch".into());
    }
    for (actual, expected) in corpus.identities().iter().zip(&manifest.datasets) {
        if actual.sha256 != expected.sha256
            || actual.provenance_sha256 != expected.provenance.sha256
            || actual.bytes != expected.bytes
        {
            return Err(format!("manifest hash mismatch: {}", expected.path.display()).into());
        }
    }
    // The same full-reference finite/nonnegative/nonblack check and initialization
    // prior as training, without a GPU session or optimizer update.
    let means = corpus.means()?;
    if corpus.low.iter().any(|&n| n < 64) {
        return Err("admission probes require at least 64x64 input".into());
    }
    let max = corpus.low.map(|n| n - 64);
    let origins = [[0, 0], [max[0], 0], [0, max[1]], max, max.map(|n| n / 2)];
    let mut crops = 0;
    for sequence in 0..corpus.sequences.len() {
        for frame in [0, corpus.length / 2, corpus.length - 1] {
            for origin in origins {
                for gain in [0.25, 1.0, 4.0] {
                    let _ = corpus.crop(sequence, frame, origin, [64; 2], gain, config)?;
                    crops += 1;
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "pass": true, "scenes": corpus.sequences.len(), "frames": corpus.len(),
            "captures": corpus.captures.len(), "decoded_crops": crops,
            "lobe_means": means, "model_evaluations": 0, "optimizer_updates": 0,
        }))?
    );
    Ok(())
}
