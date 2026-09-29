//! Complete training transactions: Adam, schedule, data identity, RNG and state.
use crate::{Result, corpus::Identity, sampler::Sampler, training::Trainer};
use ommatidia::transport::{Config, graph::LossWeights};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    pub steps: usize,
    pub batch: usize,
    pub unroll: usize,
    pub crop: [u32; 2],
    pub margin: u32,
    pub peak_rate: f32,
    pub seed: u64,
    pub weights: LossWeights,
    pub model: String,
}
#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    schema: u32,
    pub step: usize,
    pub settings: Settings,
    pub captures: Vec<Identity>,
    pub sampler: Sampler,
    pub means: [f64; 6],
    weights_sha256: String,
    state_sha256: String,
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

impl Checkpoint {
    pub fn save(
        path: &Path,
        trainer: &mut Trainer,
        settings: Settings,
        captures: Vec<Identity>,
        sampler: Sampler,
        means: [f64; 6],
    ) -> Result<()> {
        if path.exists() {
            return Err(format!("checkpoint already exists: {}", path.display()).into());
        }
        let parent = path.parent().ok_or("checkpoint needs a parent directory")?;
        std::fs::create_dir_all(parent)?;
        let staging = parent.join(format!(
            ".{}-{}.partial",
            path.file_name().unwrap().to_string_lossy(),
            std::process::id()
        ));
        std::fs::create_dir(&staging)?;
        let step = trainer.session.adam_step_count() as usize;
        if step == 0 || sampler.windows != (step * settings.batch) as u64 {
            return Err("optimizer/cursor checkpoint step mismatch".into());
        }
        trainer
            .session
            .save_checkpoint(&staging.join("model.safetensors"))?;
        std::fs::write(staging.join("model.transport.ron"), &settings.model)?;
        let state: Vec<_> = trainer
            .read_states()
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        if state
            .chunks_exact(4)
            .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite())
        {
            return Err("non-finite checkpoint state".into());
        }
        std::fs::write(staging.join("state.f32"), &state)?;
        let checkpoint = Self {
            schema: 4,
            step,
            settings,
            captures,
            sampler,
            means,
            weights_sha256: hash(&std::fs::read(staging.join("model.safetensors"))?),
            state_sha256: hash(&state),
        };
        std::fs::write(
            staging.join("trainer.json"),
            serde_json::to_vec_pretty(&checkpoint)?,
        )?;
        // Publish the bundle only after every component has been written.
        std::fs::rename(staging, path)?;
        Ok(())
    }
    pub fn restore(
        weights: &Path,
        trainer: &mut Trainer,
        settings: &Settings,
        captures: &[Identity],
    ) -> Result<Self> {
        let directory = weights.parent().ok_or("checkpoint has no directory")?;
        let metadata = directory.join("trainer.json");
        if !metadata.is_file() {
            return Err("true resume requires trainer.json, cursor state and Adam moments; weights-only warm starts are forbidden".into());
        }
        let saved: Self = serde_json::from_slice(&std::fs::read(metadata)?)?;
        if saved.schema != 4 {
            return Err("checkpoint training contract differs; F1 kernel/direct replacement requires fresh training".into());
        }
        let state = std::fs::read(directory.join("state.f32"))?;
        if &saved.settings != settings
            || saved.captures != captures
            || saved.step == 0
            || saved.step > settings.steps
            || saved.sampler.windows != (saved.step * settings.batch) as u64
            || saved.sampler.cursors.len() != settings.batch
            || saved.sampler.crop != settings.crop
            || saved.sampler.unroll != settings.unroll
            || saved.weights_sha256 != hash(&std::fs::read(weights)?)
            || saved.state_sha256 != hash(&state)
            || std::fs::read_to_string(directory.join("model.transport.ron"))? != settings.model
        {
            return Err("checkpoint settings, ordered captures, state, or weights mismatch".into());
        }
        let values = state
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        if !state.len().is_multiple_of(4) {
            return Err("truncated checkpoint state".into());
        }
        trainer.session.load_checkpoint(weights)?;
        if trainer.session.adam_step_count() as usize != saved.step {
            return Err("checkpoint lacks the matching Adam state".into());
        }
        trainer.restore_states(&values)?;
        Ok(saved)
    }
}

pub fn model_text(config: Config) -> Result<String> {
    Ok(ron::ser::to_string_pretty(
        &config,
        ron::ser::PrettyConfig::default(),
    )?)
}
