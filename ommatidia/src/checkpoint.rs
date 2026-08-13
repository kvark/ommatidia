//! Saving and loading a trained network.
//!
//! A checkpoint is two files: `<name>.safetensors` holds the weights, written
//! by meganeura, and `<name>.ron` holds the [`ModelConfig`] they were trained
//! for. The pair matters because meganeura bakes the tile extent, batch size,
//! and channel counts into a compiled plan — weights loaded into a graph built
//! from a different configuration would either fail to bind or, worse, bind to
//! the wrong tensors. The sidecar means inference never has to guess.

use std::path::{Path, PathBuf};

use crate::model::ModelConfig;

/// Paths to the two halves of a checkpoint.
#[derive(Clone, Debug)]
pub struct Paths {
    pub weights: PathBuf,
    pub config: PathBuf,
}

impl Paths {
    /// Derive both paths from a stem, ignoring any extension already on it.
    pub fn from_stem(stem: impl AsRef<Path>) -> Self {
        let stem = stem.as_ref();
        Self {
            weights: stem.with_extension("safetensors"),
            config: stem.with_extension("ron"),
        }
    }
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// The sidecar could not be parsed.
    Config(String),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Io(ref e) => write!(f, "{e}"),
            Self::Config(ref message) => write!(f, "malformed checkpoint config: {message}"),
        }
    }
}

impl std::error::Error for Error {}

/// Write the weights of `session` and the configuration beside them.
pub fn save(
    session: &mut meganeura::Session,
    config: &ModelConfig,
    stem: impl AsRef<Path>,
) -> Result<Paths, Error> {
    let paths = Paths::from_stem(stem);
    if let Some(parent) = paths.weights.parent() {
        std::fs::create_dir_all(parent)?;
    }
    session.save_checkpoint(&paths.weights)?;

    let pretty = ron::ser::PrettyConfig::new();
    let text =
        ron::ser::to_string_pretty(config, pretty).map_err(|e| Error::Config(e.to_string()))?;
    std::fs::write(&paths.config, text)?;
    Ok(paths)
}

/// Read back the configuration a checkpoint was trained with.
///
/// Load this first, build the graph from it, then hand the weights path to
/// [`meganeura::Session::load_checkpoint`].
pub fn load_config(stem: impl AsRef<Path>) -> Result<(ModelConfig, Paths), Error> {
    let paths = Paths::from_stem(stem);
    let text = std::fs::read_to_string(&paths.config)?;
    let config: ModelConfig = ron::from_str(&text)
        .map_err(|e| Error::Config(format!("{}: {e}", paths.config.display())))?;
    Ok((config, paths))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Plane, PlaneSet};
    use crate::model::{Backbone, Objective};

    #[test]
    fn paths_share_a_stem() {
        let paths = Paths::from_stem("runs/experiment.safetensors");
        assert_eq!(paths.weights.file_name().unwrap(), "experiment.safetensors");
        assert_eq!(paths.config.file_name().unwrap(), "experiment.ron");
        // A bare stem works the same way.
        assert_eq!(
            Paths::from_stem("runs/experiment").config,
            paths.config,
            "an extension on the stem should not change the pair"
        );
    }

    #[test]
    fn config_survives_the_round_trip() {
        let config = ModelConfig {
            scale: 3,
            tile: 32,
            batch: 2,
            cond_planes: PlaneSet::new().with(Plane::Color).with(Plane::Normal),
            base_channels: 48,
            level_multipliers: vec![1, 2],
            blocks_per_level: 1,
            objective: Objective::Direct,
            ..ModelConfig::default()
        };

        let dir = std::env::temp_dir().join("ommatidia-checkpoint");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.ron");
        let pretty = ron::ser::PrettyConfig::new();
        std::fs::write(&path, ron::ser::to_string_pretty(&config, pretty).unwrap()).unwrap();

        let (back, _) = load_config(dir.join("test")).unwrap();
        assert_eq!(back.scale, config.scale);
        assert_eq!(back.tile, config.tile);
        assert_eq!(back.cond_planes, config.cond_planes);
        assert_eq!(back.level_multipliers, config.level_multipliers);
        assert_eq!(back.objective, config.objective);
        // The derived quantities the graph builder relies on must match too.
        assert_eq!(back.in_channels(), config.in_channels());
        assert_eq!(back.target_len(), config.target_len());

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_missing_checkpoint_is_an_error_not_a_panic() {
        assert!(matches!(
            load_config("/nonexistent/ommatidia/checkpoint"),
            Err(Error::Io(_))
        ));
    }

    #[test]
    fn pre_attention_sidecars_default_to_the_convolutional_backbone() {
        let text = ron::ser::to_string(&ModelConfig::default()).unwrap();
        let marker = "backbone:Conv,";
        assert!(text.contains(marker), "unexpected RON shape: {text}");
        let old_text = text.replace(marker, "");
        let config: ModelConfig = ron::from_str(&old_text).unwrap();
        assert_eq!(config.backbone, Backbone::Conv);
    }
}
