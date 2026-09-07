//! The reconstruction network, as a meganeura graph.
//!
//! A timestep-conditioned U-Net that runs entirely at input resolution. The
//! deployed spatial path emits `3 * scale^2` sub-pixel residual channels. The
//! temporal experiment instead emits three denoised-colour residual channels
//! and lets the existing geometry-aware gather reconstruct output resolution.
//! See `docs/design.md` for why the network never touches output resolution.
//!
//! The same backbone serves both objectives. Under [`Objective::Diffusion`] it
//! takes a noised residual alongside the conditioning and predicts the noise;
//! under [`Objective::Direct`] it takes only the conditioning and predicts the
//! residual itself. Only the input channel count and the loss target differ.

use meganeura::{Graph, NodeId};
use serde::{Deserialize, Serialize};

use crate::dataset::{Plane, PlaneSet};
use crate::temporal;

/// What the network is trained to predict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Objective {
    /// e-prediction. Input carries a noised residual and a noise level, output
    /// is the noise. Sampling is iterative.
    Diffusion,
    /// Direct regression of the residual. One forward pass, no noise input.
    ///
    /// The fast path, and the baseline any distilled sampler has to beat.
    Direct,
}

/// Versioned normalization contract for crop-trained reconstruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backbone {
    /// Historical diffusion-derived blocks with image-wide GroupNorm.
    #[default]
    GroupNorm,
    /// Local convolutions/SiLU with 0.1-scaled residual branches and no
    /// image-wide statistics. Requires direct regression and new weights.
    Local,
}

/// Spatial quantity emitted by the network.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Prediction {
    /// Historical path: one residual for every output sub-pixel and RGB channel.
    SubpixelResidual,
    /// RGB correction at input resolution, followed by geometry-aware gather.
    LowResolutionResidual,
    /// A gather kernel over nearby input samples, one per output sub-pixel.
    ///
    /// The head predicts positive spatial weights, followed by optional guide
    /// and history gates. Convex weights constrain range, not estimator bias:
    /// sample-dependent weights and compressed-space losses can still darken
    /// radiance. `fusion` versions physical and candidate-aware mixing without
    /// changing the interpretation of released checkpoints.
    SubpixelKernel,
}

fn legacy_prediction() -> Prediction {
    Prediction::SubpixelResidual
}

/// Deterministic image reconstruction underneath the learned residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum ReconstructionBase {
    /// Historical v0.1 behavior. Kept only so old sidecars remain loadable.
    Nearest = 0,
    /// Texel-center-aligned bilinear filtering with clamp-to-edge addressing.
    Bilinear = 1,
    /// Geometry-guided low-resolution denoising followed by bilinear filtering.
    GuidedBilinear = 2,
    /// Low-resolution denoising followed by joint bilateral upsampling against
    /// a high-resolution primary-surface G-buffer.
    HighResolutionGuided = 3,
    /// No deterministic base at all: the network gathers the input samples
    /// itself. The only reconstruction that is a single operation.
    Sample = 4,
    /// Reconstruct renderer-provided diffuse, specular, and emissive radiance
    /// independently before predicting a correction to their composition.
    SplitRadianceGuided = 5,
}

fn legacy_reconstruction_base() -> ReconstructionBase {
    ReconstructionBase::Nearest
}

/// Parameters of the deterministic joint bilateral reconstruction.
///
/// They live in the checkpoint rather than only in WGSL so an updated runtime
/// cannot silently reinterpret older weights against a different base.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GuideConfig {
    pub spatial_sigma: f32,
    pub depth_sigma: f32,
    pub normal_power: f32,
    pub albedo_sigma: f32,
}

impl GuideConfig {
    /// The filter used by the published v0.2 and v0.3 checkpoints.
    pub const LEGACY: Self = Self {
        spatial_sigma: 3.0,
        depth_sigma: 0.05,
        normal_power: 32.0,
        albedo_sigma: 0.1,
    };

    /// Held-out-selected parameters with the same tap count and dispatches.
    pub const TUNED: Self = Self {
        spatial_sigma: 4.5,
        depth_sigma: 0.01,
        normal_power: 24.0,
        albedo_sigma: 0.2,
    };

    pub(crate) fn spatial_denominator(self) -> f32 {
        2.0 * self.spatial_sigma * self.spatial_sigma
    }

    pub(crate) fn depth_denominator(self) -> f32 {
        2.0 * self.depth_sigma * self.depth_sigma
    }

    pub(crate) fn albedo_denominator(self) -> f32 {
        2.0 * self.albedo_sigma * self.albedo_sigma
    }
}

fn legacy_guide_config() -> GuideConfig {
    GuideConfig::LEGACY
}

fn legacy_kernel_radius() -> u32 {
    2
}

fn legacy_demodulation_offset() -> f32 {
    0.25
}

fn legacy_head_kernel() -> u32 {
    3
}

/// How a parameter should be filled before training starts.
#[derive(Clone, Debug, PartialEq)]
pub enum InitKind {
    /// Kaiming normal, scaled by `sqrt(2 / fan_in)`, for weights behind SiLU.
    Kaiming {
        fan_in: usize,
    },
    Zeros,
    Ones,
    /// Exact starting values, for a parameter whose initial output has to be a
    /// particular function rather than a particular distribution.
    Values(Vec<f32>),
}

/// A parameter the graph declared, with enough information to initialise it.
#[derive(Clone, Debug)]
pub struct ParamInit {
    pub name: String,
    pub len: usize,
    pub kind: InitKind,
}

/// Shape of the network.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Output resolution is input resolution times this.
    pub scale: u32,
    /// Square input tile the network is compiled for.
    ///
    /// Compilation bakes in the extent, so training tiles and the runtime
    /// window have to agree. Must be divisible by `2^(levels - 1)`.
    pub tile: u32,
    pub batch: u32,
    /// Which dataset planes feed the conditioning, in storage order.
    pub cond_planes: PlaneSet,
    /// Channel width at the first level.
    pub base_channels: u32,
    /// Width multiplier per level; its length is the number of levels.
    pub level_multipliers: Vec<u32>,
    /// Residual blocks per level.
    pub blocks_per_level: usize,
    pub num_groups: u32,
    pub gn_eps: f32,
    /// Local reconstruction avoids crop/full-frame spatial-statistic drift.
    #[serde(default)]
    pub backbone: Backbone,
    /// Width of the sinusoidal timestep embedding the host computes.
    pub time_input_dim: u32,
    /// Width the timestep MLP projects to.
    pub time_embed_dim: u32,
    /// Factor the sub-pixel residual is multiplied by to reach unit variance.
    ///
    /// Measured from the training set with [`crate::batch::estimate_gain`] and
    /// carried in the checkpoint, because inference has to divide by exactly
    /// the value training multiplied by. See [`crate::batch::write_residual`]
    /// for why a diffusion model cannot be trained without it.
    pub residual_gain: f32,
    /// Weight of a relative-error term in residual training. A value of zero
    /// is ordinary compressed-space MSE; positive values keep dark surfaces
    /// from being sacrificed to small absolute error in bright regions.
    #[serde(default)]
    pub relative_loss_weight: f32,
    /// Weight of an additional MSE between 8x8 block-average residuals.
    ///
    /// Ordinary per-value MSE gives a coherent error across a smooth patch no
    /// more importance than the same energy spread across independent grain.
    /// This term matches the evaluator's 16x16-output low-frequency metric at
    /// 2x scale and specifically penalizes visible illumination mottling. The
    /// averaging lives only in the training graph, so it adds no parameters or
    /// inference work.
    #[serde(default)]
    pub low_frequency_loss_weight: f32,
    /// Maximum absolute correction in compressed radiance. Zero preserves the
    /// historical unbounded head; positive values apply `bound * tanh` before
    /// reconstruction, making residual checkpoints safe on dark surfaces.
    #[serde(default)]
    pub residual_bound: f32,
    pub objective: Objective,
    /// What the graph's output tensor represents.
    #[serde(default = "legacy_prediction")]
    pub prediction: Prediction,
    /// Image reconstruction to which the network adds its residual.
    ///
    /// Missing in v0.1 sidecars, whose weights were trained against nearest.
    #[serde(default = "legacy_reconstruction_base")]
    pub reconstruction_base: ReconstructionBase,
    /// Exact coefficients used by the CPU trainer and GPU reconstruction.
    #[serde(default = "legacy_guide_config")]
    pub guide: GuideConfig,
    /// Reconstruct illumination after dividing radiance by albedo, and
    /// multiply the exact output-resolution albedo back afterwards.
    ///
    /// The albedo is known exactly at output resolution, so a reconstruction
    /// that carries it through the filter is asking a network to recover
    /// something it was already told. Dividing it out first leaves the smoother
    /// illumination term to reconstruct and puts the texture back by
    /// multiplication. Standard in production denoisers, and measured on
    /// shadowed, textured scenes it takes the deterministic base from 47.5% to
    /// 65.4% detail retention — against 0.1 points on scenes whose materials
    /// are all one flat colour, which is why it looked worthless before.
    #[serde(default)]
    pub demodulate: bool,
    /// Blend the learned sample gather with the deterministic high-resolution
    /// guide, using one learned gate per output sub-pixel.
    ///
    /// The guide is deliberately outside the gather kernel: on flat surfaces
    /// it is a much lower-variance estimate, while on textured surfaces a
    /// gather over the validated accumulated samples preserves detail better.
    /// One extra head channel per output sub-pixel chooses between them.
    #[serde(default)]
    pub guide_mix: bool,
    /// Added to the albedo on both sides of a demodulated reconstruction.
    ///
    /// It bounds how far demodulation can rescale a pixel, and that bound is
    /// the whole ballgame: the gather runs in a compressed space tuned for
    /// radiance, and dividing by a small albedo moves a pixel somewhere that
    /// space has no precision left. Measured, 0.05 allows a 20x rescale and
    /// costs 1.5 dB against no demodulation at all; 0.25 allows 4x and gains
    /// 0.3 dB, 26% relative error, and sixteen points of detail.
    ///
    /// The same offset divides and multiplies, so a surface whose albedo does
    /// not change between input and output resolution comes back exactly and
    /// only the boundaries move. It also keeps an emissive surface, whose
    /// albedo is zero, from dividing by nothing.
    #[serde(default = "legacy_demodulation_offset")]
    pub demodulation_offset: f32,
    /// Kernel size of the output convolution.
    ///
    /// A kernel checkpoint's head is wide — 100 channels at radius two — so at
    /// 3x3 it is a quarter of the whole network's arithmetic. The features it
    /// reads already carry a large receptive field, so the spatial extent may
    /// be buying nothing; 1 makes that a measurement rather than an assumption.
    #[serde(default = "legacy_head_kernel")]
    pub head_kernel: u32,
    /// Weight of the temporal term in the training loss. Zero leaves it out.
    ///
    /// A per-frame squared error is indifferent to whether consecutive frames
    /// agree, so a reconstruction fitted to one will flicker whenever its inputs
    /// do — measured at 1.12 dB worse than deterministic accumulation overall
    /// and 3.19 dB worse on moving pixels, while every individual frame was
    /// 1.81 dB better. Stability is not a property of a single frame and does
    /// not appear in a single-frame objective.
    ///
    /// This is not carried by inference, but it is carried by the checkpoint,
    /// because it is part of how the weights came to be what they are.
    #[serde(default)]
    pub temporal_weight: f32,
    /// Extra weight the temporal loss gives a pixel per unit of motion.
    ///
    /// Zero weights every pixel with accepted history alike, which sounds fair
    /// and is not: moving pixels are 2.7% of them on these sequences, so they
    /// contribute 2.7% of the term while carrying all of the flicker.
    #[serde(default)]
    pub temporal_motion_bias: f32,
    /// Half-width, in input pixels, of the neighbourhood a
    /// [`Prediction::SubpixelKernel`] gathers from. Ignored by the other
    /// targets, and carried in the checkpoint because the runtime has to read
    /// exactly as many weight channels as training wrote.
    #[serde(default = "legacy_kernel_radius")]
    pub kernel_radius: u32,
    /// Average kernel taps in linear radiance before applying the bounded
    /// training transform.
    ///
    /// Historical checkpoints averaged `compress(radiance)`, which is not the
    /// convex radiance estimator their architecture promised: Jensen's
    /// inequality makes a wider filter systematically dark. Missing sidecar
    /// fields stay `false` so those weights retain their exact inference
    /// contract; newly trained kernel checkpoints opt in explicitly.
    #[serde(default)]
    pub linear_kernel: bool,
    /// Versioned fusion/interpolation contract. Missing sidecars retain the
    /// historical compressed-space behavior, even with linear gather taps.
    #[serde(default)]
    pub fusion: crate::fusion::Mode,
    /// Reprojected sparse samples consumed by this checkpoint. `None` keeps
    /// all existing single-frame sidecars and runtimes unchanged.
    #[serde(default)]
    pub temporal: Option<temporal::Config>,
}

impl Default for ModelConfig {
    /// The measured deployment baseline: a one-pass, 74k-parameter U-Net that
    /// stays within 0.03 dB of the 649k model on independent-path validation.
    fn default() -> Self {
        Self {
            scale: 2,
            tile: 64,
            batch: 8,
            cond_planes: PlaneSet::new()
                .with(Plane::Color)
                .with(Plane::Depth)
                .with(Plane::Normal)
                .with(Plane::DiffuseAlbedo)
                .with(Plane::SpecularF0)
                .with(Plane::Roughness),
            base_channels: 8,
            level_multipliers: vec![1, 2, 4],
            blocks_per_level: 1,
            num_groups: 8,
            gn_eps: 1e-5,
            backbone: Backbone::GroupNorm,
            time_input_dim: 64,
            time_embed_dim: 256,
            // Overwritten from the data; 1.0 leaves the residual as it is.
            residual_gain: 1.0,
            relative_loss_weight: 0.0,
            low_frequency_loss_weight: 0.0,
            residual_bound: 0.0,
            objective: Objective::Direct,
            prediction: Prediction::SubpixelResidual,
            reconstruction_base: ReconstructionBase::GuidedBilinear,
            guide: GuideConfig::TUNED,
            kernel_radius: legacy_kernel_radius(),
            linear_kernel: false,
            fusion: crate::fusion::Mode::Legacy,
            demodulate: false,
            guide_mix: false,
            demodulation_offset: legacy_demodulation_offset(),
            head_kernel: legacy_head_kernel(),
            temporal_weight: 0.0,
            temporal_motion_bias: 0.0,
            temporal: None,
        }
    }
}

impl ModelConfig {
    /// Number of levels in the U-Net.
    pub fn levels(&self) -> usize {
        self.level_multipliers.len()
    }

    /// Conditioning channels, from the plane set.
    pub fn cond_channels(&self) -> u32 {
        self.cond_planes.channels() as u32 + self.temporal_auxiliary_channels()
    }

    /// Extra conditioning a temporal checkpoint appends after the stored
    /// planes. Zero when there is no history.
    pub fn temporal_auxiliary_channels(&self) -> u32 {
        let base = match self.temporal {
            None => 0,
            Some(temporal) if self.prediction == Prediction::SubpixelKernel => {
                temporal.gather_auxiliary_channels()
            }
            Some(temporal) => temporal.auxiliary_channels(),
        };
        base + self
            .temporal
            .map_or(0, |temporal| temporal.phase_channels(self.scale))
    }

    /// Reprojected-history taps the gather reads, beyond the spatial ones.
    ///
    /// Accumulated-sample history still arrives as gather taps. Previous-output
    /// history does not: it is mixed with the spatial gather after it, one
    /// gate per output sub-pixel, so the head does not emit a second full
    /// kernel over the past.
    pub fn history_taps(&self) -> u32 {
        match self.temporal {
            Some(temporal) if self.prediction == Prediction::SubpixelKernel => {
                if temporal.previous_output {
                    0
                } else {
                    1 + u32::from(temporal.unrejected_tap)
                }
            }
            _ => 0,
        }
    }

    /// Mix gates for previous-output history, one per output sub-pixel.
    ///
    /// Zero unless this checkpoint blends the warped previous reconstruction
    /// after the spatial gather. Not counted in [`Self::gather_taps`].
    pub fn history_mix_channels(&self) -> u32 {
        match self.temporal {
            Some(temporal)
                if self.prediction == Prediction::SubpixelKernel && temporal.previous_output =>
            {
                self.scale * self.scale * self.fusion.gate_parameters()
            }
            _ => 0,
        }
    }

    /// Mix gates for the deterministic high-resolution guide, one per output
    /// sub-pixel. Not counted in [`Self::gather_taps`].
    pub fn guide_mix_channels(&self) -> u32 {
        if self.guide_mix && self.prediction == Prediction::SubpixelKernel {
            self.scale * self.scale * self.fusion.gate_parameters()
        } else {
            0
        }
    }

    /// Every tap the gather reads.
    pub fn gather_taps(&self) -> u32 {
        self.taps() + self.history_taps()
    }

    /// Input samples one output sub-pixel gathers from.
    pub fn taps(&self) -> u32 {
        let width = 2 * self.kernel_radius + 1;
        width * width
    }

    /// Offset of tap `index`, in input pixels, as `(dx, dy)`.
    ///
    /// The one definition of the tap order. `batch::write_taps`, the CPU
    /// gather, and `unpack.wgsl` all walk it, and they only agree because they
    /// all agree with this.
    pub fn tap_offset(&self, index: u32) -> (i32, i32) {
        let width = 2 * self.kernel_radius + 1;
        let radius = self.kernel_radius as i32;
        (
            (index % width) as i32 - radius,
            (index / width) as i32 - radius,
        )
    }

    /// Sub-pixel `(dx, dy)` of slot `slot`.
    pub fn sub_pixel(&self, slot: u32) -> (u32, u32) {
        (slot % self.scale, slot / self.scale)
    }

    /// Output channels for the selected prediction target.
    pub fn target_channels(&self) -> u32 {
        match self.prediction {
            Prediction::SubpixelResidual => 3 * self.scale * self.scale,
            Prediction::LowResolutionResidual => 3,
            // Spatial gather weights, sub-pixel major, plus one mix gate per
            // sub-pixel when previous-output history is blended after the gather.
            Prediction::SubpixelKernel => {
                self.scale * self.scale * self.gather_taps()
                    + self.guide_mix_channels()
                    + self.history_mix_channels()
            }
        }
    }

    /// Channels of the assembled image, which for a kernel checkpoint is not
    /// what the network emits.
    pub fn image_channels(&self) -> u32 {
        3 * self.scale * self.scale
    }

    /// Channels the loss target carries.
    ///
    /// The same as [`Self::target_channels`] everywhere except kernel
    /// prediction, where the network emits weights and the loss sits on the
    /// image those weights gathered.
    pub fn loss_channels(&self) -> u32 {
        match self.prediction {
            Prediction::SubpixelKernel => self.image_channels(),
            _ => self.target_channels(),
        }
    }

    /// Elements in one batch of loss targets.
    pub fn loss_len(&self) -> usize {
        (self.batch * self.loss_channels() * self.tile * self.tile) as usize
    }

    /// Elements in one batch of gather taps, empty unless this is a kernel
    /// checkpoint.
    pub fn tap_len(&self) -> usize {
        match self.prediction {
            Prediction::SubpixelKernel => {
                (self.batch * 3 * self.gather_taps() * self.tile * self.tile) as usize
            }
            _ => 0,
        }
    }

    /// Channels the first convolution consumes.
    pub fn in_channels(&self) -> u32 {
        match self.objective {
            Objective::Diffusion => self.target_channels() + self.cond_channels(),
            Objective::Direct => self.cond_channels(),
        }
    }

    /// Channel width at `level`.
    pub fn channels_at(&self, level: usize) -> u32 {
        self.base_channels * self.level_multipliers[level]
    }

    /// Estimate the convolution arithmetic for one frame, in GFLOP.
    ///
    /// Normalisation and activation are not counted even though they matter to
    /// measured frame time. This is therefore a floor on cost, useful for
    /// ruling a configuration out rather than predicting its runtime.
    ///
    /// `output_pixels` lets a configuration compiled for one tile be costed at
    /// the extent it would actually run at.
    pub fn flops(&self, output_pixels: usize) -> f64 {
        // The network runs at input resolution; the tile it was compiled for
        // is irrelevant to the cost per output pixel.
        let input_pixels = output_pixels as f64 / (self.scale * self.scale) as f64;
        let mut total = 0.0;
        // Two multiply-accumulates per tap, k*k taps.
        let conv = |pixels: f64, cin: u32, cout: u32, k: u32| {
            2.0 * pixels * cin as f64 * cout as f64 * (k * k) as f64
        };

        let mut pixels = input_pixels;
        let mut channels = self.base_channels;
        total += conv(pixels, self.in_channels(), self.base_channels, 3); // stem

        // Encoder, then the downsample that follows every level but the last.
        let levels = self.levels();
        let mut skips = Vec::new();
        for level in 0..levels {
            let width = self.channels_at(level);
            for _ in 0..self.blocks_per_level {
                total += conv(pixels, channels, width, 3) + conv(pixels, width, width, 3);
                if channels != width {
                    total += conv(pixels, channels, width, 1); // residual projection
                }
                channels = width;
            }
            if level + 1 < levels {
                skips.push((pixels, channels));
                pixels /= 4.0; // half in each axis
                total += conv(pixels, width, width, 3); // strided downsample
            }
        }

        // Two middle residual blocks, each with two 3x3 convolutions.
        for _ in 0..2 {
            total += conv(pixels, channels, channels, 3) * 2.0;
        }

        // Decoder: upsample, concatenate the skip, then narrow back down.
        for level in (0..levels.saturating_sub(1)).rev() {
            let (skip_pixels, skip_channels) = skips.pop().expect("a skip per level");
            pixels = skip_pixels;
            channels += skip_channels;
            let width = self.channels_at(level);
            for _ in 0..self.blocks_per_level {
                total += conv(pixels, channels, width, 3) + conv(pixels, width, width, 3);
                if channels != width {
                    total += conv(pixels, channels, width, 1);
                }
                channels = width;
            }
        }

        total += conv(pixels, channels, self.target_channels(), self.head_kernel); // head
        total / 1e9
    }

    /// Elements in one conditioning tensor of a batch.
    pub fn cond_len(&self) -> usize {
        self.cond_len_for_extent([self.tile, self.tile])
    }

    /// Elements in one conditioning tensor at a runtime extent.
    pub fn cond_len_for_extent(&self, extent: [u32; 2]) -> usize {
        (self.batch * self.cond_channels() * extent[0] * extent[1]) as usize
    }

    /// Elements in one target or output tensor of a batch.
    pub fn target_len(&self) -> usize {
        self.target_len_for_extent([self.tile, self.tile])
    }

    /// Elements in one target or output tensor at a runtime extent.
    pub fn target_len_for_extent(&self, extent: [u32; 2]) -> usize {
        (self.batch * self.target_channels() * extent[0] * extent[1]) as usize
    }

    /// Elements in one batch of timestep embeddings.
    pub fn time_len(&self) -> usize {
        (self.batch * self.time_input_dim) as usize
    }

    /// Reject configurations the graph cannot express, with a reason.
    ///
    /// Called by [`build`]; exposed so a caller can check a configuration it
    /// assembled without paying for a compile.
    pub fn validate(&self) -> Result<(), String> {
        if self.scale < 2 {
            return Err(format!("scale {} must be at least 2", self.scale));
        }
        if self.levels() == 0 {
            return Err("the network needs at least one level".into());
        }
        for level in 0..self.levels() {
            let channels = self.channels_at(level);
            if !channels.is_multiple_of(self.num_groups) {
                return Err(format!(
                    "level {level} has {channels} channels, not divisible by \
                     num_groups {}",
                    self.num_groups
                ));
            }
        }
        if self.backbone == Backbone::Local && self.objective != Objective::Direct {
            return Err(
                "the local backbone requires direct regression and a new checkpoint".into(),
            );
        }
        self.validate_extent([self.tile, self.tile])?;
        if !self.time_input_dim.is_multiple_of(2) {
            return Err(format!(
                "time_input_dim {} must be even for a sinusoidal embedding",
                self.time_input_dim
            ));
        }
        if self.cond_channels() == 0 {
            return Err("the conditioning plane set is empty".into());
        }
        if let Some(temporal) = self.temporal {
            if temporal.frames < 2 {
                return Err(format!(
                    "temporal history needs at least two frames, got {}",
                    temporal.frames
                ));
            }
            if self.cond_planes.contains(Plane::Motion) {
                return Err("motion is consumed by reprojection, not by the model".into());
            }
            if temporal.features.has_phase() && !self.cond_planes.contains(Plane::Jitter) {
                return Err("phase history needs the projection-jitter conditioning plane".into());
            }
            if temporal.features.has_phase_lobes() {
                for plane in [
                    Plane::DiffuseIllumination,
                    Plane::SpecularRadiance,
                    Plane::EmissiveRadiance,
                ] {
                    if !self.cond_planes.contains(plane) {
                        return Err(format!("phase-lobe history requires {plane:?}"));
                    }
                }
            }
        }
        if self.prediction == Prediction::LowResolutionResidual
            && !matches!(
                self.reconstruction_base,
                ReconstructionBase::HighResolutionGuided | ReconstructionBase::SplitRadianceGuided
            )
        {
            return Err(
                "low-resolution prediction needs HR-guided or split-radiance reconstruction".into(),
            );
        }
        if (self.prediction == Prediction::SubpixelKernel)
            != (self.reconstruction_base == ReconstructionBase::Sample)
        {
            return Err(
                "kernel prediction is the sample-gathering reconstruction; neither works \
                 without the other"
                    .into(),
            );
        }
        if self.demodulate {
            let supported = self.prediction == Prediction::SubpixelKernel
                || (self.prediction == Prediction::SubpixelResidual
                    && self.reconstruction_base == ReconstructionBase::HighResolutionGuided);
            if !supported {
                return Err(
                    "demodulation needs a sample gather or an HR-guided subpixel residual".into(),
                );
            }
            if !self.cond_planes.contains(Plane::DiffuseAlbedo) {
                return Err("demodulation divides by the albedo, so it has to have one".into());
            }
            if !self.demodulation_offset.is_finite() || self.demodulation_offset <= 0.0 {
                return Err(format!(
                    "demodulation offset {} must be finite and positive",
                    self.demodulation_offset
                ));
            }
        }
        if self.fusion.is_linear()
            && (!self.linear_kernel
                || self.prediction != Prediction::SubpixelKernel
                || !self.guide_mix
                || !self.temporal.is_some_and(|t| t.previous_output))
        {
            return Err("linear/candidate fusion requires linear kernel taps, guide mixing and previous-output history".into());
        }
        if self.guide_mix {
            if self.prediction != Prediction::SubpixelKernel {
                return Err("guide mixing is part of the sample gather".into());
            }
            if !self.demodulate {
                return Err("guide mixing operates in demodulated illumination space".into());
            }
            if self.temporal.is_none() {
                return Err("guide mixing needs accumulated temporal samples".into());
            }
            for plane in [Plane::Depth, Plane::Normal, Plane::DiffuseAlbedo] {
                if !self.cond_planes.contains(plane) {
                    return Err(format!(
                        "guide mixing requires the {plane:?} conditioning plane"
                    ));
                }
            }
        }
        if self.prediction == Prediction::SubpixelKernel {
            if self.kernel_radius == 0 {
                return Err("a gather kernel needs a radius of at least one".into());
            }
            if !self.cond_planes.contains(Plane::Color) {
                return Err("kernel prediction gathers colour, so it has to see it".into());
            }
            if self.objective != Objective::Direct {
                return Err("kernel prediction has no noised residual to denoise".into());
            }
        }
        if matches!(
            self.reconstruction_base,
            ReconstructionBase::GuidedBilinear | ReconstructionBase::HighResolutionGuided
        ) {
            for plane in [
                Plane::Color,
                Plane::Depth,
                Plane::Normal,
                Plane::DiffuseAlbedo,
            ] {
                if !self.cond_planes.contains(plane) {
                    return Err(format!(
                        "guided reconstruction requires the {plane:?} conditioning plane"
                    ));
                }
            }
        }
        if self.reconstruction_base == ReconstructionBase::SplitRadianceGuided {
            for plane in [
                Plane::DiffuseIllumination,
                Plane::SpecularRadiance,
                Plane::EmissiveRadiance,
                Plane::Depth,
                Plane::Normal,
                Plane::DiffuseAlbedo,
                Plane::Roughness,
            ] {
                if !self.cond_planes.contains(plane) {
                    return Err(format!(
                        "split-radiance reconstruction requires the {plane:?} conditioning plane"
                    ));
                }
            }
            if self.demodulate {
                return Err("split radiance restores diffuse albedo before the residual".into());
            }
        }
        if self.temporal_weight != 0.0 {
            if !self.temporal_weight.is_finite() || self.temporal_weight < 0.0 {
                return Err(format!(
                    "temporal weight {} must be finite and non-negative",
                    self.temporal_weight
                ));
            }
            if self.prediction != Prediction::SubpixelKernel {
                return Err("the temporal loss is defined on the gathered image".into());
            }
            if self.temporal.is_none() {
                return Err("a temporal loss needs a sequence dataset".into());
            }
        }
        if self.head_kernel == 0 || self.head_kernel.is_multiple_of(2) {
            return Err(format!(
                "head kernel {} must be odd and non-zero, for \"same\" padding",
                self.head_kernel
            ));
        }
        if !self.residual_gain.is_finite() || self.residual_gain <= 0.0 {
            return Err(format!(
                "residual_gain {} must be finite and positive",
                self.residual_gain
            ));
        }
        if !self.relative_loss_weight.is_finite() || self.relative_loss_weight < 0.0 {
            return Err(format!(
                "relative loss weight {} must be finite and non-negative",
                self.relative_loss_weight
            ));
        }
        if !self.low_frequency_loss_weight.is_finite() || self.low_frequency_loss_weight < 0.0 {
            return Err(format!(
                "low-frequency loss weight {} must be finite and non-negative",
                self.low_frequency_loss_weight
            ));
        }
        if !self.residual_bound.is_finite() || !(0.0..1.0).contains(&self.residual_bound) {
            return Err(format!(
                "residual bound {} must be finite and in [0, 1)",
                self.residual_bound
            ));
        }
        if self.residual_bound != 0.0 && self.prediction != Prediction::SubpixelResidual {
            return Err("a residual bound currently targets subpixel residuals".into());
        }
        if self.relative_loss_weight != 0.0 && self.prediction != Prediction::SubpixelResidual {
            return Err("relative loss weighting currently targets subpixel residuals".into());
        }
        if self.low_frequency_loss_weight != 0.0 && self.prediction != Prediction::SubpixelResidual
        {
            return Err("low-frequency loss currently targets subpixel residuals".into());
        }
        if self.linear_kernel && self.prediction != Prediction::SubpixelKernel {
            return Err("linear kernel space needs kernel prediction".into());
        }
        for (name, value) in [
            ("spatial_sigma", self.guide.spatial_sigma),
            ("depth_sigma", self.guide.depth_sigma),
            ("normal_power", self.guide.normal_power),
            ("albedo_sigma", self.guide.albedo_sigma),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("guide {name} {value} must be finite and positive"));
            }
        }
        Ok(())
    }

    /// Reject a runtime extent that the checkpoint's U-Net cannot express.
    pub fn validate_extent(&self, extent: [u32; 2]) -> Result<(), String> {
        if self.levels() == 0 {
            return Err("the network needs at least one level".into());
        }
        let shrink = 1u32 << (self.levels() - 1);
        for (axis, value) in [("width", extent[0]), ("height", extent[1])] {
            if !value.is_multiple_of(shrink) {
                return Err(format!(
                    "{axis} {value} is not divisible by {shrink}, which {} levels of downsampling need",
                    self.levels()
                ));
            }
            if value / shrink < 2 {
                return Err(format!(
                    "{axis} {value} collapses below 2 after {} levels",
                    self.levels()
                ));
            }
        }
        Ok(())
    }
}

/// A built network: the graph, its output, and how to initialise it.
pub struct Model {
    pub graph: Graph,
    pub config: ModelConfig,
    /// Spatial extent baked into this graph, in input pixels.
    pub input_extent: [u32; 2],
    /// Predicted noise under [`Objective::Diffusion`], predicted residual
    /// under [`Objective::Direct`]. Shape `[batch, target_channels, tile,
    /// tile]`, flattened.
    pub output: NodeId,
    /// The loss, present only when the graph was built for training.
    pub loss: Option<NodeId>,
    pub params: Vec<ParamInit>,
}

impl Model {
    /// Fill every declared parameter in a session.
    ///
    /// Deterministic given `seed`, so a run can be replayed.
    pub fn initialize(&self, session: &mut meganeura::Session, seed: u64) {
        let mut rng = crate::rng::Rng::new(seed);
        for param in &self.params {
            let data: Vec<f32> = match &param.kind {
                InitKind::Kaiming { fan_in } => {
                    let scale = (2.0 / (*fan_in).max(1) as f32).sqrt();
                    (0..param.len).map(|_| rng.normal() * scale).collect()
                }
                InitKind::Zeros => vec![0.0; param.len],
                InitKind::Ones => vec![1.0; param.len],
                InitKind::Values(values) => {
                    assert_eq!(
                        values.len(),
                        param.len,
                        "{} was given the wrong initial values",
                        param.name
                    );
                    values.clone()
                }
            };
            session.set_parameter(&param.name, &data);
        }
    }
}

/// Tracks the tensor shape as the builder walks the U-Net.
#[derive(Clone, Copy)]
struct Shape {
    channels: u32,
    h: u32,
    w: u32,
}

impl Shape {
    fn spatial(&self) -> u32 {
        self.h * self.w
    }
}

/// Collects parameter declarations as the graph is built.
struct Builder<'a> {
    g: &'a mut Graph,
    config: &'a ModelConfig,
    params: Vec<ParamInit>,
}

impl<'a> Builder<'a> {
    fn param(&mut self, name: &str, len: usize, kind: InitKind) -> NodeId {
        self.params.push(ParamInit {
            name: name.to_string(),
            len,
            kind,
        });
        self.g.parameter(name, &[len])
    }

    /// A convolution's weight, laid out as meganeura wants it: `[out, in, kh,
    /// kw]` flattened.
    fn conv_weight(&mut self, name: &str, out_c: u32, in_c: u32, k: u32) -> NodeId {
        let fan_in = (in_c * k * k) as usize;
        self.param(
            name,
            (out_c as usize) * fan_in,
            InitKind::Kaiming { fan_in },
        )
    }

    fn conv(&mut self, x: NodeId, name: &str, s: Shape, out_c: u32, k: u32, stride: u32) -> NodeId {
        let weight = self.conv_weight(name, out_c, s.channels, k);
        // "same" padding for odd kernels; 1x1 needs none.
        let padding = k / 2;
        self.g.conv2d(
            x,
            weight,
            self.config.batch,
            s.channels,
            s.h,
            s.w,
            out_c,
            k,
            k,
            stride,
            padding,
        )
    }

    fn group_norm(&mut self, x: NodeId, name: &str, s: Shape) -> NodeId {
        if self.config.backbone == Backbone::Local {
            return x;
        }
        let weight = self.param(
            &format!("{name}.weight"),
            s.channels as usize,
            InitKind::Ones,
        );
        let bias = self.param(
            &format!("{name}.bias"),
            s.channels as usize,
            InitKind::Zeros,
        );
        self.g.group_norm(
            x,
            weight,
            bias,
            self.config.batch,
            s.channels,
            s.spatial(),
            self.config.num_groups,
            self.config.gn_eps,
        )
    }

    /// `[rows, in] @ [in, out] + bias`.
    fn linear(&mut self, x: NodeId, name: &str, in_dim: u32, out_dim: u32) -> NodeId {
        let weight = self.param(
            &format!("{name}.weight"),
            (in_dim * out_dim) as usize,
            InitKind::Kaiming {
                fan_in: in_dim as usize,
            },
        );
        let weight = self.g.reshape(weight, &[in_dim as usize, out_dim as usize]);
        let bias = self.param(&format!("{name}.bias"), out_dim as usize, InitKind::Zeros);
        let out = self.g.matmul(x, weight);
        self.g.bias_add(out, bias)
    }

    /// Broadcast a per-channel vector over the spatial plane of an NCHW
    /// tensor.
    ///
    /// `[B, C] -> [B*C, 1] @ [1, HW] -> [B, C, H, W]`. The flat result of that
    /// matmul is already in NCHW order, so no transpose is needed.
    fn broadcast_spatial(&mut self, per_channel: NodeId, channels: u32, spatial: u32) -> NodeId {
        let rows = (self.config.batch * channels) as usize;
        let column = self.g.reshape(per_channel, &[rows, 1]);
        let ones = self
            .g
            .constant(vec![1.0; spatial as usize], &[1, spatial as usize]);
        let plane = self.g.matmul(column, ones);
        self.g.reshape(plane, &[rows * spatial as usize])
    }

    /// GroupNorm, SiLU, conv, add the projected timestep, GroupNorm, SiLU,
    /// conv, plus a residual that is projected when the width changes.
    fn resblock(
        &mut self,
        x: NodeId,
        time: Option<NodeId>,
        name: &str,
        s: Shape,
        out_c: u32,
    ) -> NodeId {
        let h = self.group_norm(x, &format!("{name}.norm1"), s);
        let h = self.g.silu(h);
        let mut h = self.conv(h, &format!("{name}.conv1.weight"), s, out_c, 3, 1);

        if let Some(time) = time {
            let projected = self.linear(
                time,
                &format!("{name}.time_proj"),
                self.config.time_embed_dim,
                out_c,
            );
            let plane = self.broadcast_spatial(projected, out_c, s.spatial());
            h = self.g.add(h, plane);
        }

        let wide = Shape {
            channels: out_c,
            h: s.h,
            w: s.w,
        };
        let h = self.group_norm(h, &format!("{name}.norm2"), wide);
        let h = self.g.silu(h);
        let h = self.conv(h, &format!("{name}.conv2.weight"), wide, out_c, 3, 1);
        let h = if self.config.backbone == Backbone::Local {
            let len = (self.config.batch * out_c * s.spatial()) as usize;
            let scale = self.g.constant(vec![0.1; len], &[len]);
            self.g.mul(h, scale)
        } else {
            h
        };

        if s.channels == out_c {
            self.g.add(x, h)
        } else {
            let skip = self.conv(x, &format!("{name}.skip.weight"), s, out_c, 1, 1);
            self.g.add(skip, h)
        }
    }
}

/// Starting bias for the kernel head, so an untrained network reconstructs
/// exactly texel-centre bilinear rather than something arbitrary.
///
/// The head convolution starts at zero, so at step zero every pixel gets this
/// same kernel and the model is worth precisely the baseline it has to beat.
/// Taps outside bilinear's support start at a floor rather than at nothing: a
/// weight of zero has a gradient of zero under softplus, and a tap that can
/// never be recruited is a tap that might as well not exist.
pub(crate) fn bilinear_kernel_bias(config: &ModelConfig) -> Vec<f32> {
    const FLOOR: f32 = 0.01;
    // Inverse softplus. `ln(exp(w) - 1)` loses precision for small `w`, where
    // `exp(w) - 1` is the difference of two nearby numbers, so use `expm1`.
    let inverse_softplus = |w: f32| w.exp_m1().ln();

    let scale = config.scale as f32;
    let spatial_taps = config.taps();
    let taps = config.gather_taps();
    let slots = config.scale * config.scale;
    let mut out = vec![0.0; config.target_channels() as usize];
    for slot in 0..slots {
        let (sub_x, sub_y) = config.sub_pixel(slot);
        // Where this output sub-pixel lands, in input pixels, relative to the
        // input pixel that owns it.
        let center_x = (sub_x as f32 + 0.5) / scale - 0.5;
        let center_y = (sub_y as f32 + 0.5) / scale - 0.5;
        for tap in 0..taps {
            // Accumulated-sample history starts at the floor: an untrained
            // network reconstructs the current frame and leaves the past
            // alone, so any use it makes of history later is something
            // training found.
            let weight = if tap >= spatial_taps {
                0.0
            } else {
                let (dx, dy) = config.tap_offset(tap);
                (1.0 - (dx as f32 - center_x).abs()).max(0.0)
                    * (1.0 - (dy as f32 - center_y).abs()).max(0.0)
            };
            out[(slot * taps + tap) as usize] = inverse_softplus(weight.max(FLOOR));
        }
    }
    // Mix gates sit after the spatial weights and share the softplus. Guide
    // mixing starts at 25% gather, the conservative fixed blend that cleaned
    // flat-material validation while leaving a useful gradient in both
    // directions. History starts at the floor and is ignored until training
    // finds evidence for it.
    if config.guide_mix_channels() != 0 {
        let gather_share = 0.25;
        let odds: f32 = gather_share / (1.0 - gather_share);
        let mix_bias = if config.fusion == crate::fusion::Mode::CandidateAware {
            odds.ln()
        } else {
            inverse_softplus(odds)
        };
        for slot in 0..slots {
            out[(slots * taps + slot) as usize] = mix_bias;
        }
    }
    if config.history_mix_channels() != 0 {
        let mix_bias = if config.fusion == crate::fusion::Mode::CandidateAware {
            FLOOR.ln()
        } else {
            inverse_softplus(FLOOR)
        };
        let offset = slots * taps + config.guide_mix_channels();
        for slot in 0..slots {
            out[(offset + slot) as usize] = mix_bias;
        }
    }
    out
}

/// Blend an image with a second source using one softplus gate per sub-pixel.
/// The head value `m` becomes the bounded share `m/(m+1)` and is repeated over
/// RGB. An optional validity mask hard-closes history at rejected pixels.
fn blend_subpixel(
    graph: &mut Graph,
    image: NodeId,
    source: NodeId,
    gates: NodeId,
    validity: Option<NodeId>,
    shape: [u32; 3],
) -> NodeId {
    let [batch, slots, spatial] = shape;
    let gate_len = (batch * slots * spatial) as usize;
    let gate_ones = graph.constant(vec![1.0; gate_len], &[gate_len]);
    let denominator = graph.add(gates, gate_ones);
    let mut gates = graph.div(gates, denominator);
    if let Some(validity) = validity {
        gates = graph.mul(gates, validity);
    }
    let twice = graph.concat(gates, gates, batch, slots, slots, spatial);
    let gates_rgb = graph.concat(twice, gates, batch, 2 * slots, slots, spatial);
    let image_len = (batch * 3 * slots * spatial) as usize;
    let ones = graph.constant(vec![1.0; image_len], &[image_len]);
    let negative_gates = graph.neg(gates_rgb);
    let keep = graph.add(ones, negative_gates);
    let from_image = graph.mul(keep, image);
    let from_source = graph.mul(gates_rgb, source);
    graph.add(from_image, from_source)
}

/// Reconstruct the image from predicted gather weights, inside the graph.
///
/// Only training needs this. At runtime the unpack shader gathers straight from
/// the input texture in one dispatch, and the network's output stops at the
/// weights. But a kernel that is never applied has no gradient, so the training
/// graph has to carry the gather and the loss has to sit on the image.
///
/// The reduction over taps is a 1x1 convolution against constant ones. That is
/// the one shape meganeura has no primitive for — summing a channel group —
/// and expressing it as a convolution keeps the whole gather at about sixty
/// operations instead of the thousand a per-channel decomposition would need.
fn gather(
    graph: &mut Graph,
    config: &ModelConfig,
    weights: NodeId,
    extent: [u32; 2],
    history_inputs: Option<(NodeId, NodeId)>,
) -> NodeId {
    let batch = config.batch;
    let [width, height] = extent;
    let spatial = width * height;
    let taps = config.gather_taps();
    let slots = config.scale * config.scale;
    let guide_mix = config.guide_mix_channels();
    let history_mix = config.history_mix_channels();
    let mixes = guide_mix + history_mix;
    let (weights, mix_gates) = if mixes == 0 {
        (weights, None)
    } else {
        let spatial_ch = slots * taps;
        let w = graph.split_a(weights, batch, spatial_ch, mixes, spatial);
        let m = graph.split_b(weights, batch, spatial_ch, mixes, spatial);
        (w, Some(m))
    };
    let (guide_gates, history_gates) = match (guide_mix, history_mix, mix_gates) {
        (0, 0, None) => (None, None),
        (_, 0, Some(gates)) => (Some(gates), None),
        (0, _, Some(gates)) => (None, Some(gates)),
        (_, _, Some(gates)) => (
            Some(graph.split_a(gates, batch, guide_mix, history_mix, spatial)),
            Some(graph.split_b(gates, batch, guide_mix, history_mix, spatial)),
        ),
        _ => unreachable!(),
    };

    // Peel one group of `group` channels at a time off the front.
    let peel = |graph: &mut Graph, mut rest: NodeId, groups: u32, group: u32| {
        let mut out = Vec::with_capacity(groups as usize);
        for index in 0..groups {
            let remaining = (groups - index) * group;
            if index + 1 == groups {
                out.push(rest);
            } else {
                out.push(graph.split_a(rest, batch, group, remaining - group, spatial));
                rest = graph.split_b(rest, batch, group, remaining - group, spatial);
            }
        }
        out
    };

    let per_slot = peel(graph, weights, slots, taps);
    // The sparse samples themselves, one shifted copy per tap. New kernel
    // checkpoints carry linear demodulated radiance here and apply the bounded
    // transform after averaging; legacy checkpoints retain compressed taps.
    // `batch::write_taps` fills the matching representation.
    let samples = graph.input("taps", &[(batch * 3 * taps * spatial) as usize]);
    let per_channel = peel(graph, samples, 3, taps);

    let ones = graph.constant(vec![1.0; taps as usize], &[taps as usize]);
    let sum_taps = |graph: &mut Graph, x: NodeId| {
        graph.conv2d(x, ones, batch, taps, height, width, 1, 1, 1, 1, 0)
    };

    let totals: Vec<NodeId> = per_slot.iter().map(|&slot| sum_taps(graph, slot)).collect();

    // Channel `c * slots + slot`, matching the residual layout the assembler
    // and the reference target already use.
    let mut image: Option<NodeId> = None;
    let mut written = 0u32;
    for &channel in &per_channel {
        for (slot, &weight) in per_slot.iter().enumerate() {
            let weighted = graph.mul(weight, channel);
            let summed = sum_taps(graph, weighted);
            let normalized = graph.div(summed, totals[slot]);
            image = Some(match image {
                None => normalized,
                Some(prefix) => graph.concat(prefix, normalized, batch, written, 1, spatial),
            });
            written += 1;
        }
    }
    let mut image = image.expect("a kernel checkpoint reconstructs at least one channel");
    if config.linear_kernel {
        // Runtime `compress` clamps the physical estimator to non-negative
        // radiance first. Mirror that here so tiny negative renderer values or
        // round-off cannot cross the rational transform's pole during
        // training.
        image = graph.relu(image);
        let len = (batch * 3 * slots * spatial) as usize;
        let ones = graph.constant(vec![1.0; len], &[len]);
        let denominator = graph.add(image, ones);
        image = graph.div(image, denominator);
    }
    if config.fusion.is_linear() {
        let guide = graph.input("guide", &[(batch * 3 * slots * spatial) as usize]);
        let (history, validity) = history_inputs.expect("validated recurrent fusion");
        return crate::fusion::build(
            graph,
            [image, guide, history],
            validity,
            [guide_gates.unwrap(), history_gates.unwrap()],
            config.fusion,
            [batch, slots, spatial],
        );
    }
    if let Some(gates) = guide_gates {
        let guide = graph.input("guide", &[(batch * 3 * slots * spatial) as usize]);
        image = blend_subpixel(graph, guide, image, gates, None, [batch, slots, spatial]);
    }
    if let Some(gates) = history_gates {
        // Previous reconstruction, already warped, in the same compressed
        // sub-pixel layout as `image`. Reprojection validity hard-closes the
        // gate at disocclusions and outside the frame; a rejected zero is
        // storage, not black radiance.
        let (history, history_validity) = history_inputs.unwrap_or_else(|| {
            (
                graph.input("history", &[(batch * 3 * slots * spatial) as usize]),
                graph.input("history_validity", &[(batch * slots * spatial) as usize]),
            )
        });
        image = blend_subpixel(
            graph,
            image,
            history,
            gates,
            Some(history_validity),
            [batch, slots, spatial],
        );
    }
    image
}

/// What the built graph produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ending {
    /// The network's own output: a residual, or a kernel checkpoint's weights.
    /// What the runtime wants, since the shader does the gather.
    Prediction,
    /// The reconstructed image. Only a kernel checkpoint can end here, and only
    /// the temporal loss's detached teacher asks for it — it needs the picture
    /// the previous frame produced, not the kernel that produced it.
    Image,
    /// The loss, with the graph carrying whatever inputs it needs to compute
    /// one.
    Loss,
}

/// Build the network.
///
/// With `training`, the graph gains a `target` input and an MSE loss, and
/// [`Model::loss`] is set. Without it the graph ends at the prediction, ready
/// for an inference session.
///
/// Graph inputs:
/// - `cond`: `[batch, cond_channels, tile, tile]`, the conditioning planes.
/// - `x_t`: `[batch, target_channels, tile, tile]`, the noised residual.
///   Diffusion only.
/// - `t_emb`: `[batch, time_input_dim]`, the sinusoidal timestep embedding.
///   Diffusion only.
/// - `target`: `[batch, target_channels, tile, tile]`, training only. The
///   noise under diffusion, the residual under direct regression.
pub fn build(config: &ModelConfig, training: bool) -> Result<Model, String> {
    build_for_extent(config, training, [config.tile, config.tile])
}

/// Build the network for a rectangular runtime extent.
///
/// Convolution weights are independent of the spatial dimensions, so a
/// checkpoint trained on square crops can be instantiated for a full Blade
/// frame. The extent still has to survive every U-Net downsampling level.
pub fn build_for_extent(
    config: &ModelConfig,
    training: bool,
    extent: [u32; 2],
) -> Result<Model, String> {
    let ending = if training {
        Ending::Loss
    } else {
        Ending::Prediction
    };
    build_ending(config, ending, extent)
}

/// MSE between non-overlapping block averages, independently per channel.
///
/// A fixed diagonal convolution expresses this with the operations Meganeura
/// already uses for the network. Only gradients with respect to `output` are
/// useful; the averaging kernel is a graph constant.
fn low_frequency_mse(
    graph: &mut Graph,
    output: NodeId,
    target: NodeId,
    batch: u32,
    channels: u32,
    extent: [u32; 2],
) -> NodeId {
    const BLOCK: u32 = 8;
    let [width, height] = extent;
    let block = BLOCK.min(width).min(height);
    let area = (block * block) as usize;
    let mut weights = vec![0.0; channels as usize * channels as usize * area];
    for channel in 0..channels as usize {
        let base = (channel * channels as usize + channel) * area;
        weights[base..base + area].fill(1.0 / area as f32);
    }
    let kernel = graph.constant(weights, &[channels as usize * channels as usize * area]);
    let average = |graph: &mut Graph, image| {
        graph.conv2d(
            image, kernel, batch, channels, height, width, channels, block, block, block, 0,
        )
    };
    let output = average(graph, output);
    let target = average(graph, target);
    graph.mse_loss(output, target)
}

/// Build the network for a chosen ending.
pub fn build_ending(
    config: &ModelConfig,
    ending: Ending,
    extent: [u32; 2],
) -> Result<Model, String> {
    if ending == Ending::Image && config.prediction != Prediction::SubpixelKernel {
        return Err("only a kernel checkpoint's graph reaches an image".into());
    }
    config.validate()?;
    config.validate_extent(extent)?;

    let mut graph = Graph::new();
    let mut builder = Builder {
        g: &mut graph,
        config,
        params: Vec::new(),
    };

    let batch = config.batch;
    let [width, height] = extent;
    let spatial = width * height;
    let cond = builder
        .g
        .input("cond", &[config.cond_len_for_extent(extent)]);

    // The final history blend needs these tensors in a training/image graph.
    // Runtime prediction ends at the weights and performs the blend in WGSL.
    let slots = config.scale * config.scale;
    let history_inputs =
        (config.history_mix_channels() != 0 && ending != Ending::Prediction).then(|| {
            (
                builder
                    .g
                    .input("history", &[(batch * 3 * slots * spatial) as usize]),
                builder
                    .g
                    .input("history_validity", &[(batch * slots * spatial) as usize]),
            )
        });

    // Under diffusion the network sees the noised residual next to the
    // conditioning, and the noise level tells it how much of what it sees is
    // signal. Direct regression has neither.
    let (input, time) = match config.objective {
        Objective::Diffusion => {
            let x_t = builder
                .g
                .input("x_t", &[config.target_len_for_extent(extent)]);
            let joined = builder.g.concat(
                x_t,
                cond,
                batch,
                config.target_channels(),
                config.cond_channels(),
                spatial,
            );

            let t_emb = builder
                .g
                .input("t_emb", &[batch as usize, config.time_input_dim as usize]);
            let embedded = builder.linear(
                t_emb,
                "time.in",
                config.time_input_dim,
                config.time_embed_dim,
            );
            let embedded = builder.g.silu(embedded);
            let embedded = builder.linear(
                embedded,
                "time.out",
                config.time_embed_dim,
                config.time_embed_dim,
            );
            (joined, Some(embedded))
        }
        Objective::Direct => (cond, None),
    };

    // Stem.
    let stem_shape = Shape {
        channels: config.in_channels(),
        h: height,
        w: width,
    };
    let mut h = builder.conv(input, "stem.weight", stem_shape, config.base_channels, 3, 1);
    let mut shape = Shape {
        channels: config.base_channels,
        h: height,
        w: width,
    };

    // Encoder. One skip per level, taken before the downsample.
    let levels = config.levels();
    let mut skips: Vec<(NodeId, Shape)> = Vec::new();
    for level in 0..levels {
        let width = config.channels_at(level);
        for block in 0..config.blocks_per_level {
            h = builder.resblock(h, time, &format!("down.{level}.{block}"), shape, width);
            shape.channels = width;
        }
        if level + 1 < levels {
            skips.push((h, shape));
            // Strided convolution rather than pooling: the downsample gets to
            // learn what to keep, and it is the same cost as a 3x3 stride-1.
            h = builder.conv(h, &format!("down.{level}.pool.weight"), shape, width, 3, 2);
            shape = Shape {
                channels: width,
                h: shape.h / 2,
                w: shape.w / 2,
            };
        }
    }

    // Middle at the narrowest resolution.
    let middle = shape.channels;
    h = builder.resblock(h, time, "middle.0", shape, middle);
    h = builder.resblock(h, time, "middle.1", shape, middle);

    // Decoder. Upsample, concatenate the matching skip, then narrow back down.
    for level in (0..levels - 1).rev() {
        h = builder
            .g
            .upsample_2x(h, batch, shape.channels, shape.h, shape.w);
        let upsampled = Shape {
            channels: shape.channels,
            h: shape.h * 2,
            w: shape.w * 2,
        };

        let (skip, skip_shape) = skips.pop().expect("a skip per encoder level");
        debug_assert_eq!(skip_shape.h, upsampled.h);
        h = builder.g.concat(
            h,
            skip,
            batch,
            upsampled.channels,
            skip_shape.channels,
            upsampled.spatial(),
        );
        shape = Shape {
            channels: upsampled.channels + skip_shape.channels,
            h: upsampled.h,
            w: upsampled.w,
        };

        let width = config.channels_at(level);
        for block in 0..config.blocks_per_level {
            h = builder.resblock(h, time, &format!("up.{level}.{block}"), shape, width);
            shape.channels = width;
        }
    }

    // Head. The output convolution starts at zero so the network's first
    // prediction is exactly zero — under diffusion that is a far better
    // starting point than noise, and under direct regression it means the
    // untrained network passes the input through unchanged.
    let h = builder.group_norm(h, "head.norm", shape);
    let h = builder.g.silu(h);
    let head_kernel = config.head_kernel;
    let head_weight = builder.param(
        "head.conv.weight",
        (config.target_channels() * shape.channels * head_kernel * head_kernel) as usize,
        InitKind::Zeros,
    );
    let output = builder.g.conv2d(
        h,
        head_weight,
        batch,
        shape.channels,
        shape.h,
        shape.w,
        config.target_channels(),
        head_kernel,
        head_kernel,
        1,
        head_kernel / 2,
    );

    // Kernel prediction turns the head's logits into strictly positive weights.
    // Softplus rather than an exponential: it cannot overflow, it is zero
    // nowhere so every tap keeps a gradient, and its inverse is closed-form, so
    // the bias below can start the network at an exact filter.
    let output = if config.prediction == Prediction::SubpixelKernel {
        let bias = builder.param(
            "head.kernel.bias",
            config.target_channels() as usize,
            InitKind::Values(bilinear_kernel_bias(config)),
        );
        let biased = builder
            .g
            .add_per_channel(output, bias, config.target_channels(), spatial);
        // Mix gates share this softplus so the prediction graph stays the
        // same shape as a spatial kernel; gather maps `m` to `m/(m+1)`.
        if config.fusion == crate::fusion::Mode::CandidateAware {
            let spatial_channels = config.scale * config.scale * config.gather_taps();
            let mixes = config.guide_mix_channels() + config.history_mix_channels();
            let taps = builder
                .g
                .split_a(biased, batch, spatial_channels, mixes, spatial);
            let coefficients = builder
                .g
                .split_b(biased, batch, spatial_channels, mixes, spatial);
            let positive = builder.g.softplus(taps, 1.0);
            builder.g.concat(
                positive,
                coefficients,
                batch,
                spatial_channels,
                mixes,
                spatial,
            )
        } else {
            builder.g.softplus(biased, 1.0)
        }
    } else if config.residual_bound != 0.0 {
        let bounded = builder.g.tanh(output);
        let len = config.target_len_for_extent(extent);
        let scale = builder.g.constant(
            vec![config.residual_bound * config.residual_gain; len],
            &[len],
        );
        builder.g.mul(bounded, scale)
    } else {
        output
    };

    let params = builder.params;
    if ending == Ending::Image {
        let image = gather(&mut graph, config, output, extent, history_inputs);
        graph.set_outputs(vec![image]);
        return Ok(Model {
            graph,
            config: config.clone(),
            input_extent: extent,
            output: image,
            loss: None,
            params,
        });
    }
    let training = ending == Ending::Loss;
    let loss = if training {
        let loss = match config.prediction {
            Prediction::SubpixelKernel => {
                let image = gather(&mut graph, config, output, extent, history_inputs);
                let len = (batch * config.image_channels() * spatial) as usize;
                let target = graph.input("target", &[len]);
                let mut loss = graph.mse_loss(image, target);
                if config.temporal_weight != 0.0 {
                    // The temporal metric compares this frame's change against
                    // the reference's, motion-compensated:
                    //
                    //   (out - reproj(out_prev)) - (ref - reproj(ref_prev))
                    //
                    // which rearranges to `out - target`, with
                    //
                    //   target = reproj(out_prev) + ref - reproj(ref_prev)
                    //
                    // so the whole term is an ordinary squared error against a
                    // target the host assembles. `out_prev` comes from a
                    // detached copy of the network, which is why no gradient
                    // has to flow through a reprojection the graph could not
                    // express anyway.
                    //
                    // Both sides arrive masked, so a pixel the teacher cannot
                    // reproject contributes zero rather than a wrong number.
                    let mask = graph.input("temporal_mask", &[len]);
                    let target = graph.input("temporal_target", &[len]);
                    let masked = graph.mul(image, mask);
                    let temporal_loss = graph.mse_loss(masked, target);
                    let weight = graph.scalar(config.temporal_weight);
                    let scaled = graph.mul(temporal_loss, weight);
                    loss = graph.add(loss, scaled);
                }
                loss
            }
            _ => {
                let target = graph.input("target", &[config.target_len_for_extent(extent)]);
                let mut loss = if config.relative_loss_weight == 0.0 {
                    graph.mse_loss(output, target)
                } else {
                    let scale = graph.input("loss_scale", &[config.target_len_for_extent(extent)]);
                    let output = graph.mul(output, scale);
                    let target = graph.mul(target, scale);
                    graph.mse_loss(output, target)
                };
                if config.low_frequency_loss_weight != 0.0 {
                    let low = low_frequency_mse(
                        &mut graph,
                        output,
                        target,
                        batch,
                        config.target_channels(),
                        extent,
                    );
                    let weight = graph.scalar(config.low_frequency_loss_weight);
                    let low = graph.mul(low, weight);
                    loss = graph.add(loss, low);
                }
                loss
            }
        };
        graph.set_outputs(vec![loss]);
        Some(loss)
    } else {
        graph.set_outputs(vec![output]);
        None
    };

    Ok(Model {
        graph,
        config: config.clone(),
        input_extent: extent,
        output,
        loss,
        params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> ModelConfig {
        ModelConfig {
            batch: 1,
            tile: 16,
            base_channels: 16,
            level_multipliers: vec![1, 2],
            blocks_per_level: 1,
            num_groups: 8,
            time_input_dim: 16,
            time_embed_dim: 32,
            cond_planes: PlaneSet::new().with(Plane::Color),
            // Most graph tests predate the deployment default and exercise
            // the superset path; direct-specific tests override this below.
            objective: Objective::Diffusion,
            reconstruction_base: ReconstructionBase::Bilinear,
            ..ModelConfig::default()
        }
    }

    #[test]
    fn local_backbone_has_no_normalization_parameters() {
        let mut c = small();
        c.objective = Objective::Direct;
        c.backbone = Backbone::Local;
        let model = build(&c, true).unwrap();
        assert!(model.params.iter().all(|p| !p.name.contains("norm")));
        let serialized = ron::ser::to_string(&c).unwrap();
        let restored: ModelConfig = ron::from_str(&serialized).unwrap();
        assert_eq!(restored.backbone, Backbone::Local);
        let old: ModelConfig = ron::from_str(&serialized.replace(",backbone:Local", "")).unwrap();
        assert_eq!(old.backbone, Backbone::GroupNorm);
        c.objective = Objective::Diffusion;
        assert!(c.validate().is_err());
    }

    #[test]
    fn default_is_the_measured_deployment_baseline() {
        let c = ModelConfig::default();
        assert_eq!(c.objective, Objective::Direct);
        assert_eq!(c.base_channels, 8);
        assert_eq!(c.blocks_per_level, 1);
        assert_eq!(c.batch, 8);
    }

    #[test]
    fn channel_counts_follow_the_scale() {
        let mut c = small();
        assert_eq!(c.target_channels(), 12); // scale 2 -> 3 * 4
        assert_eq!(c.cond_channels(), 3);
        assert_eq!(c.in_channels(), 15);
        c.objective = Objective::Direct;
        assert_eq!(c.in_channels(), 3, "direct sees only the conditioning");
        c.scale = 4;
        assert_eq!(c.target_channels(), 48);
    }

    #[test]
    fn temporal_low_color_uses_only_ordinary_channels() {
        let mut c = small();
        c.objective = Objective::Direct;
        c.prediction = Prediction::LowResolutionResidual;
        c.reconstruction_base = ReconstructionBase::HighResolutionGuided;
        c.cond_planes = c
            .cond_planes
            .with(Plane::Depth)
            .with(Plane::Normal)
            .with(Plane::DiffuseAlbedo);
        c.temporal = Some(crate::temporal::Config {
            frames: 4,
            rejection: crate::temporal::RejectionConfig::default(),
            features: crate::temporal::Features::Basic,
            unrejected_tap: false,
            previous_output: false,
        });
        assert_eq!(c.cond_channels(), 17); // Ten stored plus seven temporal auxiliaries.
        assert_eq!(c.target_channels(), 3);
        assert_eq!(c.in_channels(), 17);
        assert!(c.validate().is_ok());
        c.temporal.as_mut().unwrap().features = crate::temporal::Features::Variance;
        assert_eq!(c.cond_channels(), 18);
    }

    #[test]
    fn previous_output_mix_is_not_a_gather_tap() {
        let mut c = small();
        c.objective = Objective::Direct;
        c.prediction = Prediction::SubpixelKernel;
        c.reconstruction_base = ReconstructionBase::Sample;
        c.kernel_radius = 2;
        c.temporal = Some(crate::temporal::Config {
            frames: 4,
            rejection: crate::temporal::RejectionConfig::default(),
            features: crate::temporal::Features::Variance,
            unrejected_tap: false,
            previous_output: true,
        });
        assert_eq!(c.taps(), 25);
        assert_eq!(c.history_taps(), 0);
        assert_eq!(c.gather_taps(), 25);
        assert_eq!(c.history_mix_channels(), 4);
        assert_eq!(c.target_channels(), 25 * 4 + 4);
    }

    #[test]
    fn guide_mix_costs_only_one_gate_per_subpixel() {
        let mut c = small();
        c.objective = Objective::Direct;
        c.prediction = Prediction::SubpixelKernel;
        c.reconstruction_base = ReconstructionBase::Sample;
        c.kernel_radius = 2;
        c.cond_planes = c
            .cond_planes
            .with(Plane::Depth)
            .with(Plane::Normal)
            .with(Plane::DiffuseAlbedo);
        c.demodulate = true;
        c.guide_mix = true;
        c.temporal = Some(crate::temporal::Config {
            frames: 4,
            rejection: crate::temporal::RejectionConfig::default(),
            features: crate::temporal::Features::Variance,
            unrejected_tap: false,
            previous_output: true,
        });
        assert_eq!(c.gather_taps(), 25);
        assert_eq!(c.guide_mix_channels(), 4);
        assert_eq!(c.target_channels(), 25 * 4 + 4 + 4);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn candidate_fusion_versions_channels_and_builds_all_endings() {
        let mut c = ModelConfig {
            batch: 1,
            tile: 16,
            base_channels: 8,
            prediction: Prediction::SubpixelKernel,
            reconstruction_base: ReconstructionBase::Sample,
            demodulate: true,
            linear_kernel: true,
            guide_mix: true,
            temporal: Some(crate::temporal::Config {
                frames: 4,
                rejection: crate::temporal::RejectionConfig::default(),
                features: crate::temporal::Features::Variance,
                unrejected_tap: false,
                previous_output: true,
            }),
            ..ModelConfig::default()
        };
        let legacy = c.target_channels();
        for mode in [
            crate::fusion::Mode::Linear,
            crate::fusion::Mode::CandidateAware,
        ] {
            c.fusion = mode;
            c.validate().unwrap();
            assert_eq!(
                c.target_channels(),
                legacy + 8 * (mode.gate_parameters() - 1)
            );
            for ending in [Ending::Prediction, Ending::Image, Ending::Loss] {
                build_ending(&c, ending, [c.tile, c.tile]).unwrap();
            }
        }
        c.linear_kernel = false;
        assert!(c.validate().is_err());
    }

    #[test]
    fn split_reconstruction_requires_each_renderer_lobe() {
        let mut c = small();
        c.objective = Objective::Direct;
        c.reconstruction_base = ReconstructionBase::SplitRadianceGuided;
        c.cond_planes = c
            .cond_planes
            .with(Plane::Depth)
            .with(Plane::Normal)
            .with(Plane::DiffuseAlbedo)
            .with(Plane::Roughness)
            .with(Plane::DiffuseIllumination)
            .with(Plane::SpecularRadiance)
            .with(Plane::EmissiveRadiance);
        assert!(c.validate().is_ok());
        c.prediction = Prediction::LowResolutionResidual;
        assert!(
            c.validate().is_ok(),
            "a low-resolution correction is valid on a split-lobe base"
        );
        c.cond_planes = c.cond_planes.without(Plane::SpecularRadiance);
        assert!(c.validate().unwrap_err().contains("SpecularRadiance"));
    }

    #[test]
    fn validation_catches_bad_geometry() {
        let mut c = small();
        assert!(c.validate().is_ok());

        c.level_multipliers = vec![1, 2, 4, 8, 16];
        assert!(
            c.validate().unwrap_err().contains("collapses"),
            "a 16px tile cannot survive 5 levels"
        );

        c = small();
        c.tile = 12; // 4 levels shrink by 8, and 12 is not a multiple of 8
        c.level_multipliers = vec![1, 2, 4, 8];
        assert!(c.validate().unwrap_err().contains("divisible by 8"));

        c = small();
        c.num_groups = 7; // does not divide 16
        assert!(c.validate().unwrap_err().contains("num_groups"));

        c = small();
        c.scale = 1;
        assert!(c.validate().unwrap_err().contains("scale"));

        c = small();
        c.time_input_dim = 15;
        assert!(c.validate().unwrap_err().contains("even"));

        c = small();
        c.cond_planes = PlaneSet::new();
        assert!(c.validate().unwrap_err().contains("empty"));

        c = small();
        c.guide.depth_sigma = 0.0;
        assert!(c.validate().unwrap_err().contains("depth_sigma"));
    }

    #[test]
    fn inference_graph_ends_at_the_prediction() {
        let model = build(&small(), false).unwrap();
        assert!(model.loss.is_none());
        assert_eq!(model.graph.outputs(), &[model.output]);
        let ty = &model.graph.node(model.output).ty;
        assert_eq!(ty.num_elements(), model.config.target_len());
    }

    #[test]
    fn checkpoint_weights_build_for_a_rectangular_frame() {
        let config = small();
        let extent = [32, 24];
        let model = build_for_extent(&config, false, extent).unwrap();
        assert_eq!(model.input_extent, extent);
        let ty = &model.graph.node(model.output).ty;
        assert_eq!(ty.num_elements(), config.target_len_for_extent(extent),);
    }

    #[test]
    fn training_graph_ends_at_the_loss() {
        let model = build(&small(), true).unwrap();
        let loss = model.loss.expect("training graph has a loss");
        assert_eq!(model.graph.outputs(), &[loss]);
        // A loss is a scalar.
        assert_eq!(model.graph.node(loss).ty.num_elements(), 1);
    }

    #[test]
    fn low_frequency_loss_is_training_only_and_parameter_free() {
        let mut baseline = small();
        baseline.objective = Objective::Direct;
        let baseline_params = build(&baseline, true).unwrap().params.len();

        baseline.low_frequency_loss_weight = 4.0;
        let trained = build(&baseline, true).unwrap();
        assert_eq!(trained.params.len(), baseline_params);
        assert_eq!(
            trained.graph.node(trained.loss.unwrap()).ty.num_elements(),
            1
        );
        assert!(build(&baseline, false).unwrap().loss.is_none());

        baseline.prediction = Prediction::SubpixelKernel;
        baseline.reconstruction_base = ReconstructionBase::Sample;
        assert!(baseline.validate().unwrap_err().contains("low-frequency"));
    }

    #[test]
    fn every_parameter_is_declared_once() {
        let model = build(&small(), true).unwrap();
        let mut names: Vec<&str> = model.params.iter().map(|p| p.name.as_str()).collect();
        let total = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), total, "duplicate parameter names");
        assert!(total > 20, "only {total} parameters, the U-Net is too thin");

        // The head starts at zero so the first prediction is zero.
        let head = model
            .params
            .iter()
            .find(|p| p.name == "head.conv.weight")
            .expect("head weight");
        assert_eq!(head.kind, InitKind::Zeros);
    }

    #[test]
    fn direct_objective_drops_the_noise_inputs() {
        let mut c = small();
        c.objective = Objective::Direct;
        let model = build(&c, true).unwrap();
        // No timestep projection parameters at all when there is no timestep.
        assert!(
            !model.params.iter().any(|p| p.name.starts_with("time.")),
            "direct regression should not carry a timestep MLP"
        );
        assert!(!model.params.iter().any(|p| p.name.contains("time_proj")));
    }

    #[test]
    fn diffusion_objective_carries_the_timestep_mlp() {
        let model = build(&small(), true).unwrap();
        assert!(model.params.iter().any(|p| p.name == "time.in.weight"));
        assert!(model.params.iter().any(|p| p.name.contains("time_proj")));
    }
}

#[cfg(test)]
mod flop_tests {
    use super::*;

    fn default_backbone() -> ModelConfig {
        ModelConfig {
            scale: 2,
            tile: 512,
            batch: 1,
            base_channels: 64,
            level_multipliers: vec![1, 2, 4],
            blocks_per_level: 2,
            cond_planes: PlaneSet::new()
                .with(Plane::Color)
                .with(Plane::Depth)
                .with(Plane::Normal)
                .with(Plane::DiffuseAlbedo)
                .with(Plane::SpecularF0)
                .with(Plane::Roughness),
            objective: Objective::Direct,
            ..ModelConfig::default()
        }
    }

    /// Pinned against an independent hand count of the same architecture, so a
    /// change to the builder that the estimator does not follow shows up.
    #[test]
    fn the_estimate_matches_a_hand_count() {
        // 512x512 input, so 1024x1024 out.
        let gflop = default_backbone().flops(1024 * 1024);
        assert!(
            (500.0..580.0).contains(&gflop),
            "expected ~539 GFLOP for the default backbone, got {gflop:.0}"
        );
    }

    #[test]
    fn cost_scales_with_output_pixels() {
        let config = default_backbone();
        let small = config.flops(1024 * 1024);
        let big = config.flops(2048 * 2048);
        assert!(
            (big / small - 4.0).abs() < 1e-6,
            "four times the pixels should cost four times as much: {small} vs {big}",
        );
    }

    /// Halving the width should quarter the arithmetic, near enough — the stem
    /// and head scale linearly rather than quadratically, so it is not exact.
    #[test]
    fn width_dominates_the_cost() {
        let mut narrow = default_backbone();
        narrow.base_channels = 32;
        let ratio = default_backbone().flops(1 << 20) / narrow.flops(1 << 20);
        assert!(
            (3.4..4.0).contains(&ratio),
            "halving the width changed the cost by {ratio:.2}x"
        );
    }
}
