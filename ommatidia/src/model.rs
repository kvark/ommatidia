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

/// Spatial quantity emitted by the network.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Prediction {
    /// Historical path: one residual for every output sub-pixel and RGB channel.
    SubpixelResidual,
    /// RGB correction at input resolution, followed by geometry-aware gather.
    LowResolutionResidual,
    /// A gather kernel over nearby input samples, one per output sub-pixel.
    ///
    /// Denoising and upscaling stop being two stages. There is no filtered
    /// low-resolution image in the middle and no deterministic base to correct:
    /// the output pixel is a weighted average of the sparse samples themselves,
    /// and the network's whole job is deciding which of them belong to it.
    ///
    /// Predicting the weights rather than the colour is what makes this
    /// tractable. Asked for a residual over a filter, a least-squares network
    /// is being asked to predict that filter's error, which is dominated by the
    /// particular noise the renderer drew and whose conditional mean is very
    /// nearly zero — measured at 0.02 dB on this data. Asked for weights, it is
    /// choosing among samples it can see, and it cannot answer zero.
    ///
    /// Two properties come from the form rather than from training. The output
    /// is a convex combination of real radiance, so it cannot overshoot, invent
    /// energy, or emit the black pixels a normalised bilateral gather produces
    /// when it rejects every tap. And the reconstruction is one dispatch over
    /// one neighbourhood, so nothing is filtered twice.
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
    /// Reconstruct radiance divided by albedo, and multiply the exact
    /// output-resolution albedo back afterwards.
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
            time_input_dim: 64,
            time_embed_dim: 256,
            // Overwritten from the data; 1.0 leaves the residual as it is.
            residual_gain: 1.0,
            objective: Objective::Direct,
            prediction: Prediction::SubpixelResidual,
            reconstruction_base: ReconstructionBase::GuidedBilinear,
            guide: GuideConfig::TUNED,
            kernel_radius: legacy_kernel_radius(),
            demodulate: false,
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
        match self.temporal {
            None => 0,
            Some(temporal) if self.prediction == Prediction::SubpixelKernel => {
                temporal.gather_auxiliary_channels()
            }
            Some(temporal) => temporal.auxiliary_channels(),
        }
    }

    /// Reprojected-history taps the gather reads, beyond the spatial ones.
    ///
    /// One: the accumulated estimate at this pixel. Making it a tap rather than
    /// a base is the whole idea — how much to trust history becomes a weight the
    /// network predicts per output sub-pixel, alongside the weights it gives the
    /// current frame's samples, and a history it does not trust simply gets a
    /// small one.
    pub fn history_taps(&self) -> u32 {
        u32::from(self.temporal.is_some() && self.prediction == Prediction::SubpixelKernel)
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
            // One weight per sub-pixel per tap, sub-pixel major.
            Prediction::SubpixelKernel => self.scale * self.scale * self.gather_taps(),
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
        }
        if self.prediction == Prediction::LowResolutionResidual
            && self.reconstruction_base != ReconstructionBase::HighResolutionGuided
        {
            return Err("low-resolution prediction needs HR-guided reconstruction".into());
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
            if self.prediction != Prediction::SubpixelKernel {
                return Err("demodulation is part of the sample gather".into());
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
    let mut out = vec![0.0; config.target_channels() as usize];
    for slot in 0..config.scale * config.scale {
        let (sub_x, sub_y) = config.sub_pixel(slot);
        // Where this output sub-pixel lands, in input pixels, relative to the
        // input pixel that owns it.
        let center_x = (sub_x as f32 + 0.5) / scale - 0.5;
        let center_y = (sub_y as f32 + 0.5) / scale - 0.5;
        for tap in 0..taps {
            // History starts at the floor: an untrained network reconstructs
            // the current frame and leaves the past alone, so any use it makes
            // of history later is something training found.
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
    out
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
fn gather(graph: &mut Graph, config: &ModelConfig, weights: NodeId, extent: [u32; 2]) -> NodeId {
    let batch = config.batch;
    let [width, height] = extent;
    let spatial = width * height;
    let taps = config.gather_taps();
    let slots = config.scale * config.scale;

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
    // The sparse samples themselves, one shifted copy per tap, in compressed
    // space. `batch::write_taps` fills this.
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
    image.expect("a kernel checkpoint reconstructs at least one channel")
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
        builder.g.softplus(biased, 1.0)
    } else {
        output
    };

    let params = builder.params;
    if ending == Ending::Image {
        let image = gather(&mut graph, config, output, extent);
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
                let image = gather(&mut graph, config, output, extent);
                let len = (batch * config.image_channels() * spatial) as usize;
                let target = graph.input("target", &[len]);
                let spatial_loss = graph.mse_loss(image, target);
                if config.temporal_weight == 0.0 {
                    spatial_loss
                } else {
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
                    // Both sides arrive masked, so a pixel whose history was
                    // rejected contributes zero rather than a wrong number.
                    let mask = graph.input("temporal_mask", &[len]);
                    let target = graph.input("temporal_target", &[len]);
                    let masked = graph.mul(image, mask);
                    let temporal_loss = graph.mse_loss(masked, target);
                    let weight = graph.scalar(config.temporal_weight);
                    let scaled = graph.mul(temporal_loss, weight);
                    graph.add(spatial_loss, scaled)
                }
            }
            _ => {
                let target = graph.input("target", &[config.target_len_for_extent(extent)]);
                graph.mse_loss(output, target)
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
        });
        assert_eq!(c.cond_channels(), 17); // Ten stored plus seven temporal auxiliaries.
        assert_eq!(c.target_channels(), 3);
        assert_eq!(c.in_channels(), 17);
        assert!(c.validate().is_ok());
        c.temporal.as_mut().unwrap().features = crate::temporal::Features::Variance;
        assert_eq!(c.cond_channels(), 18);
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
