//! The host-facing upscaler.
//!
//! Blade users hand over their own `blade_graphics::Context`. Meganeura takes
//! it directly, so the network executes on the host's device and queue: no
//! second context, no external memory import, no cross-device copy. This does
//! require that both sides resolve to the same `blade-graphics` crate, which
//! the workspace `[patch]` section is there to enforce.
//!
//! A frame is three stages on one queue. Pack and unpack use the caller's
//! command encoder; Meganeura records the network on its own:
//!
//! 1. **Pack.** One dispatch reads the colour and G-buffer views and writes the
//!    conditioning tensor, applying the transforms in [`crate::transform`].
//! 2. **Step.** The network, once per sampler step.
//! 3. **Unpack.** One dispatch scatters the sub-pixel output into the target,
//!    adding the checkpoint's deterministic reconstruction base back.
//!
//! # Ordering
//!
//! Meganeura owns its own command encoder and submits independently, so the
//! caller has to submit the packing work before [`Upscaler::run`] and record
//! the unpacking after it. [`Upscaler::upscale`] does that sequencing; the
//! individual stages are exposed for callers that want to interleave other
//! work.

use std::{mem::size_of, sync::Arc};

use blade_graphics as gpu;

use crate::dataset::Plane;
use crate::model::{self, ModelConfig, Objective};

#[derive(blade_macros::ShaderData)]
struct PackData {
    params: PackParams,
    t_color: gpu::TextureView,
    t_diffuse_radiance: gpu::TextureView,
    t_specular_radiance: gpu::TextureView,
    t_emissive: gpu::TextureView,
    t_depth: gpu::TextureView,
    t_normal: gpu::TextureView,
    t_albedo: gpu::TextureView,
    t_specular: gpu::TextureView,
    cond: gpu::BufferPiece,
    base: gpu::BufferPiece,
}

#[derive(blade_macros::ShaderData)]
struct TemporalPackData {
    params: PackParams,
    t_color: gpu::TextureView,
    t_diffuse_radiance: gpu::TextureView,
    t_specular_radiance: gpu::TextureView,
    t_emissive: gpu::TextureView,
    t_depth: gpu::TextureView,
    t_normal: gpu::TextureView,
    t_albedo: gpu::TextureView,
    t_specular: gpu::TextureView,
    t_motion: gpu::TextureView,
    previous_low_history: gpu::BufferPiece,
    current_low_history: gpu::BufferPiece,
    cond: gpu::BufferPiece,
    base: gpu::BufferPiece,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PackParams {
    width: u32,
    height: u32,
    channels: u32,
    planes: u32,
    compose_blade_radiance: u32,
    decode_blade_gbuffer: u32,
    reconstruction_base: u32,
    _pad1: u32,
    guide_spatial_denominator: f32,
    guide_depth_denominator: f32,
    guide_normal_power: f32,
    guide_albedo_denominator: f32,
    history_frames: u32,
    history_ready: u32,
    motion_scale: f32,
    rejection_depth_delta: f32,
    rejection_normal_cosine: f32,
    rejection_albedo_delta2: f32,
    _pad2: [u32; 2],
}

#[derive(blade_macros::ShaderData)]
struct UnpackData {
    params: UnpackParams,
    base_pixels: gpu::BufferPiece,
    residual: gpu::BufferPiece,
    t_depth: gpu::TextureView,
    t_normal: gpu::TextureView,
    t_albedo: gpu::TextureView,
    t_hr_depth: gpu::TextureView,
    t_hr_normal: gpu::TextureView,
    t_hr_albedo: gpu::TextureView,
    output: gpu::TextureView,
}

#[derive(blade_macros::ShaderData)]
struct TemporalUnpackData {
    params: UnpackParams,
    base_pixels: gpu::BufferPiece,
    residual: gpu::BufferPiece,
    t_depth: gpu::TextureView,
    t_normal: gpu::TextureView,
    t_albedo: gpu::TextureView,
    t_hr_depth: gpu::TextureView,
    t_hr_normal: gpu::TextureView,
    t_hr_albedo: gpu::TextureView,
    t_motion: gpu::TextureView,
    t_history_output: gpu::TextureView,
    t_history_surface0: gpu::TextureView,
    t_history_surface1: gpu::TextureView,
    history_output: gpu::TextureView,
    history_surface0: gpu::TextureView,
    history_surface1: gpu::TextureView,
    output: gpu::TextureView,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct UnpackParams {
    width: u32,
    height: u32,
    scale: u32,
    inverse_gain: f32,
    reconstruction_base: u32,
    decode_blade_gbuffer: u32,
    decode_hr_blade_gbuffer: u32,
    kernel_radius: u32,
    demodulate: u32,
    demodulation_offset: f32,
    guide_spatial_denominator: f32,
    guide_depth_denominator: f32,
    guide_normal_power: f32,
    guide_albedo_denominator: f32,
    history_ready: u32,
    _pad1: u32,
    motion_scale: f32,
    rejection_depth_delta: f32,
    rejection_normal_cosine: f32,
    rejection_albedo_delta2: f32,
    _pad2: [u32; 2],
}

/// The textures a frame is reconstructed from.
///
/// Only the views for planes the model was configured with are read, but a
/// shader still needs every binding filled, so the unused ones have to point at
/// something. [`FrameInputs::color_only`] passes the colour view for all of
/// them, which is what a model conditioned on colour alone wants.
#[derive(Clone, Copy)]
pub struct FrameInputs {
    /// Linear radiance at input resolution, or a readable placeholder when
    /// Blade's split radiance views are composed instead.
    pub color: gpu::TextureView,
    /// Blade's demodulated diffuse radiance. Used by [`Self::from_blade`].
    pub diffuse_radiance: gpu::TextureView,
    /// Blade's specular radiance. Used by [`Self::from_blade`].
    pub specular_radiance: gpu::TextureView,
    /// Blade's emissive radiance. Used by [`Self::from_blade`].
    pub emissive: gpu::TextureView,
    /// View-space distance.
    pub depth: gpu::TextureView,
    /// World-space shading normal.
    pub normal: gpu::TextureView,
    /// Diffuse albedo.
    pub albedo: gpu::TextureView,
    /// Specular reflectance in RGB, roughness in alpha — Blade's own layout.
    pub specular: gpu::TextureView,
    /// Current-to-previous motion at input resolution. Native inputs are in
    /// input pixels; Blade's compact representation is decoded internally.
    pub motion: gpu::TextureView,
    /// Optional high-resolution view-space depth for geometry-aware upsampling.
    pub hr_depth: gpu::TextureView,
    /// Optional high-resolution world-space shading normal.
    pub hr_normal: gpu::TextureView,
    /// Optional high-resolution diffuse albedo.
    pub hr_albedo: gpu::TextureView,
    compose_blade_radiance: bool,
    decode_blade_gbuffer: bool,
    decode_hr_blade_gbuffer: bool,
    has_high_resolution_gbuffer: bool,
    has_motion: bool,
    motion_scale: f32,
}

impl FrameInputs {
    /// Inputs where every G-buffer plane is the same placeholder view.
    ///
    /// For a model conditioned on colour alone, which is what the first
    /// datasets carry.
    pub fn color_only(color: gpu::TextureView, placeholder: gpu::TextureView) -> Self {
        Self {
            color,
            diffuse_radiance: placeholder,
            specular_radiance: placeholder,
            emissive: placeholder,
            depth: placeholder,
            normal: placeholder,
            albedo: placeholder,
            specular: placeholder,
            motion: placeholder,
            hr_depth: placeholder,
            hr_normal: placeholder,
            hr_albedo: placeholder,
            compose_blade_radiance: false,
            decode_blade_gbuffer: false,
            decode_hr_blade_gbuffer: false,
            has_high_resolution_gbuffer: false,
            has_motion: false,
            motion_scale: 1.0,
        }
    }

    /// Read caller-owned linear colour and unpacked G-buffer textures.
    ///
    /// `normal` contains world-space XYZ. `specular` contains RGB F0 and
    /// roughness in alpha. This is the native integration path for renderers
    /// that do not use Blade's packed G-buffer conventions.
    pub fn from_textures(
        color: gpu::TextureView,
        depth: gpu::TextureView,
        normal: gpu::TextureView,
        albedo: gpu::TextureView,
        specular: gpu::TextureView,
    ) -> Self {
        Self {
            color,
            diffuse_radiance: color,
            specular_radiance: color,
            emissive: color,
            depth,
            normal,
            albedo,
            specular,
            motion: color,
            hr_depth: depth,
            hr_normal: normal,
            hr_albedo: albedo,
            compose_blade_radiance: false,
            decode_blade_gbuffer: false,
            decode_hr_blade_gbuffer: false,
            has_high_resolution_gbuffer: false,
            has_motion: false,
            motion_scale: 1.0,
        }
    }

    /// Add full-output-resolution geometry for joint bilateral upsampling.
    pub fn with_high_resolution_gbuffer(
        mut self,
        depth: gpu::TextureView,
        normal: gpu::TextureView,
        albedo: gpu::TextureView,
    ) -> Self {
        self.hr_depth = depth;
        self.hr_normal = normal;
        self.hr_albedo = albedo;
        self.decode_hr_blade_gbuffer = false;
        self.has_high_resolution_gbuffer = true;
        self
    }

    /// Add native current-to-previous motion in input-pixel units.
    ///
    /// A value `(1, 0)` means that the current texel reads history one input
    /// pixel to its right. The texture needs at least two floating-point
    /// channels and the same extent as `color`.
    pub fn with_motion(mut self, motion: gpu::TextureView) -> Self {
        self.motion = motion;
        self.has_motion = true;
        self.motion_scale = 1.0;
        self
    }

    /// Add Blade's compact `Rg8Snorm` motion view.
    pub fn with_blade_motion(mut self, motion: gpu::TextureView) -> Self {
        self.motion = motion;
        self.has_motion = true;
        self.motion_scale = 1.0 / 0.02;
        self
    }

    /// Add Blade's packed full-output-resolution primary-surface G-buffer.
    pub fn with_blade_high_resolution_gbuffer(
        mut self,
        gbuffer: blade_render::GBufferViews,
    ) -> Self {
        self.hr_depth = gbuffer.depth;
        self.hr_normal = gbuffer.basis;
        self.hr_albedo = gbuffer.diffuse_albedo;
        self.decode_hr_blade_gbuffer = true;
        self.has_high_resolution_gbuffer = true;
        self
    }

    /// Read the raw real-time estimate and G-buffer from a Blade ray tracer.
    ///
    /// Call this after `RayTracer::render`. To use Ommatidium *instead of*
    /// Blade's SVGF filter, pass `None` as that render call's denoiser config.
    /// The pack shader composes Blade's demodulated diffuse and specular lobes
    /// exactly as Blade's own post-process does, decodes the shading-basis
    /// quaternion, and preserves Blade's special depth value for sky pixels.
    pub fn from_blade(renderer: &blade_render::RayTracer) -> Self {
        Self::from_blade_views(renderer.view_radiance(), renderer.view_gbuffer())
    }

    /// Read composed colour from a path tracer while taking auxiliary planes
    /// from Blade's G-buffer.
    pub fn from_color_and_blade_gbuffer(
        color: gpu::TextureView,
        gbuffer: blade_render::GBufferViews,
    ) -> Self {
        Self {
            color,
            diffuse_radiance: color,
            specular_radiance: color,
            emissive: gbuffer.emissive,
            depth: gbuffer.depth,
            normal: gbuffer.basis,
            albedo: gbuffer.diffuse_albedo,
            specular: gbuffer.specular_f0,
            motion: gbuffer.motion,
            hr_depth: gbuffer.depth,
            hr_normal: gbuffer.basis,
            hr_albedo: gbuffer.diffuse_albedo,
            compose_blade_radiance: false,
            decode_blade_gbuffer: true,
            decode_hr_blade_gbuffer: true,
            has_high_resolution_gbuffer: false,
            has_motion: true,
            motion_scale: 1.0 / 0.02,
        }
    }

    /// Build inputs from views previously borrowed from a Blade ray tracer.
    pub fn from_blade_views(
        radiance: blade_render::RadianceViews,
        gbuffer: blade_render::GBufferViews,
    ) -> Self {
        Self {
            // Every shader binding must be populated. The color binding is
            // unused in this mode, so a readable radiance view is sufficient.
            color: radiance.diffuse,
            diffuse_radiance: radiance.diffuse,
            specular_radiance: radiance.specular,
            emissive: gbuffer.emissive,
            depth: gbuffer.depth,
            normal: gbuffer.basis,
            albedo: gbuffer.diffuse_albedo,
            specular: gbuffer.specular_f0,
            motion: gbuffer.motion,
            hr_depth: gbuffer.depth,
            hr_normal: gbuffer.basis,
            hr_albedo: gbuffer.diffuse_albedo,
            compose_blade_radiance: true,
            decode_blade_gbuffer: true,
            decode_hr_blade_gbuffer: true,
            has_high_resolution_gbuffer: false,
            has_motion: true,
            motion_scale: 1.0 / 0.02,
        }
    }
}

#[derive(Debug)]
pub enum UpscalerError {
    /// The configuration cannot be expressed as a graph.
    Config(String),
    /// The checkpoint could not be read.
    Checkpoint(String),
}

impl std::fmt::Display for UpscalerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Config(ref message) => write!(f, "invalid model configuration: {message}"),
            Self::Checkpoint(ref message) => write!(f, "cannot load the checkpoint: {message}"),
        }
    }
}

impl std::error::Error for UpscalerError {}

const LOW_HISTORY_PLANES: u64 = 13;

struct HistoryTexture {
    texture: gpu::Texture,
    view: gpu::TextureView,
}

impl HistoryTexture {
    fn new(
        context: &gpu::Context,
        name: &str,
        format: gpu::TextureFormat,
        size: gpu::Extent,
    ) -> Self {
        let texture = context.create_texture(gpu::TextureDesc {
            name,
            format,
            size,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::RESOURCE | gpu::TextureUsage::STORAGE,
            sample_count: 1,
            external: None,
        });
        let view = context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name,
                format,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );
        Self { texture, view }
    }

    fn destroy(self, context: &gpu::Context) {
        context.destroy_texture_view(self.view);
        context.destroy_texture(self.texture);
    }
}

struct TemporalRuntime {
    low_history: [gpu::Buffer; 2],
    output: [HistoryTexture; 2],
    surface0: [HistoryTexture; 2],
    surface1: [HistoryTexture; 2],
    /// Index containing the completed previous frame. The other index is
    /// written by the frame currently being recorded.
    previous: usize,
    ready: bool,
    textures_initialized: bool,
}

impl TemporalRuntime {
    fn new(context: &gpu::Context, input_extent: [u32; 2], scale: u32) -> Self {
        let low_bytes = input_extent[0] as u64
            * input_extent[1] as u64
            * LOW_HISTORY_PLANES
            * size_of::<f32>() as u64;
        let low_history = std::array::from_fn(|index| {
            context.create_buffer(gpu::BufferDesc {
                name: if index == 0 {
                    "ommatidia-low-history-0"
                } else {
                    "ommatidia-low-history-1"
                },
                size: low_bytes,
                memory: gpu::Memory::Device,
            })
        });
        let output_extent = gpu::Extent {
            width: input_extent[0] * scale,
            height: input_extent[1] * scale,
            depth: 1,
        };
        let output = std::array::from_fn(|index| {
            HistoryTexture::new(
                context,
                if index == 0 {
                    "ommatidia-output-history-0"
                } else {
                    "ommatidia-output-history-1"
                },
                gpu::TextureFormat::Rgba16Float,
                output_extent,
            )
        });
        let surface0 = std::array::from_fn(|index| {
            HistoryTexture::new(
                context,
                if index == 0 {
                    "ommatidia-surface-history-0a"
                } else {
                    "ommatidia-surface-history-1a"
                },
                gpu::TextureFormat::Rgba16Float,
                output_extent,
            )
        });
        let surface1 = std::array::from_fn(|index| {
            HistoryTexture::new(
                context,
                if index == 0 {
                    "ommatidia-surface-history-0b"
                } else {
                    "ommatidia-surface-history-1b"
                },
                gpu::TextureFormat::Rgba8Unorm,
                output_extent,
            )
        });
        Self {
            low_history,
            output,
            surface0,
            surface1,
            previous: 0,
            ready: false,
            textures_initialized: false,
        }
    }

    fn current(&self) -> usize {
        self.previous ^ 1
    }

    fn initialize_textures(&mut self, encoder: &mut gpu::CommandEncoder) {
        if self.textures_initialized {
            return;
        }
        for texture in self
            .output
            .iter()
            .chain(&self.surface0)
            .chain(&self.surface1)
        {
            encoder.init_texture(texture.texture);
        }
        self.textures_initialized = true;
    }

    fn advance(&mut self) {
        self.previous = self.current();
        self.ready = true;
    }

    fn reset(&mut self) {
        self.ready = false;
    }

    fn destroy(self, context: &gpu::Context) {
        for buffer in self.low_history {
            context.destroy_buffer(buffer);
        }
        for texture in self.output {
            texture.destroy(context);
        }
        for texture in self.surface0 {
            texture.destroy(context);
        }
        for texture in self.surface1 {
            texture.destroy(context);
        }
    }
}

/// Reconstructs a high resolution frame from a low resolution one.
pub struct Upscaler {
    context: Arc<gpu::Context>,
    session: meganeura::Session,
    config: ModelConfig,
    input_extent: [u32; 2],
    schedule: crate::diffusion::Schedule,
    sampler_steps: usize,
    pack_pipeline: gpu::ComputePipeline,
    unpack_pipeline: gpu::ComputePipeline,
    /// Host-side scratch for the diffusion sampler's state. Empty for direct
    /// checkpoints, whose output stays on the GPU.
    x: Vec<f32>,
    next: Vec<f32>,
    /// Device copy of the host-side diffusion sampler result. Direct
    /// checkpoints bind Meganeura's pinned graph output instead.
    residual_buffer: Option<gpu::Buffer>,
    /// Linear RGB reconstruction base produced during packing.
    base_buffer: gpu::Buffer,
    temporal: Option<TemporalRuntime>,
    seed: u64,
}

impl Upscaler {
    /// Build an upscaler on the host's context from a trained checkpoint.
    ///
    /// `stem` is the checkpoint stem: `<stem>.ron` supplies the configuration
    /// and `<stem>.safetensors` the weights. The configuration is not a
    /// parameter because the weights only mean anything in the graph they were
    /// trained in.
    pub fn from_checkpoint(
        context: Arc<gpu::Context>,
        stem: impl AsRef<std::path::Path>,
        sampler_steps: usize,
        timesteps: usize,
    ) -> Result<Self, UpscalerError> {
        let (config, _) = crate::checkpoint::load_config(&stem)
            .map_err(|e| UpscalerError::Checkpoint(e.to_string()))?;
        Self::from_checkpoint_for_extent(
            context,
            stem,
            [config.tile, config.tile],
            sampler_steps,
            timesteps,
        )
    }

    /// Build a checkpoint for a full, potentially rectangular Blade frame.
    ///
    /// Checkpoint parameters do not depend on spatial size. Meganeura's graph
    /// does, so this recompiles the same network at `input_extent` while
    /// loading the original weights.
    pub fn from_checkpoint_for_extent(
        context: Arc<gpu::Context>,
        stem: impl AsRef<std::path::Path>,
        input_extent: [u32; 2],
        sampler_steps: usize,
        timesteps: usize,
    ) -> Result<Self, UpscalerError> {
        let (mut config, paths) = crate::checkpoint::load_config(stem)
            .map_err(|e| UpscalerError::Checkpoint(e.to_string()))?;
        if config.prediction == model::Prediction::LowResolutionResidual {
            return Err(UpscalerError::Config(
                "this experimental checkpoint needs the history-enabled pack/unpack path".into(),
            ));
        }
        if let Some(temporal) = config.temporal
            && (config.objective != Objective::Direct
                || config.prediction != model::Prediction::SubpixelKernel
                || !temporal.previous_output
                || temporal.unrejected_tap)
        {
            return Err(UpscalerError::Config(
                "native temporal inference supports direct kernel checkpoints with the \
                 previous-output mix and no unrejected gather tap"
                    .into(),
            ));
        }
        // A checkpoint may have been trained with a batch; a frame is one tile.
        config.batch = 1;

        let model =
            model::build_for_extent(&config, false, input_extent).map_err(UpscalerError::Config)?;
        let mut session = meganeura::train::build(
            &model.graph,
            meganeura::SessionConfig {
                mode: meganeura::Mode::Inference,
                // The whole point: the network runs on the caller's device.
                gpu: Some(Arc::clone(&context)),
                ..Default::default()
            },
        )
        .0;
        session
            .load_checkpoint(&paths.weights)
            .map_err(|e| UpscalerError::Checkpoint(e.to_string()))?;

        let shader_source = |source: &str| {
            context.create_shader(gpu::ShaderDesc {
                source,
                naga_module: None,
            })
        };
        let pack_shader = shader_source(include_str!("shaders/pack.wgsl"));
        let unpack_shader = shader_source(include_str!("shaders/unpack.wgsl"));
        let temporal_checkpoint = config.temporal.is_some();
        let pack_layout = if temporal_checkpoint {
            <TemporalPackData as gpu::ShaderData>::layout()
        } else {
            <PackData as gpu::ShaderData>::layout()
        };
        let unpack_layout = if temporal_checkpoint {
            <TemporalUnpackData as gpu::ShaderData>::layout()
        } else {
            <UnpackData as gpu::ShaderData>::layout()
        };

        let pack_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "ommatidia-pack",
            data_layouts: &[&pack_layout],
            compute: pack_shader.at(if temporal_checkpoint {
                "pack_temporal"
            } else {
                "pack"
            }),
        });
        let unpack_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "ommatidia-unpack",
            data_layouts: &[&unpack_layout],
            compute: unpack_shader.at(if temporal_checkpoint {
                "unpack_temporal"
            } else {
                "unpack"
            }),
        });

        let per_slot = config.target_len_for_extent(input_extent);
        let residual_buffer = (config.objective == Objective::Diffusion).then(|| {
            context.create_buffer(gpu::BufferDesc {
                name: "ommatidia-residual",
                size: per_slot as u64 * 4,
                // The diffusion sampler still runs on the host.
                memory: gpu::Memory::Shared,
            })
        });
        let base_buffer = context.create_buffer(gpu::BufferDesc {
            name: "ommatidia-reconstruction-base",
            size: (input_extent[0] * input_extent[1] * 3) as u64 * 4,
            memory: gpu::Memory::Device,
        });
        let temporal =
            temporal_checkpoint.then(|| TemporalRuntime::new(&context, input_extent, config.scale));

        let host_scratch_len = if config.objective == Objective::Diffusion {
            per_slot
        } else {
            0
        };
        Ok(Self {
            context,
            session,
            schedule: crate::diffusion::Schedule::cosine(timesteps),
            sampler_steps,
            config,
            input_extent,
            pack_pipeline,
            unpack_pipeline,
            x: vec![0.0; host_scratch_len],
            next: vec![0.0; host_scratch_len],
            residual_buffer,
            base_buffer,
            temporal,
            seed: 0,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Input extent the network was compiled for.
    pub fn input_extent(&self) -> (u32, u32) {
        (self.input_extent[0], self.input_extent[1])
    }

    /// Output extent this produces.
    pub fn output_extent(&self) -> (u32, u32) {
        (
            self.input_extent[0] * self.config.scale,
            self.input_extent[1] * self.config.scale,
        )
    }

    /// Format the output texture has to have, matching the unpack shader's
    /// storage binding.
    pub const OUTPUT_FORMAT: gpu::TextureFormat = gpu::TextureFormat::Rgba16Float;

    /// Record the packing dispatch onto `encoder`.
    ///
    /// The caller submits this before [`Self::run`], because meganeura submits
    /// on its own encoder and the network has to see the packed tensor.
    pub fn pack(&mut self, encoder: &mut gpu::CommandEncoder, inputs: &FrameInputs) {
        assert!(
            (self.config.reconstruction_base != model::ReconstructionBase::HighResolutionGuided
                && !self.config.demodulate)
                || inputs.has_high_resolution_gbuffer,
            "this checkpoint requires a high-resolution G-buffer"
        );
        if self.config.temporal.is_some() {
            assert!(
                inputs.has_high_resolution_gbuffer,
                "a temporal checkpoint requires an output-resolution G-buffer"
            );
            assert!(
                inputs.has_motion,
                "a temporal checkpoint requires current-to-previous motion"
            );
        }
        if let Some(temporal) = self.temporal.as_mut() {
            temporal.initialize_textures(encoder);
        }
        let cond = self
            .session
            .input_buffer("cond")
            .expect("the graph always declares a conditioning input");
        let (width, height) = self.input_extent();
        let temporal_config = self.config.temporal;
        let rejection = temporal_config
            .map(|temporal| temporal.rejection)
            .unwrap_or_default();
        let history_ready = self.temporal.as_ref().is_some_and(|history| history.ready);
        let params = PackParams {
            width,
            height,
            channels: self.config.cond_channels(),
            planes: self.config.cond_planes.bits(),
            compose_blade_radiance: inputs.compose_blade_radiance as u32,
            decode_blade_gbuffer: inputs.decode_blade_gbuffer as u32,
            reconstruction_base: self.config.reconstruction_base as u32,
            _pad1: 0,
            guide_spatial_denominator: self.config.guide.spatial_denominator(),
            guide_depth_denominator: self.config.guide.depth_denominator(),
            guide_normal_power: self.config.guide.normal_power,
            guide_albedo_denominator: self.config.guide.albedo_denominator(),
            history_frames: temporal_config.map_or(1, |temporal| temporal.frames),
            history_ready: history_ready as u32,
            motion_scale: inputs.motion_scale,
            rejection_depth_delta: rejection.depth_delta,
            rejection_normal_cosine: rejection.normal_cosine,
            rejection_albedo_delta2: rejection.albedo_delta2,
            _pad2: [0; 2],
        };

        let mut pass = encoder.compute("ommatidia-pack");
        let mut commands = pass.with(&self.pack_pipeline);
        if let Some(temporal) = &self.temporal {
            let current = temporal.current();
            commands.bind(
                0,
                &TemporalPackData {
                    params,
                    t_color: inputs.color,
                    t_diffuse_radiance: inputs.diffuse_radiance,
                    t_specular_radiance: inputs.specular_radiance,
                    t_emissive: inputs.emissive,
                    t_depth: inputs.depth,
                    t_normal: inputs.normal,
                    t_albedo: inputs.albedo,
                    t_specular: inputs.specular,
                    t_motion: inputs.motion,
                    previous_low_history: temporal.low_history[temporal.previous].into(),
                    current_low_history: temporal.low_history[current].into(),
                    cond,
                    base: self.base_buffer.into(),
                },
            );
        } else {
            commands.bind(
                0,
                &PackData {
                    params,
                    t_color: inputs.color,
                    t_diffuse_radiance: inputs.diffuse_radiance,
                    t_specular_radiance: inputs.specular_radiance,
                    t_emissive: inputs.emissive,
                    t_depth: inputs.depth,
                    t_normal: inputs.normal,
                    t_albedo: inputs.albedo,
                    t_specular: inputs.specular,
                    cond,
                    base: self.base_buffer.into(),
                },
            );
        }
        commands.dispatch([width.div_ceil(8), height.div_ceil(8), 1]);
    }

    /// Run the network, leaving the predicted residual in its output buffer.
    ///
    /// Under [`Objective::Direct`] this is one forward pass. Under
    /// [`Objective::Diffusion`] it walks the sampler, which currently costs a
    /// host roundtrip per step — see the note on [`Self::upscale`].
    pub fn run(&mut self) {
        let per_slot = self.config.target_len_for_extent(self.input_extent);
        match self.config.objective {
            Objective::Direct => {
                self.session.step();
            }
            Objective::Diffusion => {
                let mut rng = crate::rng::Rng::new(self.seed);
                crate::diffusion::fill_normal(&mut rng, &mut self.x);

                let steps = self.schedule.sampling_timesteps(self.sampler_steps);
                for (i, &t) in steps.iter().enumerate() {
                    let embedding = crate::diffusion::timestep_embedding(
                        t,
                        self.config.time_input_dim as usize,
                        MAX_PERIOD,
                    );
                    self.session.set_input("t_emb", &embedding);
                    self.session.set_input("x_t", &self.x);
                    self.session.step();
                    self.session.wait();
                    let x0 = self.session.read_output(per_slot);
                    self.schedule.ddim_step(
                        &self.x,
                        &x0,
                        t,
                        steps.get(i + 1).copied(),
                        self.config.residual_gain,
                        &mut self.next,
                    );
                    self.x.copy_from_slice(&self.next);
                }
            }
        }
    }

    /// Record the unpacking dispatch, writing the reconstructed frame.
    ///
    /// `output` must be [`Self::OUTPUT_FORMAT`] at [`Self::output_extent`],
    /// created with `TextureUsage::STORAGE`.
    pub fn unpack(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        inputs: &FrameInputs,
        output: gpu::TextureView,
    ) {
        let residual = match self.config.objective {
            Objective::Direct => self
                .session
                .output_buffer(0)
                .expect("the inference graph always has one pinned output"),
            Objective::Diffusion => {
                let buffer = self
                    .residual_buffer
                    .expect("diffusion allocates host-visible sampler output");
                // The sampler ran on the host, so its result goes back to the
                // device. Host-coherent memory becomes visible on submit.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.x.as_ptr(),
                        buffer.data() as *mut f32,
                        self.x.len(),
                    );
                }
                buffer.into()
            }
        };

        let (width, height) = self.input_extent();
        let temporal_config = self.config.temporal;
        let rejection = temporal_config
            .map(|temporal| temporal.rejection)
            .unwrap_or_default();
        let history_ready = self.temporal.as_ref().is_some_and(|history| history.ready);
        let params = UnpackParams {
            width,
            height,
            scale: self.config.scale,
            inverse_gain: 1.0 / self.config.residual_gain,
            reconstruction_base: self.config.reconstruction_base as u32,
            decode_blade_gbuffer: inputs.decode_blade_gbuffer as u32,
            decode_hr_blade_gbuffer: inputs.decode_hr_blade_gbuffer as u32,
            kernel_radius: self.config.kernel_radius,
            demodulate: self.config.demodulate as u32,
            demodulation_offset: self.config.demodulation_offset,
            guide_spatial_denominator: self.config.guide.spatial_denominator(),
            guide_depth_denominator: self.config.guide.depth_denominator(),
            guide_normal_power: self.config.guide.normal_power,
            guide_albedo_denominator: self.config.guide.albedo_denominator(),
            history_ready: history_ready as u32,
            _pad1: 0,
            motion_scale: inputs.motion_scale,
            rejection_depth_delta: rejection.depth_delta,
            rejection_normal_cosine: rejection.normal_cosine,
            rejection_albedo_delta2: rejection.albedo_delta2,
            _pad2: [0; 2],
        };
        let mut pass = encoder.compute("ommatidia-unpack");
        let mut commands = pass.with(&self.unpack_pipeline);
        if let Some(temporal) = self.temporal.as_mut() {
            let current = temporal.current();
            commands.bind(
                0,
                &TemporalUnpackData {
                    params,
                    base_pixels: self.base_buffer.into(),
                    residual,
                    t_depth: inputs.depth,
                    t_normal: inputs.normal,
                    t_albedo: inputs.albedo,
                    t_hr_depth: inputs.hr_depth,
                    t_hr_normal: inputs.hr_normal,
                    t_hr_albedo: inputs.hr_albedo,
                    t_motion: inputs.motion,
                    t_history_output: temporal.output[temporal.previous].view,
                    t_history_surface0: temporal.surface0[temporal.previous].view,
                    t_history_surface1: temporal.surface1[temporal.previous].view,
                    history_output: temporal.output[current].view,
                    history_surface0: temporal.surface0[current].view,
                    history_surface1: temporal.surface1[current].view,
                    output,
                },
            );
            commands.dispatch([width.div_ceil(8), height.div_ceil(8), 1]);
            temporal.advance();
            return;
        }
        commands.bind(
            0,
            &UnpackData {
                params,
                base_pixels: self.base_buffer.into(),
                residual,
                t_depth: inputs.depth,
                t_normal: inputs.normal,
                t_albedo: inputs.albedo,
                t_hr_depth: inputs.hr_depth,
                t_hr_normal: inputs.hr_normal,
                t_hr_albedo: inputs.hr_albedo,
                output,
            },
        );
        commands.dispatch([width.div_ceil(8), height.div_ceil(8), 1]);
    }

    /// Reconstruct one frame, start to finish.
    ///
    /// # What this does to your encoder
    ///
    /// It submits it. Meganeura runs on its own encoder, so the packed tensor
    /// has to reach the device before the network steps, which means anything
    /// already recorded goes out with it. The encoder is then restarted and
    /// the unpacking recorded into it, for the caller to submit.
    ///
    /// A caller that wants to keep control of its own submissions should drive
    /// [`Self::pack`], [`Self::run`], and [`Self::unpack`] directly.
    ///
    /// # Cost
    ///
    /// Under diffusion this is `sampler_steps` network evaluations with a host
    /// roundtrip between each, which is nowhere near a frame budget. That is
    /// the expected shape of the first milestone: [`Objective::Direct`] is the
    /// fast path, and moving the sampler arithmetic into the graph so the
    /// whole chain stays on the device is the next step for the diffusion one.
    pub fn upscale(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        inputs: &FrameInputs,
        output: gpu::TextureView,
    ) {
        self.pack(encoder, inputs);
        self.context.submit(encoder);
        // The network's dispatches go onto meganeura's encoder on the same
        // queue, so submission order is sufficient; no CPU wait is needed.

        self.run();

        encoder.start();
        self.unpack(encoder, inputs, output);
    }

    /// Seed the sampler's starting noise, for reproducible output.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Discard recurrent history before the next frame.
    ///
    /// Call this for camera cuts, resolution changes, exposure discontinuities,
    /// or whenever the motion/surface inputs no longer describe the preceding
    /// output. The next frame still seeds the library-owned history resources,
    /// but its reconstruction is exactly the current-frame spatial path.
    pub fn reset_history(&mut self) {
        if let Some(temporal) = self.temporal.as_mut() {
            temporal.reset();
        }
    }

    /// Whether the loaded checkpoint consumes recurrent frame history.
    pub fn is_temporal(&self) -> bool {
        self.temporal.is_some()
    }

    /// Device memory reserved for recurrent history, in bytes.
    ///
    /// This excludes the model tensors and spatial pack/unpack buffers. A
    /// spatial checkpoint returns zero. Temporal storage currently consists
    /// of two low-resolution accumulation buffers plus ping-ponged output,
    /// depth/normal, and albedo textures at reconstructed resolution.
    pub fn temporal_history_bytes(&self) -> u64 {
        if self.temporal.is_none() {
            return 0;
        }
        let input_texels = self.input_extent[0] as u64 * self.input_extent[1] as u64;
        let output_texels = input_texels * self.config.scale as u64 * self.config.scale as u64;
        // Two 13-plane f32 accumulation buffers. At output resolution each
        // ping-pong set has two RGBA16F textures and one RGBA8 texture.
        2 * input_texels * LOW_HISTORY_PLANES * size_of::<f32>() as u64
            + 2 * output_texels * (8 + 8 + 4)
    }

    /// Read the scaled residual produced by the most recent run.
    ///
    /// This is a diagnostic path and waits for the GPU. The realtime direct
    /// path binds the same output buffer from [`Self::unpack`] without reading
    /// it on the host.
    pub fn read_residual(&mut self) -> Vec<f32> {
        match self.config.objective {
            Objective::Direct => {
                self.session.wait();
                self.session
                    .read_output(self.config.target_len_for_extent(self.input_extent))
            }
            Objective::Diffusion => self.x.clone(),
        }
    }

    /// The underlying meganeura session.
    pub fn session(&mut self) -> &mut meganeura::Session {
        &mut self.session
    }

    /// Planes this upscaler actually reads.
    pub fn required_planes(&self) -> impl Iterator<Item = Plane> {
        self.config.cond_planes.iter()
    }

    pub fn destroy(&mut self) {
        self.session.wait();
        if let Some(buffer) = self.residual_buffer.take() {
            self.context.destroy_buffer(buffer);
        }
        if let Some(temporal) = self.temporal.take() {
            temporal.destroy(&self.context);
        }
        self.context.destroy_buffer(self.base_buffer);
        self.context
            .destroy_compute_pipeline(&mut self.pack_pipeline);
        self.context
            .destroy_compute_pipeline(&mut self.unpack_pipeline);
    }
}

/// Frequency spread of the timestep embedding. Must match the trainer.
const MAX_PERIOD: f32 = 10_000.0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::PlaneSet;

    #[test]
    fn plane_bits_agree_with_the_shader_constants() {
        // The pack shader hardcodes these, because a WGSL constant cannot be
        // derived from a Rust enum. If the enum is reordered, this catches it.
        assert_eq!(PlaneSet::new().with(Plane::Color).bits(), 1);
        assert_eq!(PlaneSet::new().with(Plane::Depth).bits(), 2);
        assert_eq!(PlaneSet::new().with(Plane::Normal).bits(), 4);
        assert_eq!(PlaneSet::new().with(Plane::DiffuseAlbedo).bits(), 8);
        assert_eq!(PlaneSet::new().with(Plane::SpecularF0).bits(), 16);
        assert_eq!(PlaneSet::new().with(Plane::Roughness).bits(), 32);
    }

    #[test]
    fn shaders_parse() {
        // Cheap insurance: a typo in the WGSL would otherwise only surface on
        // a machine with a GPU.
        for (name, source) in [
            ("pack", include_str!("shaders/pack.wgsl")),
            ("unpack", include_str!("shaders/unpack.wgsl")),
        ] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|e| panic!("{name}.wgsl: {}", e.emit_to_string(source)));
            // Blade assigns groups and bindings from the `ShaderData` layout at
            // pipeline creation, so the source carries none — the same reason
            // blade's own `parse_shaders` test clears this flag.
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
                naga::valid::Capabilities::all(),
            );
            validator
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}.wgsl failed validation: {e:?}"));
        }
    }
}
