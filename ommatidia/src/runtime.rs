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

use std::sync::Arc;

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
    _pad2: u32,
    guide_spatial_denominator: f32,
    guide_depth_denominator: f32,
    guide_normal_power: f32,
    guide_albedo_denominator: f32,
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
            hr_depth: placeholder,
            hr_normal: placeholder,
            hr_albedo: placeholder,
            compose_blade_radiance: false,
            decode_blade_gbuffer: false,
            decode_hr_blade_gbuffer: false,
            has_high_resolution_gbuffer: false,
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
            hr_depth: depth,
            hr_normal: normal,
            hr_albedo: albedo,
            compose_blade_radiance: false,
            decode_blade_gbuffer: false,
            decode_hr_blade_gbuffer: false,
            has_high_resolution_gbuffer: false,
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
            hr_depth: gbuffer.depth,
            hr_normal: gbuffer.basis,
            hr_albedo: gbuffer.diffuse_albedo,
            compose_blade_radiance: false,
            decode_blade_gbuffer: true,
            decode_hr_blade_gbuffer: true,
            has_high_resolution_gbuffer: false,
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
            hr_depth: gbuffer.depth,
            hr_normal: gbuffer.basis,
            hr_albedo: gbuffer.diffuse_albedo,
            compose_blade_radiance: true,
            decode_blade_gbuffer: true,
            decode_hr_blade_gbuffer: true,
            has_high_resolution_gbuffer: false,
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
        if config.temporal.is_some()
            || config.prediction == model::Prediction::LowResolutionResidual
        {
            return Err(UpscalerError::Config(
                "this experimental checkpoint needs the history-enabled pack/unpack path".into(),
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
        let pack_layout = <PackData as gpu::ShaderData>::layout();
        let unpack_layout = <UnpackData as gpu::ShaderData>::layout();

        let pack_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "ommatidia-pack",
            data_layouts: &[&pack_layout],
            compute: pack_shader.at("pack"),
        });
        let unpack_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "ommatidia-unpack",
            data_layouts: &[&unpack_layout],
            compute: unpack_shader.at("unpack"),
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
    pub fn pack(&self, encoder: &mut gpu::CommandEncoder, inputs: &FrameInputs) {
        assert!(
            self.config.reconstruction_base != model::ReconstructionBase::HighResolutionGuided
                || inputs.has_high_resolution_gbuffer,
            "this checkpoint requires a high-resolution G-buffer"
        );
        let cond = self
            .session
            .input_buffer("cond")
            .expect("the graph always declares a conditioning input");
        let (width, height) = self.input_extent();

        let mut pass = encoder.compute("ommatidia-pack");
        let mut commands = pass.with(&self.pack_pipeline);
        commands.bind(
            0,
            &PackData {
                params: PackParams {
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
                },
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
        let mut pass = encoder.compute("ommatidia-unpack");
        let mut commands = pass.with(&self.unpack_pipeline);
        commands.bind(
            0,
            &UnpackData {
                params: UnpackParams {
                    width,
                    height,
                    scale: self.config.scale,
                    inverse_gain: 1.0 / self.config.residual_gain,
                    reconstruction_base: self.config.reconstruction_base as u32,
                    decode_blade_gbuffer: inputs.decode_blade_gbuffer as u32,
                    decode_hr_blade_gbuffer: inputs.decode_hr_blade_gbuffer as u32,
                    _pad2: 0,
                    guide_spatial_denominator: self.config.guide.spatial_denominator(),
                    guide_depth_denominator: self.config.guide.depth_denominator(),
                    guide_normal_power: self.config.guide.normal_power,
                    guide_albedo_denominator: self.config.guide.albedo_denominator(),
                },
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
