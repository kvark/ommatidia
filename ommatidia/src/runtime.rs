//! The host-facing upscaler.
//!
//! Blade users hand over their own `blade_graphics::Context`. Meganeura takes
//! it directly, so the network executes on the host's device and queue: no
//! second context, no external memory import, no cross-device copy. This does
//! require that both sides resolve to the same `blade-graphics` crate, which
//! the workspace `[patch]` section is there to enforce.
//!
//! A frame is three stages, all recorded onto the caller's command encoder:
//!
//! 1. **Pack.** One dispatch reads the colour and G-buffer views and writes the
//!    conditioning tensor, applying the transforms in [`crate::transform`].
//! 2. **Step.** The network, once per sampler step.
//! 3. **Unpack.** One dispatch scatters the sub-pixel output into the target,
//!    adding the nearest-neighbour base back.
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
    t_depth: gpu::TextureView,
    t_normal: gpu::TextureView,
    t_albedo: gpu::TextureView,
    t_specular: gpu::TextureView,
    cond: gpu::BufferPiece,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct PackParams {
    width: u32,
    height: u32,
    channels: u32,
    planes: u32,
}

#[derive(blade_macros::ShaderData)]
struct UnpackData {
    params: UnpackParams,
    t_color: gpu::TextureView,
    residual: gpu::BufferPiece,
    output: gpu::TextureView,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct UnpackParams {
    width: u32,
    height: u32,
    scale: u32,
    inverse_gain: f32,
}

/// The textures a frame is reconstructed from.
///
/// Only the views for planes the model was configured with are read, but a
/// shader still needs every binding filled, so the unused ones have to point at
/// something. [`FrameInputs::color_only`] passes the colour view for all of
/// them, which is what a model conditioned on colour alone wants.
#[derive(Clone, Copy)]
pub struct FrameInputs {
    /// Linear radiance at input resolution. Always required.
    pub color: gpu::TextureView,
    /// View-space distance.
    pub depth: gpu::TextureView,
    /// World-space shading normal.
    pub normal: gpu::TextureView,
    /// Diffuse albedo.
    pub albedo: gpu::TextureView,
    /// Specular reflectance in RGB, roughness in alpha — Blade's own layout.
    pub specular: gpu::TextureView,
}

impl FrameInputs {
    /// Inputs where every G-buffer plane is the same placeholder view.
    ///
    /// For a model conditioned on colour alone, which is what the first
    /// datasets carry.
    pub fn color_only(color: gpu::TextureView, placeholder: gpu::TextureView) -> Self {
        Self {
            color,
            depth: placeholder,
            normal: placeholder,
            albedo: placeholder,
            specular: placeholder,
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
    schedule: crate::diffusion::Schedule,
    sampler_steps: usize,
    pack_pipeline: gpu::ComputePipeline,
    unpack_pipeline: gpu::ComputePipeline,
    /// Host-side scratch for the sampler's state.
    x: Vec<f32>,
    next: Vec<f32>,
    /// Device copy of the residual, read by the unpack shader.
    ///
    /// The sampler runs on the host, so the result has to come back. Even for
    /// the single-pass objective it goes through here, because meganeura does
    /// not expose the `blade_graphics::Buffer` behind a graph output — binding
    /// the network's own output in place would save this copy and is the
    /// obvious thing to do once that accessor exists.
    residual_buffer: gpu::Buffer,
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
        let (mut config, paths) = crate::checkpoint::load_config(stem)
            .map_err(|e| UpscalerError::Checkpoint(e.to_string()))?;
        // A checkpoint may have been trained with a batch; a frame is one tile.
        config.batch = 1;

        let model = model::build(&config, false).map_err(UpscalerError::Config)?;
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

        let per_slot = (config.target_channels() * config.tile * config.tile) as usize;
        let residual_buffer = context.create_buffer(gpu::BufferDesc {
            name: "ommatidia-residual",
            size: per_slot as u64 * 4,
            // Host-visible and device-local, so the upload is a memcpy with no
            // staging buffer and no transfer pass.
            memory: gpu::Memory::Shared,
        });

        Ok(Self {
            context,
            session,
            schedule: crate::diffusion::Schedule::cosine(timesteps),
            sampler_steps,
            config,
            pack_pipeline,
            unpack_pipeline,
            x: vec![0.0; per_slot],
            next: vec![0.0; per_slot],
            residual_buffer,
            seed: 0,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Input extent the network was compiled for.
    pub fn input_extent(&self) -> (u32, u32) {
        (self.config.tile, self.config.tile)
    }

    /// Output extent this produces.
    pub fn output_extent(&self) -> (u32, u32) {
        let tile = self.config.tile * self.config.scale;
        (tile, tile)
    }

    /// Format the output texture has to have, matching the unpack shader's
    /// storage binding.
    pub const OUTPUT_FORMAT: gpu::TextureFormat = gpu::TextureFormat::Rgba16Float;

    /// Record the packing dispatch onto `encoder`.
    ///
    /// The caller submits this before [`Self::run`], because meganeura submits
    /// on its own encoder and the network has to see the packed tensor.
    pub fn pack(&self, encoder: &mut gpu::CommandEncoder, inputs: &FrameInputs) {
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
                },
                t_color: inputs.color,
                t_depth: inputs.depth,
                t_normal: inputs.normal,
                t_albedo: inputs.albedo,
                t_specular: inputs.specular,
                cond,
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
        let per_slot =
            (self.config.target_channels() * self.config.tile * self.config.tile) as usize;
        match self.config.objective {
            Objective::Direct => {
                self.session.step();
                self.session.wait();
                let out = self.session.read_output(per_slot);
                self.x.copy_from_slice(&out);
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
        color: gpu::TextureView,
        output: gpu::TextureView,
    ) {
        // The sampler ran on the host, so the result goes back to the device.
        // The buffer is host-coherent, and the write is made visible to the
        // dispatch below by the implicit host-domain barrier on submit.
        //
        // Safety: the buffer was allocated with exactly `self.x.len()` floats
        // and is `Memory::Shared`, so the pointer is mapped and writable.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.x.as_ptr(),
                self.residual_buffer.data() as *mut f32,
                self.x.len(),
            );
        }

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
                },
                t_color: color,
                residual: self.residual_buffer.into(),
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
        let sync_point = self.context.submit(encoder);
        // The network's dispatches go onto meganeura's encoder, on the same
        // queue, so they are ordered after this submission.
        let _ = self.context.wait_for(&sync_point, !0);

        self.run();

        encoder.start();
        self.unpack(encoder, inputs.color, output);
    }

    /// Seed the sampler's starting noise, for reproducible output.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// The scaled residual the last [`Self::run`] produced.
    ///
    /// Exposed so a test can check the unpack shader against
    /// [`crate::batch::assemble`] on the very same values.
    pub fn residual(&self) -> &[f32] {
        &self.x
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
        self.context.destroy_buffer(self.residual_buffer);
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
