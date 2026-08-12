//! Headless rendering of one low/high resolution pair.
//!
//! The low-resolution side is a sparse path trace and the high-resolution side
//! is the same path tracer accumulated toward convergence. Raw ReSTIR and SVGF
//! inputs remain available as comparison baselines.

use blade_graphics as gpu;

/// Frames the real-time estimator gets to settle its reservoirs.
///
/// Temporal reuse means the first frame after a camera cut is not what the
/// renderer actually shows, so the input has to be a settled one.
pub const RESTIR_FRAMES: usize = 8;
/// Practical path depth presented to the reconstruction network.
const INPUT_MAX_BOUNCES: u32 = 3;
/// Reference depth: long enough for the procedural scenes to settle, with
/// Blade's path tracer applying Russian roulette after the fourth bounce.
/// Keeping this separate from the input avoids training against a clean but
/// systematically truncated target.
const REFERENCE_MAX_BOUNCES: u32 = 8;
/// Limit one command submission's retained scene resources. Reference captures
/// can run for thousands of frames; keeping every transient BLAS/TLAS alive
/// until the final readback otherwise turns convergence into an OOM.
const FRAMES_PER_SUBMISSION: usize = 32;

/// An offscreen colour target plus its readback buffer.
///
/// `Rgba32Float` because the renderer writes unbounded linear radiance into
/// it: `PostProcConfig::tone_map` is cleared, so nothing has compressed the
/// values into display range, and a fixed point target would clamp everything
/// above 1.0 flat.
pub struct Target {
    texture: gpu::Texture,
    view: gpu::TextureView,
    readback: gpu::Buffer,
    pub size: gpu::Extent,
}

impl Target {
    pub fn new(context: &gpu::Context, size: gpu::Extent) -> Self {
        let format = gpu::TextureFormat::Rgba32Float;
        let texture = context.create_texture(gpu::TextureDesc {
            name: "capture",
            format,
            size,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::TARGET
                | gpu::TextureUsage::RESOURCE
                | gpu::TextureUsage::COPY,
            sample_count: 1,
            external: None,
        });
        let view = context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "capture",
                format,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );
        let readback = context.create_buffer(gpu::BufferDesc {
            name: "capture-readback",
            size: (size.width * size.height) as u64 * 16,
            memory: gpu::Memory::Shared,
        });
        Self {
            texture,
            view,
            readback,
            size,
        }
    }

    pub fn view(&self) -> gpu::TextureView {
        self.view
    }

    pub fn texture(&self) -> gpu::Texture {
        self.texture
    }

    /// Copy the target back, returning interleaved linear RGB.
    ///
    /// Submits and waits: the generator is offline, and overlapping the
    /// readback with the next scene would only complicate it.
    pub fn read_linear(
        &self,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Vec<f32> {
        {
            let mut transfer = encoder.transfer("capture-readback");
            transfer.copy_texture_to_buffer(
                self.texture.into(),
                self.readback.into(),
                self.size.width * 16,
                self.size,
            );
        }
        let sync_point = context.submit(encoder);
        assert!(
            context.wait_for(&sync_point, 30_000).unwrap(),
            "GPU timed out during readback"
        );

        let texel_count = (self.size.width * self.size.height) as usize;
        let mut mapped = vec![0.0f32; texel_count * 4];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.readback.data() as *const f32,
                mapped.as_mut_ptr(),
                mapped.len(),
            );
        }

        let mut out = Vec::with_capacity(texel_count * 3);
        for texel in mapped.chunks_exact(4) {
            out.extend_from_slice(&texel[..3]);
        }
        out
    }

    pub fn destroy(self, context: &gpu::Context) {
        context.destroy_buffer(self.readback);
        context.destroy_texture_view(self.view);
        context.destroy_texture(self.texture);
    }
}

/// `Upscaler::OUTPUT_FORMAT` target used to exercise the live Blade path.
pub struct NeuralTarget {
    texture: gpu::Texture,
    view: gpu::TextureView,
    readback: gpu::Buffer,
    size: gpu::Extent,
}

impl NeuralTarget {
    pub fn new(context: &gpu::Context, size: gpu::Extent) -> Self {
        let format = ommatidia::Upscaler::OUTPUT_FORMAT;
        let texture = context.create_texture(gpu::TextureDesc {
            name: "ommatidia-live-output",
            format,
            size,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::STORAGE | gpu::TextureUsage::COPY,
            sample_count: 1,
            external: None,
        });
        let view = context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "ommatidia-live-output",
                format,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );
        let readback = context.create_buffer(gpu::BufferDesc {
            name: "ommatidia-live-readback",
            size: (size.width * size.height) as u64 * 8,
            memory: gpu::Memory::Shared,
        });
        Self {
            texture,
            view,
            readback,
            size,
        }
    }

    pub fn texture(&self) -> gpu::Texture {
        self.texture
    }

    pub fn view(&self) -> gpu::TextureView {
        self.view
    }

    pub fn read_linear(
        &self,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
    ) -> Vec<f32> {
        {
            let mut transfer = encoder.transfer("ommatidia-live-readback");
            transfer.copy_texture_to_buffer(
                self.texture.into(),
                self.readback.into(),
                self.size.width * 8,
                self.size,
            );
        }
        let sync_point = context.submit(encoder);
        assert!(
            context.wait_for(&sync_point, 30_000).unwrap(),
            "GPU timed out reading Ommatidium output"
        );

        let texels = (self.size.width * self.size.height) as usize;
        let mut mapped = vec![half::f16::ZERO; texels * 4];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.readback.data() as *const half::f16,
                mapped.as_mut_ptr(),
                mapped.len(),
            );
        }
        let mut out = Vec::with_capacity(texels * 3);
        for texel in mapped.chunks_exact(4) {
            out.extend(texel[..3].iter().map(|v| v.to_f32()));
        }
        out
    }

    pub fn destroy(self, context: &gpu::Context) {
        context.destroy_buffer(self.readback);
        context.destroy_texture_view(self.view);
        context.destroy_texture(self.texture);
    }
}

/// Which estimator to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    /// Raw ReSTIR, without Blade's built-in SVGF pass: a comparison input.
    RealTime,
    /// Sparse unbiased paths: the primary network input.
    PathTrace { frames: usize },
    /// Accumulated path tracing: the reference.
    Canonical { frames: usize },
}

impl Pass {
    fn mode(self) -> blade_render::RenderMode {
        match self {
            Self::RealTime => blade_render::RenderMode::RealTime,
            Self::PathTrace { .. } | Self::Canonical { .. } => blade_render::RenderMode::Canonical,
        }
    }

    fn frames(self) -> usize {
        match self {
            Self::RealTime => RESTIR_FRAMES,
            Self::PathTrace { frames } => frames,
            Self::Canonical { frames } => frames,
        }
    }

    fn ray_config(self) -> blade_render::RayConfig {
        blade_render::RayConfig {
            // The canonical mode takes environment samples at every vertex of
            // every path, so it needs far fewer per frame to stay affordable.
            num_environment_samples: match self {
                Self::RealTime => 4,
                Self::PathTrace { .. } | Self::Canonical { .. } => 1,
            },
            // One path per pixel for sparse input; four per accumulated frame
            // makes offline reference generation converge faster.
            num_brdf_samples: match self {
                Self::PathTrace { .. } => 1,
                Self::RealTime | Self::Canonical { .. } => 4,
            },
            // The dummy environment map carries no importance sampling data.
            environment_importance_sampling: false,
            max_bounces: match self {
                Self::Canonical { .. } => REFERENCE_MAX_BOUNCES,
                Self::RealTime | Self::PathTrace { .. } => INPUT_MAX_BOUNCES,
            },
            max_accumulated_samples: 0,
            tap_count: 2,
            tap_radius: 16,
            tap_confidence_near: 8,
            tap_confidence_far: 4,
            t_start: 0.01,
            pairwise_mis: true,
            defensive_mis: 0.1,
        }
    }
}

/// What one render produced.
pub struct Frame {
    /// Interleaved linear RGB radiance.
    pub color: Vec<f32>,
    /// Planar geometry and material channels, when a probe was supplied.
    ///
    /// Laid out as [`crate::gbuffer::PLANES`] describes, ready to sit after the
    /// colour planes in a dataset record.
    pub gbuffer: Option<Vec<f32>>,
}

/// Render one frame of one scene and read back the linear radiance.
///
/// `objects` and `camera` are shared between the two passes, so the pair lines
/// up pixel for pixel modulo the resolution.
#[allow(clippy::too_many_arguments)]
pub fn capture(
    renderer: &mut blade_render::RayTracer,
    target: &Target,
    context: &gpu::Context,
    encoder: &mut gpu::CommandEncoder,
    asset_hub: &blade_render::AssetHub,
    objects: &[blade_render::Object],
    camera: &blade_render::Camera,
    pass: Pass,
    svgf_input: bool,
    probe: Option<&crate::gbuffer::Probe>,
) -> Frame {
    let debug_config = blade_render::DebugConfig::default();
    let mut temp = blade_render::FrameResources::default();

    encoder.start();
    asset_hub.flush(encoder, &mut temp.buffers);

    for frame in 0..pass.frames() {
        renderer.build_scene(encoder, objects, None, asset_hub, context, &mut temp);
        renderer.prepare(
            encoder,
            camera,
            blade_render::FrameConfig {
                frozen: false,
                debug_draw: false,
                reset_variance: frame == 0,
                reset_reservoirs: frame == 0,
                reset_accumulation: frame == 0,
            },
        );
        renderer.render(
            encoder,
            pass.mode(),
            debug_config,
            pass.ray_config(),
            (pass == Pass::RealTime && svgf_input).then_some(blade_render::DenoiserConfig {
                num_passes: 3,
                temporal_weight: 0.1,
            }),
        );

        if (frame + 1).is_multiple_of(FRAMES_PER_SUBMISSION) && frame + 1 < pass.frames() {
            let sync_point = context.submit(encoder);
            assert!(
                context.wait_for(&sync_point, 30_000).unwrap(),
                "GPU timed out during accumulated reference capture"
            );
            for buffer in temp.buffers.drain(..) {
                context.destroy_buffer(buffer);
            }
            for structure in temp.acceleration_structures.drain(..) {
                context.destroy_acceleration_structure(structure);
            }
            encoder.start();
        }
    }

    if pass != Pass::RealTime {
        renderer.fill_gbuffer(encoder, debug_config);
    }

    // Read the G-buffer before the post processing, while it still describes
    // the frame the loop above just finished.
    if let Some(probe) = probe {
        probe.record(encoder, &renderer.view_gbuffer());
    }

    encoder.init_texture(target.texture());
    {
        let mut render_pass = encoder.render(
            "capture",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: target.view(),
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        );
        renderer.post_proc(
            &mut render_pass,
            debug_config,
            blade_render::PostProcConfig {
                // The whole point: capture the composed radiance before
                // anything compresses it into display range. The exposure
                // parameters are unused when tone mapping is off.
                tone_map: false,
                ..Default::default()
            },
            &[],
            &[],
        );
    }

    // This submits and waits, which is also what makes the probe's buffer
    // readable below.
    let color = target.read_linear(context, encoder);
    let gbuffer = probe.map(|probe| probe.read());

    for buffer in temp.buffers {
        context.destroy_buffer(buffer);
    }
    for structure in temp.acceleration_structures {
        context.destroy_acceleration_structure(structure);
    }
    Frame { color, gbuffer }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_paths_are_deeper_than_realtime_inputs() {
        let input = Pass::PathTrace { frames: 1 }.ray_config();
        let reference = Pass::Canonical { frames: 1 }.ray_config();

        assert_eq!(input.num_brdf_samples, 1);
        assert_eq!(input.max_bounces, INPUT_MAX_BOUNCES);
        assert_eq!(reference.num_brdf_samples, 4);
        assert_eq!(reference.max_bounces, REFERENCE_MAX_BOUNCES);
        assert!(reference.max_bounces > input.max_bounces);
    }
}
