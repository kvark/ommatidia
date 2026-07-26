//! Capturing blade's G-buffer alongside the colour.
//!
//! This is the structural advantage a renderer has over photographic
//! super-resolution: it is not handed an image, it is asked to produce one, and
//! it knows the exact geometry and material behind every pixel. Silhouettes do
//! not have to be guessed from colour gradients, texture detail is separable
//! from lighting through the albedo, and the width of a specular highlight is
//! told rather than inferred. At input resolution it is all free — the renderer
//! filled these targets on its way to shading.

use blade_graphics as gpu;
use ommatidia::dataset::{Plane, PlaneSet};

/// Planes this probe produces, in the order it writes them.
///
/// Colour is absent because the post processing already produced it, and
/// motion because it only means something once samples become trajectories.
pub const PLANES: [Plane; 5] = [
    Plane::Depth,
    Plane::Normal,
    Plane::DiffuseAlbedo,
    Plane::SpecularF0,
    Plane::Roughness,
];

/// The plane set a dataset gains from this probe.
pub fn plane_set() -> PlaneSet {
    PLANES.into_iter().collect()
}

/// Channels the probe writes per pixel.
pub fn channels() -> usize {
    PLANES.iter().map(|p| p.channels()).sum()
}

#[derive(blade_macros::ShaderData)]
struct ProbeData {
    params: Params,
    t_depth: gpu::TextureView,
    t_basis: gpu::TextureView,
    t_diffuse_albedo: gpu::TextureView,
    t_specular_f0: gpu::TextureView,
    planes: gpu::BufferPiece,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Params {
    width: u32,
    height: u32,
}

/// Reads the renderer's G-buffer into a planar float buffer.
pub struct Probe {
    pipeline: gpu::ComputePipeline,
    buffer: gpu::Buffer,
    size: gpu::Extent,
    len: usize,
}

impl Probe {
    pub fn new(context: &gpu::Context, size: gpu::Extent) -> Self {
        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("gbuffer.wgsl"),
            naga_module: None,
        });
        let layout = <ProbeData as gpu::ShaderData>::layout();
        let pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "ommatidia-gbuffer-probe",
            data_layouts: &[&layout],
            compute: shader.at("probe"),
        });

        let len = channels() * (size.width * size.height) as usize;
        let buffer = context.create_buffer(gpu::BufferDesc {
            name: "gbuffer-planes",
            size: len as u64 * 4,
            // Host-visible, so the readback is a memcpy with no transfer pass.
            memory: gpu::Memory::Shared,
        });

        Self {
            pipeline,
            buffer,
            size,
            len,
        }
    }

    /// Record the probe against the views of the frame just rendered.
    pub fn record(&self, encoder: &mut gpu::CommandEncoder, views: &blade_render::GBufferViews) {
        let mut pass = encoder.compute("ommatidia-gbuffer-probe");
        let mut commands = pass.with(&self.pipeline);
        commands.bind(
            0,
            &ProbeData {
                params: Params {
                    width: self.size.width,
                    height: self.size.height,
                },
                t_depth: views.depth,
                t_basis: views.basis,
                t_diffuse_albedo: views.diffuse_albedo,
                t_specular_f0: views.specular_f0,
                planes: self.buffer.into(),
            },
        );
        commands.dispatch([self.size.width.div_ceil(8), self.size.height.div_ceil(8), 1]);
    }

    /// Copy out what the last recorded probe wrote.
    ///
    /// The caller has to have submitted and waited; the generator does that as
    /// part of reading the colour back.
    pub fn read(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.len];
        // Safety: the buffer is `Memory::Shared`, so it is mapped for the
        // lifetime of the allocation, and holds exactly `self.len` floats.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buffer.data() as *const f32,
                out.as_mut_ptr(),
                self.len,
            );
        }
        out
    }

    pub fn destroy(mut self, context: &gpu::Context) {
        context.destroy_buffer(self.buffer);
        context.destroy_compute_pipeline(&mut self.pipeline);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probe_matches_the_dataset_plane_order() {
        // The generator concatenates colour and then these, so the shader's
        // write order has to be the order `PlaneSet::iter` walks.
        let set = plane_set().with(Plane::Color);
        let walked: Vec<Plane> = set.iter().collect();
        assert_eq!(walked[0], Plane::Color);
        assert_eq!(&walked[1..], &PLANES);
        assert_eq!(channels(), 11);
        assert_eq!(set.channels(), 14);
    }

    #[test]
    fn shader_parses() {
        let source = include_str!("gbuffer.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("gbuffer.wgsl: {}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            // Blade fills in groups and bindings at pipeline creation.
            naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("gbuffer.wgsl failed validation: {e:?}"));
    }
}
