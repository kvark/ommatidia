//! Readback of Blade's canonical primary-lobe accumulators.

use blade_graphics as gpu;
use ommatidia::dataset::{Plane, PlaneSet};

pub const PLANES: [Plane; 3] = [
    Plane::DiffuseIllumination,
    Plane::SpecularRadiance,
    Plane::EmissiveRadiance,
];

pub fn plane_set() -> PlaneSet {
    PLANES.into_iter().collect()
}

#[derive(blade_macros::ShaderData)]
struct ProbeData {
    params: Params,
    t_diffuse: gpu::TextureView,
    t_specular: gpu::TextureView,
    t_emissive: gpu::TextureView,
    planes: gpu::BufferPiece,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Params {
    width: u32,
    height: u32,
    _pad: [u32; 2],
}

pub struct Probe {
    pipeline: gpu::ComputePipeline,
    buffer: gpu::Buffer,
    size: gpu::Extent,
    len: usize,
}

impl Probe {
    pub fn new(context: &gpu::Context, size: gpu::Extent) -> Self {
        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("radiance.wgsl"),
            naga_module: None,
        });
        let layout = <ProbeData as gpu::ShaderData>::layout();
        let pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "ommatidia-radiance-probe",
            data_layouts: &[&layout],
            compute: shader.at("probe"),
        });
        let len = plane_set().channels() * (size.width * size.height) as usize;
        let buffer = context.create_buffer(gpu::BufferDesc {
            name: "canonical-radiance-lobes",
            size: len as u64 * 4,
            memory: gpu::Memory::Shared,
        });
        Self {
            pipeline,
            buffer,
            size,
            len,
        }
    }

    pub fn record(
        &self,
        encoder: &mut gpu::CommandEncoder,
        views: &blade_render::AccumulatedRadianceViews,
    ) {
        let mut pass = encoder.compute("ommatidia-radiance-probe");
        let mut commands = pass.with(&self.pipeline);
        commands.bind(
            0,
            &ProbeData {
                params: Params {
                    width: self.size.width,
                    height: self.size.height,
                    _pad: [0; 2],
                },
                t_diffuse: views.diffuse,
                t_specular: views.specular,
                t_emissive: views.emissive,
                planes: self.buffer.into(),
            },
        );
        commands.dispatch([self.size.width.div_ceil(8), self.size.height.div_ceil(8), 1]);
    }

    pub fn read(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.len];
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
    fn planes_have_dataset_order() {
        assert_eq!(plane_set().iter().collect::<Vec<_>>(), PLANES);
        assert_eq!(plane_set().channels(), 9);
    }

    #[test]
    fn shader_parses() {
        let source = include_str!("radiance.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("radiance.wgsl: {}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("radiance.wgsl failed validation: {e:?}"));
    }
}
