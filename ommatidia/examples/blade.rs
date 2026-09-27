//! Headless Blade integration: caller-owned buffers, encoder and submission.
//!
//! The small synthetic upload stands in for a renderer's observation passes;
//! the final copy stands in for a compositor consuming reconstructed RGBA.
//! No checkpoint is loaded: this demonstrates scheduling, not trained quality.
use blade_graphics as gpu;
use ommatidia::transport::{Config, Ray, Surface, native::Native};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device_id = std::env::args()
        .nth(1)
        .map(|value| ommatidia::gpu::parse_device_id(&value))
        .transpose()?;
    // In an application, use the renderer's existing Arc<Context>.
    let context = Arc::new(unsafe {
        gpu::Context::init(gpu::ContextDesc {
            validation: true,
            device_id,
            ..Default::default()
        })?
    });
    let config = Config::default();
    let low = [8, 8];
    let pixels = (low[0] * low[1] * config.scale.pow(2)) as usize;
    let rays = vec![
        Ray {
            diffuse: [0.6, 0.5, 0.4, 0.0],
            specular: [0.1, 0.1, 0.1, 0.0],
            normal_depth: [0.0, 0.0, 1.0, 1.0],
        };
        (low[0] * low[1]) as usize
    ];
    let surfaces = vec![
        Surface {
            normal_depth: [0.0, 0.0, 1.0, 1.0],
            albedo_roughness: [0.8, 0.6, 0.4, 0.5],
            specular_f0: [0.04, 0.04, 0.04, 0.0],
            ..Surface::default()
        };
        pixels
    ];
    let rays_size = std::mem::size_of_val(rays.as_slice()) as u64;
    let surfaces_size = std::mem::size_of_val(surfaces.as_slice()) as u64;
    let rgba_size = (pixels * 16) as u64;
    let buffer = |name, size, memory| context.create_buffer(gpu::BufferDesc { name, size, memory });
    let upload = buffer(
        "example-upload",
        rays_size + surfaces_size,
        gpu::Memory::Shared,
    );
    let ray_buffer = buffer("renderer-rays", rays_size, gpu::Memory::Device);
    let surface_buffer = buffer("renderer-surfaces", surfaces_size, gpu::Memory::Device);
    let output = buffer("reconstructed-rgba", rgba_size, gpu::Memory::Device);
    let readback = buffer("example-readback", rgba_size, gpu::Memory::Shared);
    unsafe {
        std::ptr::copy_nonoverlapping(rays.as_ptr(), upload.data().cast::<Ray>(), rays.len());
        std::ptr::copy_nonoverlapping(
            surfaces.as_ptr(),
            upload.data().add(rays_size as usize).cast::<Surface>(),
            surfaces.len(),
        );
    }
    let mut reconstruction = Native::new(Arc::clone(&context), config, low)?;
    // Load the application's v4 weights before recording; reset() on camera cuts.
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "graphics-and-reconstruction",
        buffer_count: 2,
        manual_barriers: false, // Required by Session::record.
    });
    encoder.start();
    {
        let mut pass = encoder.transfer("renderer-observations");
        pass.copy_buffer_to_buffer(upload.at(0), ray_buffer.at(0), rays_size);
        pass.copy_buffer_to_buffer(upload.at(rays_size), surface_buffer.at(0), surfaces_size);
    }
    reconstruction.record_prepare(
        &mut encoder,
        ray_buffer.at(0),
        surface_buffer.at(0),
        [0.0; 2],
        1.0,
    );
    reconstruction.session.record(&mut encoder)?;
    reconstruction.record_resolve(&mut encoder, surface_buffer.at(0), output.at(0));
    encoder
        .transfer("consume-reconstruction")
        .copy_buffer_to_buffer(output.at(0), readback.at(0), rgba_size);

    // One caller-owned submission: there are no waits between the passes above.
    let sync = context.submit(&mut encoder);
    reconstruction.session.track_submission(sync);
    // Only the example's CPU readback needs a wait. A renderer can continue to
    // record GPU work and report its latest submission. Its normal frame fences
    // must still protect command-buffer reuse and any reused upload buffers.
    reconstruction.session.wait();
    let rgba = unsafe { std::slice::from_raw_parts(readback.data().cast::<f32>(), pixels * 4) };
    assert!(rgba.iter().all(|v| v.is_finite()));
    assert!(rgba.chunks_exact(4).all(|pixel| pixel[3] == 1.0));
    println!(
        "One graphics encoder/submission: {pixels} finite RGBA pixels on {}",
        context.device_information().device_name
    );

    drop(reconstruction);
    context.destroy_command_encoder(&mut encoder);
    for buffer in [upload, ray_buffer, surface_buffer, output, readback] {
        context.destroy_buffer(buffer);
    }
    Ok(())
}
