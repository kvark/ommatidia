//! The runtime's shaders have to agree with the CPU batching exactly.
//!
//! This is the one contract that fails silently: the network trains against
//! what `ommatidia::batch` produces, so if `pack.wgsl` or `unpack.wgsl` drifts
//! from it, training keeps looking perfect and the renderer keeps producing
//! garbage. Nothing else in the system would notice.
//!
//! ```sh
//! cargo test -p ommatidia --test gpu_runtime -- --ignored --nocapture
//! ```

use std::sync::Arc;

use blade_graphics as gpu;
use half::f16;
use ommatidia::batch::{self, Crop};
use ommatidia::dataset::{Layout, Plane, PlaneSet, Sample};
use ommatidia::model::{ModelConfig, Objective};
use ommatidia::rng::Rng;
use ommatidia::runtime::{FrameInputs, Upscaler};

const TILE: u32 = 16;
const SCALE: u32 = 2;

fn context() -> Option<Arc<gpu::Context>> {
    let device_id = std::env::var("OMMATIDIA_TEST_DEVICE_ID")
        .ok()
        .map(|value| ommatidia::gpu::parse_device_id(&value))
        .transpose()
        .expect("invalid OMMATIDIA_TEST_DEVICE_ID");
    let desc = gpu::ContextDesc {
        validation: true,
        device_id,
        ..Default::default()
    };
    match unsafe { gpu::Context::init(desc) } {
        Ok(context) => Some(Arc::new(context)),
        Err(e) => {
            println!("skipping: no GPU context ({e:?})");
            None
        }
    }
}

fn config() -> ModelConfig {
    ModelConfig {
        scale: SCALE,
        tile: TILE,
        batch: 1,
        cond_planes: PlaneSet::new().with(Plane::Color),
        base_channels: 16,
        level_multipliers: vec![1, 2],
        blocks_per_level: 1,
        num_groups: 8,
        time_input_dim: 16,
        time_embed_dim: 32,
        // Deliberately not 1, so a runtime that ignored the gain would show up.
        residual_gain: 7.5,
        gn_eps: 1e-5,
        objective: Objective::Direct,
    }
}

/// An untrained checkpoint is enough: the question is whether the two paths
/// agree, not whether the weights are any good.
fn write_checkpoint(config: &ModelConfig, stem: &std::path::Path, context: Arc<gpu::Context>) {
    let model = ommatidia::model::build(config, false).expect("build");
    let mut session = ommatidia::gpu::inference_session(&model.graph, context);
    model.initialize(&mut session, 3);
    ommatidia::checkpoint::save(&mut session, config, stem).expect("save");
}

/// A colour texture holding `values` as interleaved RGB.
fn color_texture(
    context: &gpu::Context,
    encoder: &mut gpu::CommandEncoder,
    values: &[f32],
    width: u32,
    height: u32,
) -> (gpu::Texture, gpu::TextureView, gpu::Buffer) {
    let format = gpu::TextureFormat::Rgba32Float;
    let size = gpu::Extent {
        width,
        height,
        depth: 1,
    };
    let texture = context.create_texture(gpu::TextureDesc {
        name: "test-color",
        format,
        size,
        dimension: gpu::TextureDimension::D2,
        array_layer_count: 1,
        mip_level_count: 1,
        usage: gpu::TextureUsage::RESOURCE | gpu::TextureUsage::COPY,
        sample_count: 1,
        external: None,
    });
    let view = context.create_texture_view(
        texture,
        gpu::TextureViewDesc {
            name: "test-color",
            format,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        },
    );

    let staging = context.create_buffer(gpu::BufferDesc {
        name: "test-color-staging",
        size: (width * height) as u64 * 16,
        memory: gpu::Memory::Upload,
    });
    unsafe {
        let ptr = staging.data() as *mut f32;
        for i in 0..(width * height) as usize {
            for c in 0..3 {
                *ptr.add(i * 4 + c) = values[i * 3 + c];
            }
            *ptr.add(i * 4 + 3) = 1.0;
        }
    }

    encoder.init_texture(texture);
    let mut transfer = encoder.transfer("upload");
    transfer.copy_buffer_to_texture(staging.into(), width * 16, texture.into(), size);
    (texture, view, staging)
}

/// The GPU pack has to reproduce `batch::write_conditioning` value for value.
#[test]
#[ignore = "requires a GPU"]
fn pack_matches_the_cpu_path() {
    let Some(context) = context() else { return };
    let config = config();
    let dir = std::env::temp_dir().join("ommatidia-gpu-runtime-pack");
    std::fs::create_dir_all(&dir).unwrap();
    let stem = dir.join("model");
    write_checkpoint(&config, &stem, Arc::clone(&context));

    let mut upscaler =
        Upscaler::from_checkpoint(Arc::clone(&context), &stem, 4, 100).expect("upscaler");

    // High dynamic range on purpose: the compression is the part most likely
    // to differ between the two implementations.
    let mut rng = Rng::new(11);
    let texels = (TILE * TILE) as usize;
    let colors: Vec<f32> = (0..texels * 3).map(|_| rng.uniform() * 40.0).collect();

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "pack-test",
        buffer_count: 1,
        manual_barriers: false,
    });
    encoder.start();
    let (texture, view, staging) = color_texture(&context, &mut encoder, &colors, TILE, TILE);
    upscaler.pack(&mut encoder, &FrameInputs::color_only(view, view));
    let sync_point = context.submit(&mut encoder);
    assert!(context.wait_for(&sync_point, 30_000).unwrap());

    // What the CPU path would have produced from the same pixels.
    let layout = Layout {
        scale: SCALE,
        lr_width: TILE,
        lr_height: TILE,
        lr_source: ommatidia::dataset::InputSource::RawRestir,
        lr_planes: PlaneSet::new().with(Plane::Color),
        hr_planes: PlaneSet::new().with(Plane::Color),
    };
    let mut planar = vec![f16::ZERO; layout.lr_len()];
    for i in 0..texels {
        for c in 0..3 {
            planar[c * texels + i] = f16::from_f32(colors[i * 3 + c]);
        }
    }
    let sample = Sample {
        lr: planar,
        hr: vec![f16::ZERO; layout.hr_len()],
    };
    let mut expected = vec![0.0; config.cond_len()];
    batch::write_conditioning(
        &sample,
        &layout,
        config.cond_planes,
        Crop {
            x: 0,
            y: 0,
            tile: TILE,
        },
        0,
        &mut expected,
    );

    let actual = read_input(&mut upscaler, "cond", expected.len());
    let mut worst = 0.0f32;
    for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        let difference = (a - b).abs();
        assert!(
            difference < 1e-3,
            "element {i}: gpu {a} vs cpu {b} (the shader and the batcher disagree)"
        );
        worst = worst.max(difference);
    }
    println!("pack: worst difference from the CPU path = {worst:e}");

    upscaler.destroy();
    drop(upscaler);
    context.destroy_buffer(staging);
    context.destroy_texture_view(view);
    context.destroy_texture(texture);
    context.destroy_command_encoder(&mut encoder);
}

/// Read one of the session's input buffers back to the host.
///
/// They are `Memory::Shared`, so the pointer is directly readable.
fn read_input(upscaler: &mut Upscaler, name: &str, len: usize) -> Vec<f32> {
    let (ptr, size) = upscaler
        .session()
        .input_host_ptr(name)
        .unwrap_or_else(|| panic!("no input named {name}"));
    assert!(size >= len * 4, "{name} is smaller than expected");
    let mut out = vec![0.0f32; len];
    unsafe {
        std::ptr::copy_nonoverlapping(ptr as *const f32, out.as_mut_ptr(), len);
    }
    out
}

/// A full frame through the runtime has to match `batch::assemble`.
#[test]
#[ignore = "requires a GPU"]
fn upscale_matches_the_cpu_path() {
    let Some(context) = context() else { return };
    let config = config();
    let dir = std::env::temp_dir().join("ommatidia-gpu-runtime-upscale");
    std::fs::create_dir_all(&dir).unwrap();
    let stem = dir.join("model");
    write_checkpoint(&config, &stem, Arc::clone(&context));

    let mut upscaler =
        Upscaler::from_checkpoint(Arc::clone(&context), &stem, 4, 100).expect("upscaler");
    let (out_width, out_height) = upscaler.output_extent();

    let mut rng = Rng::new(5);
    let texels = (TILE * TILE) as usize;
    let colors: Vec<f32> = (0..texels * 3).map(|_| rng.uniform() * 12.0).collect();

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "upscale-test",
        buffer_count: 2,
        manual_barriers: false,
    });
    encoder.start();
    let (texture, view, staging) = color_texture(&context, &mut encoder, &colors, TILE, TILE);

    let format = Upscaler::OUTPUT_FORMAT;
    let out_size = gpu::Extent {
        width: out_width,
        height: out_height,
        depth: 1,
    };
    let output = context.create_texture(gpu::TextureDesc {
        name: "test-output",
        format,
        size: out_size,
        dimension: gpu::TextureDimension::D2,
        array_layer_count: 1,
        mip_level_count: 1,
        usage: gpu::TextureUsage::STORAGE | gpu::TextureUsage::COPY,
        sample_count: 1,
        external: None,
    });
    let output_view = context.create_texture_view(
        output,
        gpu::TextureViewDesc {
            name: "test-output",
            format,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        },
    );
    encoder.init_texture(output);

    upscaler.upscale(
        &mut encoder,
        &FrameInputs::color_only(view, view),
        output_view,
    );

    // Read the result back.
    let readback = context.create_buffer(gpu::BufferDesc {
        name: "test-readback",
        size: (out_width * out_height) as u64 * 8,
        memory: gpu::Memory::Shared,
    });
    {
        let mut transfer = encoder.transfer("readback");
        transfer.copy_texture_to_buffer(output.into(), readback.into(), out_width * 8, out_size);
    }
    let sync_point = context.submit(&mut encoder);
    assert!(context.wait_for(&sync_point, 30_000).unwrap());

    let count = (out_width * out_height * 4) as usize;
    let mut halves = vec![f16::ZERO; count];
    unsafe {
        std::ptr::copy_nonoverlapping(readback.data() as *const f16, halves.as_mut_ptr(), count);
    }

    // The CPU reference: the same residual the network produced, assembled the
    // same way.
    let residual = upscaler.read_residual();
    let expected = batch::assemble(
        &colors,
        &residual,
        TILE as usize,
        TILE as usize,
        SCALE as usize,
        config.residual_gain,
    );

    let mut worst = 0.0f32;
    for texel in 0..(out_width * out_height) as usize {
        for c in 0..3 {
            let gpu_value = halves[texel * 4 + c].to_f32();
            let cpu_value = expected[texel * 3 + c];
            // f16 storage in the output texture sets the tolerance.
            let difference = (gpu_value - cpu_value).abs() / cpu_value.abs().max(1.0);
            assert!(
                difference < 2e-2,
                "texel {texel} channel {c}: gpu {gpu_value} vs cpu {cpu_value}"
            );
            worst = worst.max(difference);
        }
    }
    println!("upscale: worst relative difference from the CPU path = {worst:e}");

    // With a zero-initialised head the residual is zero, so the result has to
    // be exactly nearest-neighbour upsampling — a property that pins the
    // sub-pixel indexing independently of the network.
    for y in 0..TILE as usize {
        for x in 0..TILE as usize {
            for c in 0..3 {
                let source = colors[(y * TILE as usize + x) * 3 + c];
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let oy = y * SCALE as usize + dy;
                    let ox = x * SCALE as usize + dx;
                    let got = halves[(oy * out_width as usize + ox) * 4 + c].to_f32();
                    let difference = (got - source).abs() / source.abs().max(1.0);
                    assert!(
                        difference < 2e-2,
                        "sub-pixel ({dx},{dy}) of ({x},{y}) channel {c}: \
                         {got} should replicate {source}"
                    );
                }
            }
        }
    }
    println!("upscale: a zero residual reproduces nearest-neighbour exactly");

    upscaler.destroy();
    drop(upscaler);
    context.destroy_buffer(readback);
    context.destroy_buffer(staging);
    context.destroy_texture_view(output_view);
    context.destroy_texture(output);
    context.destroy_texture_view(view);
    context.destroy_texture(texture);
    context.destroy_command_encoder(&mut encoder);
}
