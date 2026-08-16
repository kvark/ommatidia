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
use std::{fs::File, io::BufReader, path::Path};

use blade_graphics as gpu;
use half::f16;
use ommatidia::batch::{self, Crop};
use ommatidia::dataset::{Layout, Plane, PlaneSet, Sample};
use ommatidia::model::{ModelConfig, Objective, ReconstructionBase};
use ommatidia::rng::Rng;
use ommatidia::runtime::{FrameInputs, Upscaler};

const TILE: u32 = 16;
const SCALE: u32 = 2;
const SNAPSHOT_SSIM_THRESHOLD: f64 = 0.995;

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
            assert!(
                std::env::var_os("OMMATIDIA_REQUIRE_GPU").is_none(),
                "required GPU context could not be created: {e:?}"
            );
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
        prediction: ommatidia::model::Prediction::SubpixelResidual,
        reconstruction_base: ReconstructionBase::Bilinear,
        guide: ommatidia::model::GuideConfig::TUNED,
        kernel_radius: 2,
        demodulate: false,
        demodulation_offset: 0.25,
        head_kernel: 3,
        temporal_weight: 0.0,
        temporal: None,
    }
}

fn guided_config() -> ModelConfig {
    ModelConfig {
        cond_planes: PlaneSet::new()
            .with(Plane::Color)
            .with(Plane::Depth)
            .with(Plane::Normal)
            .with(Plane::DiffuseAlbedo)
            .with(Plane::SpecularF0)
            .with(Plane::Roughness),
        reconstruction_base: ReconstructionBase::GuidedBilinear,
        ..config()
    }
}

fn hr_guided_config() -> ModelConfig {
    ModelConfig {
        reconstruction_base: ReconstructionBase::HighResolutionGuided,
        ..guided_config()
    }
}

fn kernel_config() -> ModelConfig {
    ModelConfig {
        prediction: ommatidia::model::Prediction::SubpixelKernel,
        reconstruction_base: ReconstructionBase::Sample,
        kernel_radius: 2,
        ..guided_config()
    }
}

fn demodulating_kernel_config() -> ModelConfig {
    ModelConfig {
        demodulate: true,
        ..kernel_config()
    }
}

/// An untrained checkpoint is enough: the question is whether the two paths
/// agree, not whether the weights are any good.
fn write_checkpoint(config: &ModelConfig, stem: &std::path::Path, context: Arc<gpu::Context>) {
    let model = ommatidia::model::build(config, false).expect("build");
    let mut session = ommatidia::gpu::inference_session(&model.graph, context);
    model.initialize(&mut session, 3);
    // Exercise the network itself, not only the texture plumbing. Production
    // models train this zero-initialised head; the reference test gives it a
    // small deterministic projection so changes anywhere in the backbone are
    // observable in the output image.
    let head = model
        .params
        .iter()
        .find(|parameter| parameter.name == "head.conv.weight")
        .expect("head parameter");
    let weights: Vec<f32> = (0..head.len)
        .map(|index| ((index as f32 * 0.173).sin()) * 0.002)
        .collect();
    session.set_parameter(&head.name, &weights);
    ommatidia::checkpoint::save(&mut session, config, stem).expect("save");
}

fn srgb8(value: f32) -> u8 {
    let mapped = ommatidia::transform::compress(value.max(0.0));
    let encoded = if mapped <= 0.0031308 {
        12.92 * mapped
    } else {
        1.055 * mapped.powf(1.0 / 2.4) - 0.055
    };
    (encoded.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn save_png(path: &Path, rgba: &[u8], width: u32, height: u32) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut encoder = png::Encoder::new(File::create(path).unwrap(), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(rgba)
        .unwrap();
}

fn load_png(path: &Path) -> (Vec<u8>, u32, u32) {
    let mut reader = png::Decoder::new(BufReader::new(File::open(path).unwrap()))
        .read_info()
        .unwrap();
    let mut bytes = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut bytes).unwrap();
    bytes.truncate(info.buffer_size());
    (bytes, info.width, info.height)
}

/// Global luminance SSIM. The image is deliberately small and strongly
/// patterned; a global window is less forgiving than averaging many blocks.
fn ssim(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let luminance = |pixel: &[u8]| {
        0.2126 * pixel[0] as f64 + 0.7152 * pixel[1] as f64 + 0.0722 * pixel[2] as f64
    };
    let count = (a.len() / 4) as f64;
    let (mut sum_a, mut sum_b) = (0.0, 0.0);
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        sum_a += luminance(pa);
        sum_b += luminance(pb);
    }
    let (mean_a, mean_b) = (sum_a / count, sum_b / count);
    let (mut var_a, mut var_b, mut covariance) = (0.0, 0.0, 0.0);
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let da = luminance(pa) - mean_a;
        let db = luminance(pb) - mean_b;
        var_a += da * da;
        var_b += db * db;
        covariance += da * db;
    }
    var_a /= count;
    var_b /= count;
    covariance /= count;
    const C1: f64 = 6.5025;
    const C2: f64 = 58.5225;
    ((2.0 * mean_a * mean_b + C1) * (2.0 * covariance + C2))
        / ((mean_a * mean_a + mean_b * mean_b + C1) * (var_a + var_b + C2))
}

fn check_snapshot(rgba: &[u8], width: u32, height: u32) {
    let reference =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/reference/upscale-runtime.png");
    if std::env::var_os("OMMATIDIA_UPDATE_SNAPSHOTS").is_some() {
        save_png(&reference, rgba, width, height);
        println!("updated {}", reference.display());
        return;
    }
    let (expected, expected_width, expected_height) = load_png(&reference);
    assert_eq!((expected_width, expected_height), (width, height));
    let score = ssim(rgba, &expected);
    println!("upscale-runtime: SSIM = {score:.6}");
    if score < SNAPSHOT_SSIM_THRESHOLD {
        let actual = reference.with_file_name("upscale-runtime_actual.png");
        save_png(&actual, rgba, width, height);
        panic!(
            "GPU image SSIM {score:.6} is below {SNAPSHOT_SSIM_THRESHOLD}; wrote {}",
            actual.display()
        );
    }
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
    let config = hr_guided_config();
    let dir = std::env::temp_dir().join("ommatidia-gpu-runtime-upscale");
    std::fs::create_dir_all(&dir).unwrap();
    let stem = dir.join("model");
    write_checkpoint(&config, &stem, Arc::clone(&context));

    let mut upscaler =
        Upscaler::from_checkpoint(Arc::clone(&context), &stem, 4, 100).expect("upscaler");
    let (out_width, out_height) = upscaler.output_extent();

    let mut rng = Rng::new(5);
    let texels = (TILE * TILE) as usize;
    let colors: Vec<f32> = (0..texels * 3)
        .map(|_| f16::from_f32(rng.uniform() * 12.0).to_f32())
        .collect();
    let mut depths = vec![0.0f32; texels * 3];
    let mut normals = vec![0.0f32; texels * 3];
    let mut albedos = vec![0.0f32; texels * 3];
    let specular = vec![0.25f32; texels * 3];
    for y in 0..TILE as usize {
        for x in 0..TILE as usize {
            let index = y * TILE as usize + x;
            depths[index * 3] = 2.0 + y as f32 * 0.25;
            let (normal, albedo) = if x < TILE as usize / 2 {
                ([0.0, 0.0, 1.0], [0.25, 0.5, 0.75])
            } else {
                ([0.6, 0.0, 0.8], [0.75, 0.25, 0.5])
            };
            normals[index * 3..index * 3 + 3].copy_from_slice(&normal);
            albedos[index * 3..index * 3 + 3].copy_from_slice(&albedo);
        }
    }
    let hr_texels = (TILE * SCALE * TILE * SCALE) as usize;
    let mut hr_depths = vec![0.0f32; hr_texels * 3];
    let mut hr_normals = vec![0.0f32; hr_texels * 3];
    let mut hr_albedos = vec![0.0f32; hr_texels * 3];
    for y in 0..(TILE * SCALE) as usize {
        for x in 0..(TILE * SCALE) as usize {
            let source_x = x / SCALE as usize;
            let source_y = y / SCALE as usize;
            let source = source_y * TILE as usize + source_x;
            let destination = y * (TILE * SCALE) as usize + x;
            hr_depths[destination * 3] = depths[source * 3];
            hr_normals[destination * 3..destination * 3 + 3]
                .copy_from_slice(&normals[source * 3..source * 3 + 3]);
            hr_albedos[destination * 3..destination * 3 + 3]
                .copy_from_slice(&albedos[source * 3..source * 3 + 3]);
        }
    }

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "upscale-test",
        buffer_count: 2,
        manual_barriers: false,
    });
    encoder.start();
    let (texture, view, staging) = color_texture(&context, &mut encoder, &colors, TILE, TILE);
    let (depth_texture, depth_view, depth_staging) =
        color_texture(&context, &mut encoder, &depths, TILE, TILE);
    let (normal_texture, normal_view, normal_staging) =
        color_texture(&context, &mut encoder, &normals, TILE, TILE);
    let (albedo_texture, albedo_view, albedo_staging) =
        color_texture(&context, &mut encoder, &albedos, TILE, TILE);
    let (specular_texture, specular_view, specular_staging) =
        color_texture(&context, &mut encoder, &specular, TILE, TILE);
    let (hr_depth_texture, hr_depth_view, hr_depth_staging) = color_texture(
        &context,
        &mut encoder,
        &hr_depths,
        TILE * SCALE,
        TILE * SCALE,
    );
    let (hr_normal_texture, hr_normal_view, hr_normal_staging) = color_texture(
        &context,
        &mut encoder,
        &hr_normals,
        TILE * SCALE,
        TILE * SCALE,
    );
    let (hr_albedo_texture, hr_albedo_view, hr_albedo_staging) = color_texture(
        &context,
        &mut encoder,
        &hr_albedos,
        TILE * SCALE,
        TILE * SCALE,
    );

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
        &FrameInputs::from_textures(view, depth_view, normal_view, albedo_view, specular_view)
            .with_high_resolution_gbuffer(hr_depth_view, hr_normal_view, hr_albedo_view),
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
    let layout = Layout {
        scale: SCALE,
        lr_width: TILE,
        lr_height: TILE,
        lr_source: ommatidia::dataset::InputSource::PathTrace,
        lr_planes: config.cond_planes,
        hr_planes: PlaneSet::new()
            .with(Plane::Color)
            .with(Plane::Depth)
            .with(Plane::Normal)
            .with(Plane::DiffuseAlbedo),
    };
    let mut planar = vec![f16::ZERO; layout.lr_len()];
    for (plane, values) in [
        (Plane::Color, colors.as_slice()),
        (Plane::Normal, normals.as_slice()),
        (Plane::DiffuseAlbedo, albedos.as_slice()),
        (Plane::SpecularF0, specular.as_slice()),
    ] {
        let base = layout.lr_planes.channel_offset(plane).unwrap();
        for component in 0..3 {
            for index in 0..texels {
                planar[(base + component) * texels + index] =
                    f16::from_f32(values[index * 3 + component]);
            }
        }
    }
    let depth_base = layout.lr_planes.channel_offset(Plane::Depth).unwrap();
    let roughness_base = layout.lr_planes.channel_offset(Plane::Roughness).unwrap();
    for index in 0..texels {
        planar[depth_base * texels + index] = f16::from_f32(depths[index * 3]);
        planar[roughness_base * texels + index] = f16::ONE;
    }
    let mut hr = vec![f16::ZERO; layout.hr_len()];
    for (plane, values) in [
        (Plane::Normal, hr_normals.as_slice()),
        (Plane::DiffuseAlbedo, hr_albedos.as_slice()),
    ] {
        let base = layout.hr_planes.channel_offset(plane).unwrap();
        for component in 0..3 {
            for index in 0..hr_texels {
                hr[(base + component) * hr_texels + index] =
                    f16::from_f32(values[index * 3 + component]);
            }
        }
    }
    let hr_depth_base = layout.hr_planes.channel_offset(Plane::Depth).unwrap();
    for index in 0..hr_texels {
        hr[hr_depth_base * hr_texels + index] = f16::from_f32(hr_depths[index * 3]);
    }
    let sample = Sample { lr: planar, hr };
    let guided = batch::high_resolution_guided_base(
        &sample,
        &layout,
        Crop {
            x: 0,
            y: 0,
            tile: TILE,
        },
        config.guide,
    );
    let expected = batch::assemble(
        &colors,
        Some(&guided),
        &residual,
        [TILE as usize; 2],
        &config,
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

    // Pin the complete GPU image as well as comparing it numerically to the
    // CPU assembly path. This catches coherent visual changes that can hide
    // inside a per-element tolerance and leaves an artifact on CI failure.
    let mut rgba = Vec::with_capacity((out_width * out_height * 4) as usize);
    for texel in halves.chunks_exact(4) {
        rgba.extend(texel[..3].iter().map(|value| srgb8(value.to_f32())));
        rgba.push(255);
    }
    check_snapshot(&rgba, out_width, out_height);

    upscaler.destroy();
    drop(upscaler);
    context.destroy_buffer(readback);
    context.destroy_buffer(staging);
    context.destroy_buffer(depth_staging);
    context.destroy_buffer(normal_staging);
    context.destroy_buffer(albedo_staging);
    context.destroy_buffer(specular_staging);
    context.destroy_buffer(hr_depth_staging);
    context.destroy_buffer(hr_normal_staging);
    context.destroy_buffer(hr_albedo_staging);
    context.destroy_texture_view(output_view);
    context.destroy_texture(output);
    context.destroy_texture_view(view);
    context.destroy_texture_view(depth_view);
    context.destroy_texture_view(normal_view);
    context.destroy_texture_view(albedo_view);
    context.destroy_texture_view(specular_view);
    context.destroy_texture_view(hr_depth_view);
    context.destroy_texture_view(hr_normal_view);
    context.destroy_texture_view(hr_albedo_view);
    context.destroy_texture(texture);
    context.destroy_texture(depth_texture);
    context.destroy_texture(normal_texture);
    context.destroy_texture(albedo_texture);
    context.destroy_texture(specular_texture);
    context.destroy_texture(hr_depth_texture);
    context.destroy_texture(hr_normal_texture);
    context.destroy_texture(hr_albedo_texture);
    context.destroy_command_encoder(&mut encoder);
}

/// The kernel path has no deterministic base, so nothing in the output is
/// recognisable except through the gather itself. If `unpack.wgsl` and
/// `batch::assemble_kernel` disagreed about tap order, sub-pixel order, or the
/// space the weights apply in, training would still converge and the renderer
/// would still produce an image — a plausible, wrong one. This is the test that
/// catches that.
#[test]
#[ignore = "requires a GPU"]
fn kernel_upscale_matches_the_cpu_path() {
    check_kernel_parity(&kernel_config(), "kernel");
}

/// Demodulation multiplies the exact output-resolution albedo back after the
/// gather, so it is the one part of the reconstruction that reads a texture the
/// gather never touched. If the CPU and the shader disagreed about which texel
/// that is, every surface would come back tinted by its neighbour.
#[test]
#[ignore = "requires a GPU"]
fn demodulated_kernel_upscale_matches_the_cpu_path() {
    check_kernel_parity(&demodulating_kernel_config(), "demodulated kernel");
}

fn check_kernel_parity(config: &ModelConfig, label: &str) {
    let Some(context) = context() else { return };
    let dir = std::env::temp_dir().join(format!("ommatidia-gpu-runtime-{label}").replace(' ', "-"));
    std::fs::create_dir_all(&dir).unwrap();
    let stem = dir.join("model");
    write_checkpoint(config, &stem, Arc::clone(&context));

    let mut upscaler =
        Upscaler::from_checkpoint(Arc::clone(&context), &stem, 1, 100).expect("upscaler");
    let (out_width, out_height) = upscaler.output_extent();

    let mut rng = Rng::new(11);
    let texels = (TILE * TILE) as usize;
    // Round-tripped through f16, because that is what the dataset would store
    // and what the CPU reference will read back.
    let colors: Vec<f32> = (0..texels * 3)
        .map(|_| f16::from_f32(rng.uniform() * 12.0).to_f32())
        .collect();
    let mut depths = vec![0.0f32; texels * 3];
    let mut normals = vec![0.0f32; texels * 3];
    let mut albedos = vec![0.0f32; texels * 3];
    let specular = vec![0.25f32; texels * 3];
    for y in 0..TILE as usize {
        for x in 0..TILE as usize {
            let index = y * TILE as usize + x;
            depths[index * 3] = 2.0 + y as f32 * 0.25;
            let (normal, albedo) = if x < TILE as usize / 2 {
                ([0.0, 0.0, 1.0], [0.25, 0.5, 0.75])
            } else {
                ([0.6, 0.0, 0.8], [0.75, 0.25, 0.5])
            };
            normals[index * 3..index * 3 + 3].copy_from_slice(&normal);
            albedos[index * 3..index * 3 + 3].copy_from_slice(&albedo);
        }
    }

    // Output-resolution albedo that is not a copy of the input's, so a path
    // that read the wrong resolution would land on the wrong values.
    let hr_texels = (TILE * SCALE * TILE * SCALE) as usize;
    let mut hr_albedos = vec![0.0f32; hr_texels * 3];
    for y in 0..(TILE * SCALE) as usize {
        for x in 0..(TILE * SCALE) as usize {
            let destination = (y * (TILE * SCALE) as usize + x) * 3;
            let checker = if (x / 3 + y / 3).is_multiple_of(2) {
                0.8
            } else {
                0.2
            };
            hr_albedos[destination] = checker;
            hr_albedos[destination + 1] = checker * 0.6;
            hr_albedos[destination + 2] = checker * 0.35;
        }
    }
    let hr_depths = vec![1.0f32; hr_texels * 3];
    let hr_normals = vec![0.0f32; hr_texels * 3];

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "kernel-upscale-test",
        buffer_count: 2,
        manual_barriers: false,
    });
    encoder.start();
    let (_texture, view, _staging) = color_texture(&context, &mut encoder, &colors, TILE, TILE);
    let (_dt, depth_view, _ds) = color_texture(&context, &mut encoder, &depths, TILE, TILE);
    let (_nt, normal_view, _ns) = color_texture(&context, &mut encoder, &normals, TILE, TILE);
    let (_at, albedo_view, _as) = color_texture(&context, &mut encoder, &albedos, TILE, TILE);
    let (_st, specular_view, _ss) = color_texture(&context, &mut encoder, &specular, TILE, TILE);
    let (_hdt, hr_depth_view, _hds) = color_texture(
        &context,
        &mut encoder,
        &hr_depths,
        TILE * SCALE,
        TILE * SCALE,
    );
    let (_hnt, hr_normal_view, _hns) = color_texture(
        &context,
        &mut encoder,
        &hr_normals,
        TILE * SCALE,
        TILE * SCALE,
    );
    let (_hat, hr_albedo_view, _has) = color_texture(
        &context,
        &mut encoder,
        &hr_albedos,
        TILE * SCALE,
        TILE * SCALE,
    );

    let format = Upscaler::OUTPUT_FORMAT;
    let out_size = gpu::Extent {
        width: out_width,
        height: out_height,
        depth: 1,
    };
    let output = context.create_texture(gpu::TextureDesc {
        name: "kernel-test-output",
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
            name: "kernel-test-output",
            format,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        },
    );
    encoder.init_texture(output);

    upscaler.upscale(
        &mut encoder,
        &FrameInputs::from_textures(view, depth_view, normal_view, albedo_view, specular_view)
            .with_high_resolution_gbuffer(hr_depth_view, hr_normal_view, hr_albedo_view),
        output_view,
    );

    let readback = context.create_buffer(gpu::BufferDesc {
        name: "kernel-test-readback",
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

    // The same weights the network produced, gathered on the CPU.
    let weights = upscaler.read_residual();
    let layout = Layout {
        scale: SCALE,
        lr_width: TILE,
        lr_height: TILE,
        lr_source: ommatidia::dataset::InputSource::PathTrace,
        lr_planes: PlaneSet::new()
            .with(Plane::Color)
            .with(Plane::DiffuseAlbedo),
        hr_planes: PlaneSet::new()
            .with(Plane::Color)
            .with(Plane::DiffuseAlbedo),
    };
    let mut planar = vec![f16::ZERO; layout.lr_len()];
    let color_base = layout.lr_planes.channel_offset(Plane::Color).unwrap();
    let albedo_base = layout
        .lr_planes
        .channel_offset(Plane::DiffuseAlbedo)
        .unwrap();
    for component in 0..3 {
        for index in 0..texels {
            planar[(color_base + component) * texels + index] =
                f16::from_f32(colors[index * 3 + component]);
            planar[(albedo_base + component) * texels + index] =
                f16::from_f32(albedos[index * 3 + component]);
        }
    }
    let mut hr = vec![f16::ZERO; layout.hr_len()];
    let hr_albedo_base = layout
        .hr_planes
        .channel_offset(Plane::DiffuseAlbedo)
        .unwrap();
    for component in 0..3 {
        for index in 0..hr_texels {
            hr[(hr_albedo_base + component) * hr_texels + index] =
                f16::from_f32(hr_albedos[index * 3 + component]);
        }
    }
    let sample = Sample { lr: planar, hr };
    let expected = batch::assemble_kernel(
        &sample,
        &layout,
        Crop {
            x: 0,
            y: 0,
            tile: TILE,
        },
        &weights,
        config,
        None,
    );

    let mut worst = 0.0f32;
    for texel in 0..(out_width * out_height) as usize {
        for c in 0..3 {
            let gpu_value = halves[texel * 4 + c].to_f32();
            let cpu_value = expected[texel * 3 + c];
            let difference = (gpu_value - cpu_value).abs() / cpu_value.abs().max(1.0);
            assert!(
                difference < 2e-2,
                "texel {texel} channel {c}: gpu {gpu_value} vs cpu {cpu_value}"
            );
            worst = worst.max(difference);
        }
    }
    println!("{label} upscale: worst relative difference from the CPU path = {worst:e}");

    // A gather of non-negative samples with non-negative weights cannot leave
    // the range of what it read, whatever the network learned. Demodulation
    // rescales by the albedo afterwards, so the bound only holds without it.
    if !config.demodulate {
        let (low, high) = colors
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        for texel in 0..(out_width * out_height) as usize {
            for c in 0..3 {
                let value = halves[texel * 4 + c].to_f32();
                assert!(
                    value >= low - 1e-2 && value <= high + 1e-2,
                    "texel {texel} channel {c} left the input range: {value} not in [{low}, {high}]"
                );
            }
        }
    }
}
