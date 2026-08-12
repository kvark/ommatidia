//! Generate an ommatidia training set by rendering with Blade.
//!
//! ```sh
//! cargo run --release -p ommatidia-data -- --out data/train.omd --samples 256
//! ```
//!
//! Each sample is a fresh procedural scene seen from a fresh camera, rendered
//! twice: once through the real-time estimator at low resolution, once through
//! the canonical path tracer at high resolution.

mod gbuffer;
mod render;
mod scene;

use std::path::PathBuf;

use blade_graphics as gpu;
use half::f16;
use ommatidia::dataset::{self, InputSource, Layout, Plane, PlaneSet, Sample};
use ommatidia::rng::Rng;

struct Args {
    out: PathBuf,
    samples: usize,
    lr_width: u32,
    lr_height: u32,
    scale: u32,
    canonical_frames: usize,
    seed: u64,
    preview: Option<PathBuf>,
    gbuffer: bool,
    svgf_input: bool,
    checkpoint: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            out: PathBuf::from("data/train.omd"),
            samples: 64,
            lr_width: 128,
            lr_height: 128,
            scale: 2,
            canonical_frames: 256,
            seed: 0,
            preview: None,
            gbuffer: true,
            svgf_input: false,
            checkpoint: None,
        }
    }
}

const USAGE: &str = "\
generate an ommatidia training set

usage: ommatidia-data [options]

  --out PATH                where to write the dataset  [data/train.omd]
  --samples N               number of scene/camera pairs  [64]
  --lr WxH                  low resolution extent  [128x128]
  --scale S                 high resolution is low times this  [2]
  --canonical-frames N      path tracer samples per reference frame  [256]
  --seed N                  base seed for scenes and cameras  [0]
  --preview DIR             also write PNGs of the first few pairs
  --no-gbuffer              store only colour, leaving out the G-buffer planes
  --svgf-input              capture Blade's built-in variance-guided filter
                            instead of raw ReSTIR; baseline comparisons only
  --checkpoint STEM         also run this Ommatidium checkpoint directly on
                            the live Blade views and write predicted previews
  -h, --help                this message
";

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut value = || argv.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--out" => args.out = PathBuf::from(value()?),
            "--samples" => {
                args.samples = value()?.parse().map_err(|e| format!("--samples: {e}"))?
            }
            "--lr" => {
                let text = value()?;
                let (w, h) = text
                    .split_once(['x', 'X'])
                    .ok_or_else(|| format!("--lr wants WxH, got {text:?}"))?;
                args.lr_width = w.parse().map_err(|e| format!("--lr width: {e}"))?;
                args.lr_height = h.parse().map_err(|e| format!("--lr height: {e}"))?;
            }
            "--scale" => args.scale = value()?.parse().map_err(|e| format!("--scale: {e}"))?,
            "--canonical-frames" => {
                args.canonical_frames = value()?
                    .parse()
                    .map_err(|e| format!("--canonical-frames: {e}"))?
            }
            "--seed" => args.seed = value()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--preview" => args.preview = Some(PathBuf::from(value()?)),
            "--no-gbuffer" => args.gbuffer = false,
            "--svgf-input" => args.svgf_input = true,
            "--checkpoint" => args.checkpoint = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown option {other:?}\n\n{USAGE}")),
        }
    }
    if args.scale < 2 {
        return Err(format!("--scale must be at least 2, got {}", args.scale));
    }
    if args.samples == 0 {
        return Err("--samples must be positive".into());
    }
    Ok(args)
}

/// Where blade-render keeps its shader sources.
///
/// The renderer loads WGSL from disk at runtime, so the generator has to be
/// told where the checkout is. It sits beside ommatidia by default.
fn shader_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("BLADE_SHADER_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../blade/blade-render/code")
        .canonicalize()
        .unwrap_or_else(|e| {
            panic!("cannot find blade's shader directory, set BLADE_SHADER_DIR: {e}")
        })
}

/// Brings up the context, worker pool, asset hub, and cooked shaders.
struct Harness {
    context: std::sync::Arc<gpu::Context>,
    choir: std::sync::Arc<choir::Choir>,
    workers: Vec<choir::WorkerHandle>,
    asset_hub: blade_render::AssetHub,
    shaders: blade_render::Shaders,
}

impl Harness {
    fn new() -> Self {
        let context = std::sync::Arc::new(
            unsafe {
                gpu::Context::init(gpu::ContextDesc {
                    // Both estimators trace rays, so acceleration structures
                    // and ray queries have to be asked for up front.
                    ray_tracing: true,
                    validation: cfg!(debug_assertions),
                    timing: false,
                    capture: false,
                    overlay: false,
                    device_id: device_id(),
                    ..Default::default()
                })
            }
            .expect("no usable GPU context"),
        );
        assert!(
            context
                .capabilities()
                .ray_query
                .contains(gpu::ShaderVisibility::COMPUTE),
            "the generator needs ray queries in compute shaders"
        );
        log::info!("device: {}", context.device_information().device_name);

        let choir = choir::Choir::new();
        let workers = (0..num_workers())
            .map(|i| choir.add_worker(&format!("ommatidia-data-{i}")))
            .collect();
        let cache = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/data-assets");
        let asset_hub = blade_render::AssetHub::new(&cache, &choir, &context);
        let (shaders, task) = blade_render::Shaders::load(&shader_dir(), &asset_hub, true);
        task.join();

        Self {
            context,
            choir,
            workers,
            asset_hub,
            shaders,
        }
    }

    fn destroy(mut self) {
        self.asset_hub.destroy();
        self.workers.clear();
        drop(self.choir);
    }
}

/// Adapter selection, by the backend-reported numeric device ID.
///
/// On Vulkan that is the PCI device ID rather than an adapter ordinal, so it
/// is conventionally written in hex. Matches meganeura's `MEGANEURA_DEVICE_ID`
/// so a machine with several GPUs can pin both to the same one.
fn device_id() -> Option<u32> {
    let value = std::env::var("OMMATIDIA_DEVICE_ID").ok()?;
    let value = value.trim();
    let parsed = match value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        Some(hex) => u32::from_str_radix(hex, 16).ok(),
        None => value.parse().ok(),
    };
    if parsed.is_none() {
        log::warn!("ignoring invalid OMMATIDIA_DEVICE_ID={value:?}");
    }
    parsed
}

fn num_workers() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(2)
}

fn make_renderer(
    harness: &Harness,
    encoder: &mut gpu::CommandEncoder,
    size: gpu::Extent,
) -> blade_render::RayTracer {
    blade_render::RayTracer::new(
        encoder,
        &harness.context,
        harness.shaders.clone(),
        &harness.asset_hub.shaders,
        &blade_render::RenderConfig {
            surface_size: size,
            surface_info: gpu::SurfaceInfo {
                format: gpu::TextureFormat::Rgba32Float,
                alpha: gpu::AlphaMode::Ignored,
            },
            // Linear leaves the values unencoded, which is what the tone map
            // inversion in `render` assumes.
            color_space: gpu::ColorSpace::Linear,
            max_debug_lines: 1,
        },
    )
}

/// Interleaved RGB from the renderer to the planar channel-major layout the
/// dataset stores, clamped into `f16` range.
fn to_planes(rgb: &[f32], texels: usize) -> Vec<f16> {
    assert_eq!(rgb.len(), texels * 3);
    let mut out = vec![f16::ZERO; texels * 3];
    for texel in 0..texels {
        for channel in 0..3 {
            let value = rgb[texel * 3 + channel].clamp(0.0, dataset::F16_MAX);
            out[channel * texels + texel] = f16::from_f32(value);
        }
    }
    out
}

/// One record's low resolution block: the colour planes, then whatever the
/// G-buffer probe produced.
///
/// The probe already writes planar in dataset order, so this is a colour
/// transpose followed by a straight conversion.
fn to_record(frame: &render::Frame, texels: usize) -> Vec<f16> {
    let mut out = to_planes(&frame.color, texels);
    if let Some(ref planes) = frame.gbuffer {
        assert_eq!(planes.len(), gbuffer::channels() * texels);
        out.reserve(planes.len());
        // Depth is the one channel that can exceed the f16 range, since a miss
        // is recorded as a very large distance.
        out.extend(
            planes
                .iter()
                .map(|&v| f16::from_f32(v.clamp(-dataset::F16_MAX, dataset::F16_MAX))),
        );
    }
    out
}

/// Report the range of every stored plane.
///
/// Worth the few lines: a G-buffer that came out empty, or a channel that was
/// silently written in the wrong order, is obvious here and invisible after a
/// training run. Depth should span the scene, normals should reach both signs,
/// and the unorm channels should stay inside `[0, 1]`.
fn report_planes(record: &[f16], layout: &Layout) {
    let texels = layout.lr_texels();
    println!("input planes:");
    let mut channel = 0;
    for plane in layout.lr_planes.iter() {
        for component in 0..plane.channels() {
            let values = &record[channel * texels..(channel + 1) * texels];
            let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
            for value in values {
                let value = value.to_f32();
                low = low.min(value);
                high = high.max(value);
            }
            println!("  {plane:?}[{component}]  {low:>12.4} .. {high:>12.4}");
            channel += 1;
        }
    }
}

/// Tone map and write a PNG, for eyeballing that the pair actually lines up.
fn write_preview(path: &std::path::Path, rgb: &[f32], width: u32, height: u32) {
    let mut bytes = Vec::with_capacity((width * height * 4) as usize);
    for texel in rgb.chunks_exact(3) {
        for &linear in texel {
            let mapped = linear / (1.0 + linear);
            // sRGB encode so the preview looks like the frame would.
            let encoded = if mapped <= 0.0031308 {
                12.92 * mapped
            } else {
                1.055 * mapped.powf(1.0 / 2.4) - 0.055
            };
            bytes.push((encoded.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
        bytes.push(255);
    }
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = std::fs::File::create(path).expect("cannot create preview");
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&bytes)
        .unwrap();
}

fn main() {
    env_logger::init();
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };

    let lr_size = gpu::Extent {
        width: args.lr_width,
        height: args.lr_height,
        depth: 1,
    };
    let hr_size = gpu::Extent {
        width: args.lr_width * args.scale,
        height: args.lr_height * args.scale,
        depth: 1,
    };

    // The input carries the geometry and material the renderer already knew;
    // the reference is only the colour the network has to reach.
    let lr_planes = if args.gbuffer {
        gbuffer::plane_set().with(Plane::Color)
    } else {
        PlaneSet::new().with(Plane::Color)
    };
    let layout = Layout {
        scale: args.scale,
        lr_width: args.lr_width,
        lr_height: args.lr_height,
        lr_source: if args.svgf_input {
            InputSource::Svgf
        } else {
            InputSource::RawRestir
        },
        lr_planes,
        hr_planes: PlaneSet::new().with(Plane::Color),
    };

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).expect("cannot create the output directory");
    }
    let mut writer = dataset::Writer::create(&args.out, layout).expect("cannot create the dataset");

    let harness = Harness::new();
    let context = std::sync::Arc::clone(&harness.context);
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "ommatidia-data",
        buffer_count: 2,
        manual_barriers: false,
    });
    encoder.start();

    // Two renderers rather than one that resizes: the targets, reservoirs, and
    // acceleration structures all depend on the extent, and the pair alternates
    // between them on every sample.
    let mut lr_renderer = make_renderer(&harness, &mut encoder, lr_size);
    let mut hr_renderer = make_renderer(&harness, &mut encoder, hr_size);
    let lr_target = render::Target::new(&context, lr_size);
    let hr_target = render::Target::new(&context, hr_size);
    let neural_target = args
        .checkpoint
        .as_ref()
        .map(|_| render::NeuralTarget::new(&context, hr_size));
    let mut upscaler = args.checkpoint.as_ref().map(|stem| {
        ommatidia::Upscaler::from_checkpoint_for_extent(
            context.clone(),
            stem,
            [lr_size.width, lr_size.height],
            1,
            1000,
        )
        .unwrap_or_else(|e| panic!("cannot load {}: {e}", stem.display()))
    });
    // Only the input side carries a G-buffer: the reference is what the
    // network has to reach, and it is reached in colour.
    let lr_probe = args.gbuffer.then(|| gbuffer::Probe::new(&context, lr_size));
    let sync_point = context.submit(&mut encoder);
    assert!(
        context.wait_for(&sync_point, 30_000).unwrap(),
        "GPU timed out during setup"
    );

    let scene_config = scene::SceneConfig::default();
    let mut rng = Rng::new(args.seed);
    let started = std::time::Instant::now();
    // Watched because it is the one number that says whether the capture is
    // really high dynamic range: a peak pinned at 1.0 means something clamped.
    let mut peak = 0.0f32;

    println!(
        "generating {} samples at {}x{} -> {}x{} on {}",
        args.samples,
        lr_size.width,
        lr_size.height,
        hr_size.width,
        hr_size.height,
        context.device_information().device_name,
    );

    for index in 0..args.samples {
        // A fresh scene per sample, so the network sees layout variety rather
        // than one scene from many angles.
        let geometries = scene::build(
            &scene_config,
            args.seed ^ (index as u64).wrapping_mul(0x9E37_79B9),
        );
        let model = harness
            .asset_hub
            .models
            .baker
            .create_model(&format!("scene{index}"), geometries);
        let handle = harness.asset_hub.models.insert(model);
        let objects = vec![blade_render::Object::from(handle)];
        let camera = scene::camera(&scene_config, &mut rng);

        let lr = render::capture(
            &mut lr_renderer,
            &lr_target,
            &context,
            &mut encoder,
            &harness.asset_hub,
            &objects,
            &camera,
            render::Pass::RealTime,
            args.svgf_input,
            lr_probe.as_ref(),
        );
        let hr = render::capture(
            &mut hr_renderer,
            &hr_target,
            &context,
            &mut encoder,
            &harness.asset_hub,
            &objects,
            &camera,
            render::Pass::Canonical {
                frames: args.canonical_frames,
            },
            false,
            None,
        );

        let predicted = match (&mut upscaler, &neural_target) {
            (Some(upscaler), Some(target)) => {
                encoder.start();
                encoder.init_texture(target.texture());
                upscaler.upscale(
                    &mut encoder,
                    &ommatidia::FrameInputs::from_blade(&lr_renderer),
                    target.view(),
                );
                Some(target.read_linear(&context, &mut encoder))
            }
            _ => None,
        };

        if let Some(ref dir) = args.preview
            && index < 4
        {
            write_preview(
                &dir.join(format!("{index:03}-lr.png")),
                &lr.color,
                lr_size.width,
                lr_size.height,
            );
            write_preview(
                &dir.join(format!("{index:03}-hr.png")),
                &hr.color,
                hr_size.width,
                hr_size.height,
            );
            if let Some(ref predicted) = predicted {
                write_preview(
                    &dir.join(format!("{index:03}-predicted.png")),
                    predicted,
                    hr_size.width,
                    hr_size.height,
                );
            }
        }

        let record = to_record(&lr, layout.lr_texels());
        if index == 0 {
            report_planes(&record, &layout);
        }
        writer
            .write(&Sample {
                lr: record,
                hr: to_planes(&hr.color, layout.hr_texels()),
            })
            .expect("cannot write a sample");

        peak = peak.max(hr.color.iter().copied().fold(0.0f32, f32::max));

        if index % 8 == 0 || index + 1 == args.samples {
            let done = index + 1;
            let rate = done as f32 / started.elapsed().as_secs_f32();
            println!("  {done}/{} ({rate:.1}/s)", args.samples);
        }
    }

    let count = writer.finish().expect("cannot finish the dataset");
    println!(
        "wrote {count} samples to {} in {:.1}s, peak radiance {peak:.2}",
        args.out.display(),
        started.elapsed().as_secs_f32()
    );
    if peak <= 1.0 {
        println!(
            "warning: nothing in the reference exceeds 1.0, so the capture may \
             have been clamped into display range"
        );
    }

    if let Some(probe) = lr_probe {
        probe.destroy(&context);
    }
    if let Some(mut upscaler) = upscaler {
        upscaler.destroy();
    }
    if let Some(target) = neural_target {
        target.destroy(&context);
    }
    lr_target.destroy(&context);
    hr_target.destroy(&context);
    lr_renderer.destroy(&context);
    hr_renderer.destroy(&context);
    context.destroy_command_encoder(&mut encoder);
    harness.destroy();
}
