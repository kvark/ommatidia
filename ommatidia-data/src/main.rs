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
    canonical_bounces: u32,
    input_frames: usize,
    sequence_frames: usize,
    camera_motion: f32,
    random_camera_motion: f32,
    object_motion: f32,
    enclosed: bool,
    ground_patches: usize,
    seed: u64,
    preview: Option<PathBuf>,
    gbuffer: bool,
    hr_gbuffer: bool,
    svgf_input: bool,
    restir_input: bool,
    checkpoint: Option<PathBuf>,
    reference_from: Option<PathBuf>,
    device_id: Option<u32>,
    shader_dir: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            out: PathBuf::from("data/train.omd"),
            samples: 64,
            lr_width: 128,
            lr_height: 128,
            scale: 2,
            canonical_frames: 1024,
            canonical_bounces: render::REFERENCE_MAX_BOUNCES,
            input_frames: 1,
            sequence_frames: 1,
            camera_motion: 0.0,
            random_camera_motion: 0.0,
            object_motion: 0.0,
            enclosed: false,
            ground_patches: 0,
            seed: 0,
            preview: None,
            gbuffer: true,
            hr_gbuffer: false,
            svgf_input: false,
            restir_input: false,
            checkpoint: None,
            reference_from: None,
            device_id: None,
            shader_dir: None,
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
  --canonical-frames N      accumulated reference frames, 4 spp each [1024]
  --canonical-bounces N     maximum reference path depth [8]
  --input-frames N          sparse path-traced input samples per pixel [1]
  --sequence-frames N       consecutive frames per scene [1]
  --camera-motion F         world-X camera translation per sequence frame [0]
  --random-camera-motion F  deterministic curved camera motion, with nominal
                            translation F per frame [0]
  --enclosed                put the scene in a room, so the emissive spheres
                            are the only light. Without it the fallback
                            environment is a white furnace and nothing in the
                            frame can be in shadow
  --ground-patches N        subdivide the central ground into N by N patches
                            with independent albedo, giving the frame detail
                            finer than a whole object  [0]
  --object-motion F         move one sphere and one box independently, with
                            nominal translation F per frame [0]
  --seed N                  base seed for scenes and cameras  [0]
  --device-id ID            adapter ID for this standalone process (hex or decimal)
  --shader-dir PATH         blade-render shader directory [../blade/blade-render/code]
  --preview DIR             also write PNGs of the first few pairs
  --no-gbuffer              store only colour, leaving out the G-buffer planes
  --hr-gbuffer              also store a high-resolution G-buffer for
                            geometry-aware upsampling experiments
  --svgf-input              capture Blade's built-in variance-guided filter
                            instead of raw ReSTIR; baseline comparisons only
  --restir-input            capture raw ReSTIR instead of sparse path tracing;
                            baseline comparisons only
  --checkpoint STEM         also run this Ommatidium checkpoint directly on
                            the live Blade views and write predicted previews
  --reference-from PATH     copy high-resolution records from a matched .omd
                            instead of rendering them again
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
            "--canonical-bounces" => {
                args.canonical_bounces = value()?
                    .parse()
                    .map_err(|e| format!("--canonical-bounces: {e}"))?
            }
            "--input-frames" => {
                args.input_frames = value()?
                    .parse()
                    .map_err(|e| format!("--input-frames: {e}"))?
            }
            "--sequence-frames" => {
                args.sequence_frames = value()?
                    .parse()
                    .map_err(|e| format!("--sequence-frames: {e}"))?
            }
            "--camera-motion" => {
                args.camera_motion = value()?
                    .parse()
                    .map_err(|e| format!("--camera-motion: {e}"))?
            }
            "--random-camera-motion" => {
                args.random_camera_motion = value()?
                    .parse()
                    .map_err(|e| format!("--random-camera-motion: {e}"))?
            }
            "--object-motion" => {
                args.object_motion = value()?
                    .parse()
                    .map_err(|e| format!("--object-motion: {e}"))?
            }
            "--enclosed" => args.enclosed = true,
            "--ground-patches" => {
                args.ground_patches = value()?
                    .parse()
                    .map_err(|e| format!("--ground-patches: {e}"))?
            }
            "--seed" => args.seed = value()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--device-id" => args.device_id = Some(ommatidia::gpu::parse_device_id(&value()?)?),
            "--shader-dir" => args.shader_dir = Some(PathBuf::from(value()?)),
            "--preview" => args.preview = Some(PathBuf::from(value()?)),
            "--no-gbuffer" => args.gbuffer = false,
            "--hr-gbuffer" => args.hr_gbuffer = true,
            "--svgf-input" => args.svgf_input = true,
            "--restir-input" => args.restir_input = true,
            "--checkpoint" => args.checkpoint = Some(PathBuf::from(value()?)),
            "--reference-from" => args.reference_from = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown option {other:?}\n\n{USAGE}")),
        }
    }
    if args.scale < 2 {
        return Err(format!("--scale must be at least 2, got {}", args.scale));
    }
    if args.samples == 0 {
        return Err("--samples must be positive".into());
    }
    if args.input_frames == 0 {
        return Err("--input-frames must be positive".into());
    }
    if args.sequence_frames == 0 {
        return Err("--sequence-frames must be positive".into());
    }
    if !args.camera_motion.is_finite() {
        return Err("--camera-motion must be finite".into());
    }
    if !args.random_camera_motion.is_finite() || args.random_camera_motion < 0.0 {
        return Err("--random-camera-motion must be finite and non-negative".into());
    }
    if !args.object_motion.is_finite() || args.object_motion < 0.0 {
        return Err("--object-motion must be finite and non-negative".into());
    }
    if args.camera_motion != 0.0 && args.random_camera_motion != 0.0 {
        return Err("--camera-motion and --random-camera-motion are mutually exclusive".into());
    }
    let has_motion =
        args.camera_motion != 0.0 || args.random_camera_motion != 0.0 || args.object_motion != 0.0;
    if has_motion && args.sequence_frames == 1 {
        return Err("motion needs --sequence-frames above one".into());
    }
    if has_motion && args.reference_from.is_some() {
        return Err("motion cannot reuse static references".into());
    }
    if args.svgf_input && args.restir_input {
        return Err("--svgf-input and --restir-input are mutually exclusive".into());
    }
    if args.reference_from.is_some() && !args.gbuffer {
        return Err("--reference-from needs the G-buffer to verify scene alignment".into());
    }
    Ok(args)
}

struct ActiveSequence {
    objects: Vec<blade_render::Object>,
    base_camera: blade_render::Camera,
    motion_seed: u64,
    moving_start: usize,
}

fn translation_transform(offset: [f32; 3]) -> gpu::Transform {
    gpu::Transform {
        x: [1.0, 0.0, 0.0, offset[0]].into(),
        y: [0.0, 1.0, 0.0, offset[1]].into(),
        z: [0.0, 0.0, 1.0, offset[2]].into(),
    }
}

impl ActiveSequence {
    fn animate_objects(&mut self, frame: usize, step: f32) {
        for (moving_index, object) in self.objects[self.moving_start..].iter_mut().enumerate() {
            object.prev_transform = translation_transform(scene::object_motion(
                self.motion_seed,
                moving_index,
                frame.saturating_sub(1),
                step,
            ));
            object.transform = translation_transform(scene::object_motion(
                self.motion_seed,
                moving_index,
                frame,
                step,
            ));
        }
    }
}

/// Where blade-render keeps its shader sources.
///
/// The renderer loads WGSL from disk at runtime, so the generator has to be
/// told where the checkout is. It sits beside ommatidia by default.
fn shader_dir(override_dir: Option<&std::path::Path>) -> PathBuf {
    if let Some(dir) = override_dir {
        return dir.to_owned();
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../blade/blade-render/code")
        .canonicalize()
        .unwrap_or_else(|e| panic!("cannot find blade's shader directory; pass --shader-dir: {e}"))
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
    fn new(device_id: Option<u32>, shader_dir_override: Option<&std::path::Path>) -> Self {
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
                    device_id,
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
        let (shaders, task) =
            blade_render::Shaders::load(&shader_dir(shader_dir_override), &asset_hub, true);
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

/// Planar dataset RGB back to the interleaved representation used for
/// previews, peak accounting, and live frame records.
fn from_planes(rgb: &[f16], texels: usize) -> Vec<f32> {
    assert_eq!(rgb.len(), texels * 3);
    let mut out = vec![0.0; texels * 3];
    for texel in 0..texels {
        for channel in 0..3 {
            out[texel * 3 + channel] = rgb[channel * texels + texel].to_f32();
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
        assert!(
            [gbuffer::channels(false), gbuffer::channels(true)]
                .into_iter()
                .any(|channels| planes.len() == channels * texels),
            "unexpected G-buffer plane count"
        );
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

/// Report the range of every stored plane in the first record.
///
/// Worth the few lines: a G-buffer that came out empty, or a channel that was
/// silently written in the wrong order, is obvious here and invisible after a
/// training run. Depth should span the scene, normals should reach both signs,
/// and the unorm channels should stay inside `[0, 1]`. Motion is expected to
/// be zero here because the first frame of a sequence has no predecessor.
fn report_planes(record: &[f16], layout: &Layout) {
    let texels = layout.lr_texels();
    println!("first-record input planes:");
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

    // High-resolution geometry is opt-in: it costs a cheap full-resolution
    // primary-surface pass in an application, but may recover silhouettes that
    // no low-resolution signal can locate.
    let lr_planes = if args.gbuffer {
        gbuffer::plane_set(args.sequence_frames > 1).with(Plane::Color)
    } else {
        PlaneSet::new().with(Plane::Color)
    };
    let layout = Layout {
        scale: args.scale,
        lr_width: args.lr_width,
        lr_height: args.lr_height,
        lr_source: if args.svgf_input {
            InputSource::Svgf
        } else if args.restir_input {
            InputSource::RawRestir
        } else {
            InputSource::PathTrace
        },
        lr_planes,
        hr_planes: if args.hr_gbuffer {
            gbuffer::plane_set(false).with(Plane::Color)
        } else {
            PlaneSet::new().with(Plane::Color)
        },
    };

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).expect("cannot create the output directory");
    }
    let sequence_length = u32::try_from(args.sequence_frames).expect("sequence length exceeds u32");
    let mut writer = dataset::Writer::create_sequence(&args.out, layout, sequence_length)
        .expect("cannot create the dataset");
    let mut reference_reader = args.reference_from.as_ref().map(|path| {
        let reader = dataset::Reader::open(path)
            .unwrap_or_else(|e| panic!("cannot open reference dataset {}: {e}", path.display()));
        let source = reader.layout();
        assert_eq!(
            reader.sequence_length(),
            1,
            "reference source must contain independent scene records"
        );
        assert_eq!(source.scale, layout.scale, "reference scale differs");
        assert_eq!(source.lr_width, layout.lr_width, "reference width differs");
        assert_eq!(
            source.lr_height, layout.lr_height,
            "reference height differs"
        );
        assert_eq!(
            source.lr_planes,
            layout.lr_planes.without(Plane::Motion),
            "reference input planes differ"
        );
        assert!(
            source.hr_planes.contains(Plane::Color),
            "reference dataset has no high-resolution colour"
        );
        assert!(
            reader.len() >= args.samples,
            "reference dataset has {} samples, need {}",
            reader.len(),
            args.samples,
        );
        reader
    });

    let harness = Harness::new(args.device_id, args.shader_dir.as_deref());
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
    let need_hr_render = args.reference_from.is_none() || args.hr_gbuffer;
    let mut hr_renderer = need_hr_render.then(|| make_renderer(&harness, &mut encoder, hr_size));
    let lr_target = render::Target::new(&context, lr_size);
    let hr_target = need_hr_render.then(|| render::Target::new(&context, hr_size));
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
    let lr_probe = args
        .gbuffer
        .then(|| gbuffer::Probe::new(&context, lr_size, args.sequence_frames > 1));
    let hr_probe = args
        .hr_gbuffer
        .then(|| gbuffer::Probe::new(&context, hr_size, false));
    let sync_point = context.submit(&mut encoder);
    assert!(
        context.wait_for(&sync_point, 30_000).unwrap(),
        "GPU timed out during setup"
    );

    let scene_config = scene::SceneConfig {
        enclosed: args.enclosed,
        ground_patches: args.ground_patches,
        ..scene::SceneConfig::default()
    };
    let mut rng = Rng::new(args.seed);
    let started = std::time::Instant::now();
    // Watched because it is the one number that says whether the capture is
    // really high dynamic range: a peak pinned at 1.0 means something clamped.
    let mut peak = 0.0f32;

    let record_count = args
        .samples
        .checked_mul(args.sequence_frames)
        .expect("sample count overflow");
    println!(
        "generating {} records ({} scenes × {} frames) at {}x{} -> {}x{} on {}",
        record_count,
        args.samples,
        args.sequence_frames,
        lr_size.width,
        lr_size.height,
        hr_size.width,
        hr_size.height,
        context.device_information().device_name,
    );

    let mut active_sequence: Option<ActiveSequence> = None;
    for index in 0..record_count {
        let scene_index = index / args.sequence_frames;
        let sequence_frame = index % args.sequence_frames;
        if sequence_frame == 0 {
            // A fresh scene per sequence. Blade advances its stochastic frame
            // index while the optional camera and object trajectories move.
            let geometries = scene::build(
                &scene_config,
                args.seed ^ (scene_index as u64).wrapping_mul(0x9E37_79B9),
            );
            let (static_geometry, moving_geometry) = if args.object_motion == 0.0 {
                (geometries, Vec::new())
            } else {
                scene::split_moving_geometry(geometries, args.seed ^ scene_index as u64)
            };
            let mut objects = Vec::with_capacity(1 + moving_geometry.len());
            let static_model = harness
                .asset_hub
                .models
                .baker
                .create_model(&format!("scene{scene_index}"), static_geometry);
            objects.push(blade_render::Object::from(
                harness.asset_hub.models.insert(static_model),
            ));
            let moving_start = objects.len();
            for (moving_index, geometry) in moving_geometry.into_iter().enumerate() {
                let model = harness.asset_hub.models.baker.create_model(
                    &format!("scene{scene_index}-moving{moving_index}"),
                    vec![geometry],
                );
                objects.push(blade_render::Object::from(
                    harness.asset_hub.models.insert(model),
                ));
            }
            active_sequence = Some(ActiveSequence {
                objects,
                base_camera: scene::camera(&scene_config, &mut rng),
                motion_seed: args.seed ^ (scene_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03),
                moving_start,
            });
        }
        let sequence = active_sequence.as_mut().expect("sequence was initialized");
        sequence.animate_objects(sequence_frame, args.object_motion);
        let mut camera = sequence.base_camera;
        camera.pos.x += args.camera_motion * sequence_frame as f32;
        let random_offset = scene::camera_motion(
            sequence.motion_seed,
            sequence_frame,
            args.random_camera_motion,
        );
        camera.pos.x += random_offset[0];
        camera.pos.y += random_offset[1];
        camera.pos.z += random_offset[2];

        let input_pass = if args.svgf_input || args.restir_input {
            render::Pass::RealTime
        } else {
            render::Pass::PathTrace {
                frames: args.input_frames,
            }
        };
        let lr = render::capture(
            &mut lr_renderer,
            &lr_target,
            &context,
            &mut encoder,
            &harness.asset_hub,
            &sequence.objects,
            &camera,
            input_pass,
            args.svgf_input,
            lr_probe.as_ref(),
        );
        let (hr, reference_lr) = if let Some(reader) = &mut reference_reader {
            let source_layout = *reader.layout();
            let sample = reader
                .sample(scene_index)
                .unwrap_or_else(|e| panic!("cannot read reference sample {scene_index}: {e}"));
            let color_len = Plane::Color.channels() * source_layout.hr_texels();
            let gbuffer = if args.hr_gbuffer {
                render::capture(
                    hr_renderer.as_mut().expect("HR G-buffer renderer exists"),
                    hr_target.as_ref().expect("HR G-buffer target exists"),
                    &context,
                    &mut encoder,
                    &harness.asset_hub,
                    &sequence.objects,
                    &camera,
                    render::Pass::PathTrace { frames: 1 },
                    false,
                    hr_probe.as_ref(),
                )
                .gbuffer
            } else {
                None
            };
            (
                render::Frame {
                    color: from_planes(&sample.hr[..color_len], source_layout.hr_texels()),
                    gbuffer,
                },
                Some(sample.lr),
            )
        } else {
            (
                render::capture(
                    hr_renderer.as_mut().expect("reference renderer exists"),
                    hr_target.as_ref().expect("reference target exists"),
                    &context,
                    &mut encoder,
                    &harness.asset_hub,
                    &sequence.objects,
                    &camera,
                    render::Pass::Canonical {
                        frames: args.canonical_frames,
                        max_bounces: args.canonical_bounces,
                    },
                    false,
                    hr_probe.as_ref(),
                ),
                None,
            )
        };

        let predicted = match (&mut upscaler, &neural_target) {
            (Some(upscaler), Some(target)) => {
                encoder.start();
                encoder.init_texture(target.texture());
                let inputs = if input_pass == render::Pass::RealTime {
                    ommatidia::FrameInputs::from_blade(&lr_renderer)
                } else {
                    ommatidia::FrameInputs::from_color_and_blade_gbuffer(
                        lr_target.view(),
                        lr_renderer.view_gbuffer(),
                    )
                };
                let inputs = if args.hr_gbuffer {
                    inputs.with_blade_high_resolution_gbuffer(
                        hr_renderer
                            .as_ref()
                            .expect("HR G-buffer renderer exists")
                            .view_gbuffer(),
                    )
                } else {
                    inputs
                };
                upscaler.upscale(&mut encoder, &inputs, target.view());
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
        if let Some(reference_lr) = reference_lr {
            let gbuffer_start = Plane::Color.channels() * layout.lr_texels();
            let comparable_end = reference_lr.len();
            assert_eq!(
                record[gbuffer_start..comparable_end],
                reference_lr[gbuffer_start..],
                "reference sample {scene_index} describes a different scene or camera"
            );
        }
        if index == 0 {
            report_planes(&record, &layout);
        }
        writer
            .write(&Sample {
                lr: record,
                hr: to_record(&hr, layout.hr_texels()),
            })
            .expect("cannot write a sample");

        peak = peak.max(hr.color.iter().copied().fold(0.0f32, f32::max));

        if index % 8 == 0 || index + 1 == record_count {
            let done = index + 1;
            let rate = done as f32 / started.elapsed().as_secs_f32();
            println!("  {done}/{record_count} ({rate:.1}/s)");
        }
    }

    let count = writer.finish().expect("cannot finish the dataset");
    println!(
        "wrote {count} records ({} sequences) to {} in {:.1}s, peak radiance {peak:.2}",
        count as usize / args.sequence_frames,
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
    if let Some(probe) = hr_probe {
        probe.destroy(&context);
    }
    if let Some(mut upscaler) = upscaler {
        upscaler.destroy();
    }
    if let Some(target) = neural_target {
        target.destroy(&context);
    }
    lr_target.destroy(&context);
    if let Some(target) = hr_target {
        target.destroy(&context);
    }
    lr_renderer.destroy(&context);
    if let Some(mut renderer) = hr_renderer {
        renderer.destroy(&context);
    }
    context.destroy_command_encoder(&mut encoder);
    harness.destroy();
}
