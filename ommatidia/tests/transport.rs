use meganeura::reference::{Feeds, evaluate_outputs};
use ommatidia::transport::{State, cpu, graph, native};
use std::sync::Arc;
include!("fixtures/transport.rs");

fn gpu_context(timing: bool) -> Arc<blade_graphics::Context> {
    let device_id = std::env::var("MEGANEURA_DEVICE_ID")
        .ok()
        .map(|v| ommatidia::gpu::parse_device_id(&v))
        .transpose()
        .unwrap();
    let context = ommatidia::gpu::create_context(device_id, timing);
    println!("test adapter: {}", context.device_information().device_name);
    context
}

// The Phase 5 capacity sweep changes size, not architecture. Run the same
// reference/gradient gates for each planned size without duplicating tests.
fn capacity_config() -> Config {
    let setting = |name, default| {
        std::env::var(name)
            .map(|v| v.parse().unwrap_or_else(|_| panic!("{name} must be a u32")))
            .unwrap_or(default)
    };
    let config = Config {
        channels: setting("OMMATIDIA_TEST_CHANNELS", Config::default().channels),
        levels: setting("OMMATIDIA_TEST_LEVELS", Config::default().levels),
        ..Config::default()
    };
    println!(
        "capacity check: width={}, levels={}",
        config.channels, config.levels
    );
    config
}

fn observation_feeds(feeds: &mut Feeds, tag: &str, p: &cpu::Prepared, first: bool) {
    for (name, values) in [
        ("features", p.features.as_slice()),
        ("samples", &p.samples),
        ("metadata", &p.metadata),
        ("valid", &p.validity),
        ("exposure", &[p.exposure]),
    ] {
        feeds.set(&format!("{tag}.{name}"), values);
    }
    if first {
        feeds.set(&format!("{tag}.history"), &p.history);
    }
    for k in 0..4 {
        feeds.set_u32(&format!("{tag}.warp{k}"), &p.indices[k]);
        feeds.set(&format!("{tag}.coeff{k}"), &p.coefficients[k]);
    }
}
fn target_feeds(feeds: &mut Feeds, tag: &str, frame: &Frame, target: &Target, config: Config) {
    let n = frame.surfaces.len();
    let width = (frame.low[0] * config.scale) as usize;
    let mut albedo = vec![0.0; 3 * n];
    let mut emission = albedo.clone();
    let mut rgb = albedo.clone();
    for (i, s) in frame.surfaces.iter().enumerate() {
        for c in 0..3 {
            let j = config.index(frame.low, c, i % width, i / width);
            albedo[j] = s.albedo_roughness[c];
            emission[j] = s.emission[c];
            rgb[j] = target.rgb[3 * i + c];
        }
    }
    feeds.set(&format!("{tag}.target"), &target.lobes);
    feeds.set(&format!("{tag}.rgb.albedo"), &albedo);
    feeds.set(&format!("{tag}.rgb.emission"), &emission);
    feeds.set(&format!("{tag}.rgb.target"), &rgb);
}
fn nonzero_parameters(model: &graph::Network) -> Vec<Vec<f32>> {
    use ommatidia::neural::InitKind;
    let mut rng = ommatidia::rng::Rng::new(71);
    model
        .params
        .iter()
        .map(|p| {
            let (scale, bias) = match p.kind {
                InitKind::Kaiming { fan_in } => ((2.0 / fan_in as f32).sqrt(), 0.0),
                InitKind::Zeros => (0.002, 0.0),
                InitKind::Constant(bias) => (0.002, bias),
            };
            (0..p.len).map(|_| bias + scale * rng.normal()).collect()
        })
        .collect()
}
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance * (1.0 + b.abs()),
            "index {i}: {a} vs {b}"
        );
    }
}

#[test]
fn contract_and_target_isolation() {
    let config = Config::default();
    for version in [1, 2, 3] {
        assert!(Config { version, ..config }.validate([8, 8]).is_err());
    }
    assert!(
        Config::parse(
            "(version:3,scale:2,channels:16,exposure:1.0,diffuse_frames:16.0,specular_frames:8.0)"
        )
        .unwrap_err()
        .contains("v4")
    );
    let ron = ron::to_string(&config).unwrap();
    assert_eq!(Config::parse(&ron).unwrap().version, 4);
    let (mut frame, mut target) = fixture(config, 0, 1);
    let p = cpu::prepare(&frame, &State::default(), config);
    target.lobes.fill(999.0);
    assert_eq!(
        p.features,
        cpu::prepare(&frame, &State::default(), config).features
    );
    assert!(p.history.iter().all(|v| *v == 0.0));
    assert!(p.validity.iter().all(|v| *v == 0.0));
    for exposure in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        frame.exposure = exposure;
        assert!(frame.validate(config).is_err());
    }
    let a = graph::build(config, [8, 8], 1).unwrap();
    let b = graph::build(config, [8, 8], 4).unwrap();
    assert_eq!(a.params.len(), b.params.len(), "weights must be tied");
    assert_eq!(a.graph.outputs().len(), 4, "loss and carried state outputs");
}
#[test]
fn shader_parses_and_validates() {
    let mut module =
        naga::front::wgsl::parse_str(include_str!("../src/transport/prepare.wgsl")).unwrap();
    // Blade supplies bindings by ShaderData field name at pipeline creation.
    for (binding, (_, variable)) in module.global_variables.iter_mut().enumerate() {
        variable.binding = Some(naga::ResourceBinding {
            group: 0,
            binding: binding as u32,
        });
    }
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn crop_margin_excludes_every_spatial_loss_and_normalizes_interior() {
    let config = Config {
        channels: 1,
        ..Config::default()
    };
    let (frame, mut target) = fixture(config, 0, 1);
    let low = frame.low;
    let n = frame.surfaces.len();
    let model = graph::build_training(config, low, 1, 4).unwrap();
    let mut feeds = Feeds::new();
    feeds.fill_random(&model.graph, 7, 0.0);
    observation_feeds(
        &mut feeds,
        "f0",
        &cpu::prepare(&frame, &State::default(), config),
        true,
    );
    target_feeds(&mut feeds, "f0", &frame, &target, config);
    feeds.set("loss.weights", &graph::LossWeights::default().values());
    for (name, values) in graph::LOSS_MASK_NAMES
        .iter()
        .zip(graph::loss_masks(config, low, [4; 4]).unwrap())
    {
        feeds.set(name, &values);
    }
    let loss = evaluate_outputs(&model.graph, &feeds).unwrap()[0].data[0];
    let mask = graph::loss_mask(config, low, 6, [4; 4]).unwrap();
    assert_eq!(mask.iter().sum::<f32>(), 6.0 * 8.0 * 8.0);
    for (i, v) in target.lobes.iter_mut().enumerate() {
        if mask[i] == 0.0 {
            *v = 12.0;
        }
    }
    for i in 0..n {
        if mask[config.index(low, 0, i % 16, i / 16)] == 0.0 {
            target.rgb[3 * i..3 * i + 3].fill(14.0);
        }
    }
    target_feeds(&mut feeds, "f0", &frame, &target, config);
    assert_eq!(
        evaluate_outputs(&model.graph, &feeds).unwrap()[0].data[0],
        loss
    );
    target.rgb[(8 * 16 + 8) * 3] += 1.0;
    target_feeds(&mut feeds, "f0", &frame, &target, config);
    assert_ne!(
        evaluate_outputs(&model.graph, &feeds).unwrap()[0].data[0],
        loss
    );
    // The exact same graph must supervise a real image edge when that margin
    // is zero, not silently discard it like an artificial crop boundary.
    for (name, values) in graph::LOSS_MASK_NAMES
        .iter()
        .zip(graph::loss_masks(config, low, [0, 4, 4, 4]).unwrap())
    {
        feeds.set(name, &values);
    }
    let with_edge = evaluate_outputs(&model.graph, &feeds).unwrap()[0].data[0];
    target.rgb[(8 * 16) * 3] += 10.0;
    target_feeds(&mut feeds, "f0", &frame, &target, config);
    assert_ne!(
        evaluate_outputs(&model.graph, &feeds).unwrap()[0].data[0],
        with_edge
    );
    assert!(graph::build_training(config, low, 1, 8).is_err());
}

#[test]
#[ignore = "requires Vulkan or Metal"]
fn alpha_output_preserves_reconstruction_and_reset_gates() {
    let context = gpu_context(false);
    let config = Config {
        channels: 2,
        ..Config::default()
    };
    let mut plain = native::Native::new(Arc::clone(&context), config, [8; 2]).unwrap();
    let mut debug = native::Native::with_alpha_output(context, config, [8; 2]).unwrap();
    for (p, values) in plain
        .network
        .params
        .iter()
        .zip(nonzero_parameters(&plain.network))
    {
        plain.session.set_parameter(&p.name, &values);
        debug.session.set_parameter(&p.name, &values);
    }
    assert!(plain.read_alpha().is_none());
    for step in 0..4 {
        let frame = fixture(config, step, 67).0;
        if step == 3 {
            plain.reset();
            debug.reset();
        }
        let expected = plain.process(&frame).unwrap();
        let actual = debug.process(&frame).unwrap();
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let alpha = debug.read_alpha().unwrap();
        assert_eq!(alpha.len(), frame.surfaces.len() * 2);
        assert!(
            alpha
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        );
        if step == 0 || step == 3 {
            assert!(alpha.iter().all(|v| *v == 0.0));
        } else {
            assert!(alpha.iter().any(|v| *v > 0.0));
        }
    }
}

#[test]
fn reset_observations_ignore_motion_without_a_previous_frame() {
    let config = Config::default();
    let (mut stationary, _) = fixture(config, 0, 1);
    for surface in &mut stationary.surfaces {
        surface.motion = [0.0; 4];
    }
    let mut moving = stationary.clone();
    for surface in &mut moving.surfaces {
        // Magnitudes observed on the two Phase 5 cold spike frames.
        surface.motion = [1305.0, -306.0, 0.0, 0.0];
    }
    let expected = cpu::prepare(&stationary, &State::default(), config);
    let actual = cpu::prepare(&moving, &State::default(), config);
    for (i, (a, b)) in expected.features.iter().zip(&actual.features).enumerate() {
        assert_eq!(
            a, b,
            "reset feature {i} must not depend on absent-frame motion"
        );
    }
    let previous = State {
        values: vec![0.0; config.state_channels() * moving.surfaces.len()],
    };
    let warm = cpu::prepare(&moving, &previous, config);
    let base = 12 * (moving.low[0] * moving.low[1]) as usize;
    for c in 0..2 {
        assert_eq!(
            warm.features[base + config.index(moving.low, 11 + c, 0, 0)],
            moving.surfaces[0].motion[c]
        );
    }
}

#[test]
#[ignore = "requires Vulkan or Metal"]
fn cold_motion_cannot_contaminate_reconstruction_or_carried_state() {
    let context = gpu_context(false);
    let config = Config {
        channels: 2,
        ..Config::default()
    };
    let mut expected = native::Native::new(Arc::clone(&context), config, [8; 2]).unwrap();
    let mut actual = native::Native::new(Arc::clone(&context), config, [8; 2]).unwrap();
    for (p, values) in expected
        .network
        .params
        .iter()
        .zip(nonzero_parameters(&expected.network))
    {
        expected.session.set_parameter(&p.name, &values);
        actual.session.set_parameter(&p.name, &values);
    }
    for step in 0..4 {
        let (mut frame, _) = fixture(config, step, 19);
        if step % 2 == 0 {
            expected.reset();
            actual.reset();
            for surface in &mut frame.surfaces {
                surface.motion = [0.0; 4];
            }
        }
        let mut changed = frame.clone();
        if step % 2 == 0 {
            for surface in &mut changed.surfaces {
                surface.motion = [1305.0, -306.0, 0.0, 0.0];
            }
        }
        close(
            &expected.process(&frame).unwrap(),
            &actual.process(&changed).unwrap(),
            0.0,
        );
        close(
            &expected.read_state().values,
            &actual.read_state().values,
            0.0,
        );
    }
}

#[test]
#[ignore = "requires Vulkan or Metal"]
fn caller_encoder_matches_offline_across_frames_and_cuts() {
    use blade_graphics as gpu;
    let context = gpu_context(false);
    let config = Config {
        channels: 1,
        ..Config::default()
    };
    let mut reference = native::Native::new(Arc::clone(&context), config, [8; 2]).unwrap();
    let mut recorded = native::Native::new(Arc::clone(&context), config, [8; 2]).unwrap();
    for (p, values) in reference
        .network
        .params
        .iter()
        .zip(nonzero_parameters(&reference.network))
    {
        reference.session.set_parameter(&p.name, &values);
        recorded.session.set_parameter(&p.name, &values);
    }
    let frames: Vec<_> = (0..9)
        .map(|step| {
            let mut frame = fixture(config, step, 53).0;
            frame.exposure = [0.25, 1.0, 4.0][step % 3];
            frame
        })
        .collect();
    let expected: Vec<_> = frames
        .iter()
        .enumerate()
        .map(|(step, frame)| {
            if step == 3 {
                reference.reset();
            }
            reference.process(frame).unwrap()
        })
        .collect();
    let buffer = |name, size, memory| context.create_buffer(gpu::BufferDesc { name, size, memory });
    let upload = |data: &[u8]| {
        let b = buffer("renderer-upload", data.len() as u64, gpu::Memory::Shared);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), b.data(), data.len());
        }
        b
    };
    // Distinct uploads/results keep host access out of the in-flight loop.
    let sources: Vec<_> = frames[..8]
        .iter()
        .map(|frame| {
            [
                upload(bytemuck::cast_slice(&frame.rays)),
                upload(bytemuck::cast_slice(&frame.surfaces)),
            ]
        })
        .collect();
    let pixels = frames[0].surfaces.len();
    let rays_size = std::mem::size_of_val(frames[0].rays.as_slice()) as u64;
    let surfaces_size = std::mem::size_of_val(frames[0].surfaces.as_slice()) as u64;
    let rgba_size = (pixels * 16) as u64;
    let rays = buffer("renderer-rays", rays_size, gpu::Memory::Device);
    let surfaces = buffer("renderer-surfaces", surfaces_size, gpu::Memory::Device);
    let output = buffer("renderer-rgba", rgba_size, gpu::Memory::Device);
    let readbacks: Vec<_> = (0..8)
        .map(|_| upload(bytemuck::cast_slice(&vec![f32::NAN; pixels * 4])))
        .collect();
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "renderer-and-network",
        buffer_count: 2,
        manual_barriers: false,
    });
    let mut in_flight = [None, None];
    for batch in 0..4 {
        // Blade rotates two command buffers; the caller owns their fences.
        // Wait only when reusing a slot, not between passes or recorded frames.
        if let Some(sync) = in_flight[batch % 2].take() {
            assert!(context.wait_for(&sync, 60_000).unwrap());
        }
        encoder.start();
        for step in batch * 2..batch * 2 + 2 {
            if step == 3 {
                recorded.reset();
            }
            let frame = &frames[step];
            {
                let mut pass = encoder.transfer("renderer-observations");
                pass.copy_buffer_to_buffer(sources[step][0].at(0), rays.at(0), rays_size);
                pass.copy_buffer_to_buffer(sources[step][1].at(0), surfaces.at(0), surfaces_size);
            }
            recorded.record_prepare(
                &mut encoder,
                rays.at(0),
                surfaces.at(0),
                frame.jitter,
                frame.exposure,
            );
            recorded.session.record(&mut encoder).unwrap();
            recorded.record_resolve(&mut encoder, surfaces.at(0), output.at(0));
            encoder.transfer("renderer-consumer").copy_buffer_to_buffer(
                output.at(0),
                readbacks[step].at(0),
                rgba_size,
            );
        }
        let sync = context.submit(&mut encoder);
        in_flight[batch % 2] = Some(sync.clone());
        recorded.session.track_submission(sync);
    }
    // Mixing in the offline path must wait for the tracked caller submission
    // before host uploads. History must carry over without an explicit host wait.
    let last = recorded.process(&frames[8]).unwrap();
    close(&last, &expected[8], 0.0);
    close(
        &recorded.read_state().values,
        &reference.read_state().values,
        0.0,
    );
    for (step, readback) in readbacks.iter().enumerate() {
        let rgba = unsafe { std::slice::from_raw_parts(readback.data().cast::<f32>(), pixels * 4) };
        assert!(rgba.chunks_exact(4).all(|p| p[3] == 1.0));
        let actual: Vec<_> = rgba
            .chunks_exact(4)
            .flat_map(|pixel| pixel[..3].iter().map(|v| v.to_bits()))
            .collect();
        let expected: Vec<_> = expected[step].iter().map(|v| v.to_bits()).collect();
        assert_eq!(actual, expected, "caller-encoder frame {step}");
    }
    drop(recorded);
    context.destroy_command_encoder(&mut encoder);
    for b in sources
        .into_iter()
        .flatten()
        .chain(readbacks)
        .chain([rays, surfaces, output])
    {
        context.destroy_buffer(b);
    }
}

#[test]
#[ignore = "requires Vulkan or Metal"]
fn accumulated_microbatch_gradient_is_the_mean_not_the_sum() {
    let context = gpu_context(false);
    let config = Config {
        channels: 1,
        ..Config::default()
    };
    let model = graph::build_training(config, [8; 2], 2, 4).unwrap();
    let mut session = ommatidia::gpu::training_session(&model.graph, context);
    for (name, values) in graph::LOSS_MASK_NAMES
        .iter()
        .zip(graph::loss_masks(config, [8; 2], [4; 4]).unwrap())
    {
        session.set_input(name, &values);
    }
    let parameters = nonzero_parameters(&model);
    for (p, v) in model.params.iter().zip(&parameters) {
        session.set_parameter(&p.name, v);
    }
    let feed = |session: &mut meganeura::Session, seed| {
        for slot in 0..2 {
            let (frame, target) = fixture(config, slot, seed);
            let state = if slot == 0 {
                State::default()
            } else {
                State {
                    values: vec![0.0; config.state_channels() * frame.surfaces.len()],
                }
            };
            let tag = format!("f{slot}");
            graph::feed(
                session,
                &tag,
                &cpu::prepare(&frame, &state, config),
                &target,
                slot,
            );
            graph::feed_rgb(session, &tag, &frame, &target, config);
        }
    };
    for seeds in [[7; 4], [7, 8, 9, 10]] {
        session.clear_optimizer();
        session.clear_grad_accumulate();
        let mut gradients = model
            .params
            .iter()
            .map(|p| vec![0.0; p.len])
            .collect::<Vec<_>>();
        for seed in seeds {
            feed(&mut session, seed);
            session.step();
            session.wait();
            for (p, sum) in model.params.iter().zip(&mut gradients) {
                let mut gradient = vec![0.0; p.len];
                session.read_param_grad(&p.name, &mut gradient);
                for (s, g) in sum.iter_mut().zip(gradient) {
                    *s += g / 4.0;
                }
            }
        }
        session.set_grad_accumulate(4);
        session.zero_grad();
        for (i, seed) in seeds.into_iter().enumerate() {
            feed(&mut session, seed);
            if i == 3 {
                session.set_learning_rate(0.125);
            }
            session.step();
            session.wait();
        }
        let names = model
            .params
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>();
        let actual = session.read_params(&names);
        for ((p, expected), (before, after)) in model
            .params
            .iter()
            .zip(gradients)
            .zip(parameters.iter().zip(actual))
        {
            for (i, ((a, b), g)) in before.iter().zip(after).zip(expected).enumerate() {
                let measured = (a - b) / 0.125;
                assert!(
                    (measured - g).abs() <= 2e-6 + 0.002 * g.abs(),
                    "{}[{i}] mean {g} vs {measured}",
                    p.name
                );
            }
        }
        session.clear_optimizer();
        for (p, v) in model.params.iter().zip(&parameters) {
            session.set_parameter(&p.name, v);
        }
    }
}
#[test]
fn v4_convolution_budget() {
    let c = Config::default();
    assert_eq!(c.input_channels(), 132);
    let model = graph::build(c, [128, 128], 0).unwrap();
    let small = graph::build(c, [8, 8], 0).unwrap().macs();
    assert_eq!(small * 256, model.macs());
    // F1 adds 52 predicted values per output pixel at the selected 16x4 size.
    // Performance is reported, not a quality gate; this exceeds v3's old count.
    assert_eq!(model.macs(), 1_076_887_552);
    assert_eq!(model.params.iter().map(|p| p.len).sum::<usize>(), 657_792);
    let loss_macs = 12 * 12 * 4 * 4 * 2 * 2;
    assert_eq!(
        graph::build(c, [8, 8], 2).unwrap().macs(),
        2 * (small + loss_macs)
    );
    println!(
        "{} parameters, {} MACs, {} FLOPs/HR pixel",
        model.params.iter().map(|p| p.len).sum::<usize>(),
        model.macs(),
        2 * model.macs() / 65536
    );
}
#[test]
fn lobe_supervision_detects_errors_that_cancel_in_rgb() {
    let config = Config {
        channels: 1,
        levels: 3, // This loss-only fixture deliberately uses a 4x4 LR grid.
        ..Default::default()
    };
    let model = graph::build(config, [4, 4], 1).unwrap();
    let n = 64;
    let mut feeds = Feeds::new();
    feeds.fill_random(&model.graph, 7, 0.0);
    for k in 0..4 {
        feeds.set_u32(
            &format!("f0.warp{k}"),
            &vec![0; config.state_channels() * n],
        );
    }
    feeds.set("f0.exposure", &[1.0]);
    feeds.set(
        "f0.samples",
        &vec![1.0; 6 * 16 * ommatidia::transport::KERNEL_TAPS],
    );
    feeds.set("f0.rgb.albedo", &vec![0.5; 3 * n]);
    feeds.set("f0.rgb.target", &vec![1.5; 3 * n]);
    feeds.set("f0.target", &vec![1.0; 6 * n]);
    let loss = |f: &Feeds| evaluate_outputs(&model.graph, f).unwrap()[0].data[0];
    let mut weights = graph::LossWeights::default();
    feeds.set("loss.weights", &weights.values());
    assert!(loss(&feeds) < 1e-20);
    let mut wrong = vec![1.5; 6 * n];
    wrong[3 * n..].fill(0.75);
    feeds.set("f0.target", &wrong);
    assert!(loss(&feeds) > 1e-4);
    weights.lobes = 0.0;
    feeds.set("loss.weights", &weights.values());
    assert!(loss(&feeds) < 1e-20);
}

#[test]
#[ignore = "requires a GPU with timestamp support"]
fn timed_execution_matches_uninstrumented_outputs() {
    let context = gpu_context(true);
    let config = Config {
        channels: 2,
        ..Config::default()
    };
    let (frame, _) = fixture(config, 0, 1);
    let mut plain = native::Native::new(Arc::clone(&context), config, frame.low).unwrap();
    let mut timed =
        native::Native::with_timing(Arc::clone(&context), config, frame.low, true).unwrap();
    assert!(timed.gpu_timings().is_none());
    for i in 0..8 {
        let (frame, _) = fixture(config, i, 1);
        assert_eq!(
            plain.process(&frame).unwrap(),
            timed.process(&frame).unwrap()
        );
        assert!(plain.gpu_timings().is_none());
        let times = timed.gpu_timings().unwrap();
        assert!(times.iter().all(|d| !d.is_zero()));
    }
    assert!(timed.buffer_memory_bytes() > timed.session.memory_summary().total_allocated_bytes());
}

#[test]
#[ignore = "requires Vulkan or Metal; set MEGANEURA_DEVICE_ID to select the adapter"]
fn inference_convolution_preserves_f32_operands() {
    // The tiny full-model fixture can miss the large-grid cooperative path.
    // A center-tap identity convolution at the production spatial extent must
    // retain this f32 value, which rounds to 1.0 in f16.
    let context = gpu_context(false);
    let mut graph = meganeura::Graph::new();
    let input = graph.input("input", &[16 * 128 * 128]);
    let weight = graph.parameter("weight", &[16 * 16 * 9]);
    let output = graph.conv2d(input, weight, 1, 16, 128, 128, 16, 3, 3, 1, 1);
    graph.set_outputs(vec![output]);
    let mut session = ommatidia::gpu::inference_session(&graph, context);
    let value = 1.0001f32;
    for dispatch in &session.plan().dispatches {
        println!("identity convolution dispatch: {:?}", dispatch.shader);
    }
    session.set_input("input", &vec![value; 16 * 128 * 128]);
    let mut weights = vec![0.0; 16 * 16 * 9];
    for channel in 0..16 {
        weights[(channel * 16 + channel) * 9 + 4] = 1.0;
    }
    session.set_parameter("weight", &weights);
    session.step();
    session.wait();
    let mut actual = vec![0.0; 16 * 128 * 128];
    session.read_output_by_index(0, &mut actual);
    assert!(actual.iter().all(|v| v.is_finite()));
    let error = actual
        .into_iter()
        .map(|got| (got - value).abs())
        .fold(0.0, f32::max);
    println!("identity convolution maximum absolute error: {error}");
    assert!(
        error <= 1e-6,
        "inference changed the input precision: {error}"
    );
}

#[test]
#[ignore = "requires Vulkan or Metal"]
fn native_recurrence_reset_hdr_matches_independent_reference() {
    let config = capacity_config();
    let context = gpu_context(false);
    let mut native = native::Native::new(context, config, [8, 8]).unwrap();
    let model = graph::build(config, [8, 8], 0).unwrap();
    let mut feeds = Feeds::new();
    for (p, values) in model.params.iter().zip(nonzero_parameters(&model)) {
        feeds.set(&p.name, &values);
        native.session.set_parameter(&p.name, &values);
    }
    let mut old = State::default();
    for iteration in 0..24 {
        let (mut frame, _) = fixture(config, iteration % 12, 71);
        if iteration % 8 == 0 {
            native.reset();
            old = State::default();
        }
        if iteration >= 12 {
            frame.exposure = 1.0 / 1024.0;
            for ray in &mut frame.rays {
                for c in 0..3 {
                    ray.diffuse[c] *= 1024.0;
                    ray.specular[c] *= 1024.0;
                }
            }
            for s in &mut frame.surfaces {
                s.emission = [10.0, 20.0, 30.0, 0.0];
            }
        }
        let p = cpu::prepare(&frame, &old, config);
        observation_feeds(&mut feeds, "f0", &p, true);
        let expected = evaluate_outputs(&model.graph, &feeds).unwrap();
        let lobes: Vec<_> = expected[0].data.iter().map(|v| *v as f32).collect();
        let latent: Vec<_> = expected[1].data.iter().map(|v| *v as f32).collect();
        let (state, rgb) = cpu::commit(&frame, &lobes, &latent, config);
        let actual = native.process(&frame).unwrap();
        let prepared = native.read_prepared(&frame);
        assert_eq!(prepared.indices, p.indices);
        for k in 0..4 {
            close(&prepared.coefficients[k], &p.coefficients[k], 1e-6);
        }
        close(&prepared.features, &p.features, 1e-6);
        close(&prepared.samples, &p.samples, 1e-6);
        close(&prepared.metadata, &p.metadata, 1e-6);
        close(&prepared.history, &p.history, 1e-5);
        close(&cpu::warp(&prepared), &cpu::warp(&p), 1e-5);
        assert_eq!(prepared.validity, p.validity);
        assert_eq!(prepared.exposure, p.exposure);
        close(&actual, &rgb, 1e-5);
        close(&native.read_state().values, &state.values, 1e-5);
        old = state;
    }
}

#[test]
#[ignore = "requires Vulkan or Metal"]
fn exposure_equivariance_through_nonzero_network_and_recurrence() {
    let config = Config::default();
    let context = gpu_context(false);
    let mut a = native::Native::new(Arc::clone(&context), config, [8, 8]).unwrap();
    let mut b = native::Native::new(context, config, [8, 8]).unwrap();
    for (p, values) in a.network.params.iter().zip(nonzero_parameters(&a.network)) {
        a.session.set_parameter(&p.name, &values);
        b.session.set_parameter(&p.name, &values);
    }
    for step in 0..8 {
        let (frame, _) = fixture(config, step, 9);
        let mut scaled = frame.clone();
        scaled.exposure /= 8.0;
        for r in &mut scaled.rays {
            for c in 0..3 {
                r.diffuse[c] *= 8.0;
                r.specular[c] *= 8.0;
            }
        }
        for s in &mut scaled.surfaces {
            for c in 0..3 {
                s.emission[c] *= 8.0;
            }
        }
        let original = a.process(&frame).unwrap();
        let actual = b.process(&scaled).unwrap();
        let expected: Vec<_> = original.iter().map(|v| 8.0 * v).collect();
        close(&actual, &expected, 1e-5);
    }
}

#[test]
#[ignore = "requires Vulkan or Metal; checks every parameter and carried output"]
fn two_frame_training_matches_reference() {
    use meganeura::reference::gradients;
    let config = capacity_config();
    let model = graph::build_training(config, [8, 8], 2, 4).unwrap();
    let mut feeds = Feeds::new();
    for (name, values) in graph::LOSS_MASK_NAMES
        .iter()
        .zip(graph::loss_masks(config, [8; 2], [0, 4, 4, 0]).unwrap())
    {
        feeds.set(name, &values);
    }
    for (p, values) in model.params.iter().zip(nonzero_parameters(&model)) {
        feeds.set(&p.name, &values);
    }
    feeds.set("loss.weights", &graph::LossWeights::default().values());
    let mut old = State::default();
    for step in 0..2 {
        let (frame, target) = fixture(config, step, 19);
        let p = cpu::prepare(&frame, &old, config);
        let tag = format!("f{step}");
        observation_feeds(&mut feeds, &tag, &p, step == 0);
        target_feeds(&mut feeds, &tag, &frame, &target, config);
        old = cpu::commit(
            &frame,
            &target.lobes,
            &vec![0.1; config.latent_channels as usize * frame.surfaces.len()],
            config,
        )
        .0;
    }
    let report = gradients::check(
        &model.graph,
        &feeds,
        &gradients::Options {
            max_elementwise: 0,
            ..Default::default()
        },
    )
    .unwrap();
    println!("finite differences\n{report}");
    assert!(report.passed(), "{report}");
    // Propagated magnitude estimates in the pinned reference helper overflow
    // to infinity on this recurrent exp graph, even though all actual values
    // and derivatives are finite. Compare every element to the f64 derivative
    // using fixed finite tolerances, not an infinite propagated allowance.
    let backward = meganeura::autodiff::differentiate(&model.graph);
    let expected = evaluate_outputs(&backward, &feeds).unwrap();
    let context = gpu_context(false);
    for fused in [true, false] {
        let mut session = meganeura::train::build(
            &model.graph,
            meganeura::SessionConfig {
                mode: meganeura::Mode::Training,
                gpu: Some(Arc::clone(&context)),
                options: meganeura::CompileOptions {
                    fuse_dispatches: fused,
                    ..Default::default()
                },
                runtime: meganeura::SessionOptions {
                    poison: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .0;
        for node in model.graph.nodes() {
            match &node.op {
                meganeura::graph::Op::Input { name } if node.ty.dtype == meganeura::DType::U32 => {
                    session.set_input_u32(name, &feeds.u32(name).unwrap())
                }
                meganeura::graph::Op::Input { name } => {
                    session.set_input(name, &feeds.f32(name).unwrap())
                }
                meganeura::graph::Op::Parameter { name } => {
                    session.set_parameter(name, &feeds.f32(name).unwrap())
                }
                _ => {}
            }
        }
        session.step();
        session.wait();
        assert_eq!(session.num_outputs(), 4);
        for (index, want) in expected[..4].iter().enumerate() {
            let mut got = vec![0.0; want.len()];
            session.read_output_by_index(index, &mut got);
            for (g, w) in got.iter().zip(&want.data) {
                assert!(g.is_finite() && w.is_finite());
                assert!(
                    (f64::from(*g) - w).abs() < 1e-5 * (1.0 + w.abs()),
                    "output {index}, fused={fused}: {g} vs {w}"
                );
            }
        }
        for (param, want) in model.params.iter().zip(&expected[4..]) {
            let mut got = vec![0.0; param.len];
            session.read_param_grad(&param.name, &mut got);
            let max = want.data.iter().map(|v| v.abs()).fold(0.0, f64::max);
            assert!(
                max.is_finite() && max > 0.0,
                "uninformative gradient {}",
                param.name
            );
            let mut worst = 0.0_f64;
            for (i, (g, w)) in got.iter().zip(&want.data).enumerate() {
                let allowed = 1e-3 * (w.abs() + 0.01 * max) + 1e-10;
                let error = (f64::from(*g) - w).abs();
                assert!(
                    g.is_finite() && w.is_finite() && error <= allowed,
                    "{}[{i}], fused={fused}: {g} vs {w}; error {error}, allowed {allowed}",
                    param.name
                );
                worst = worst.max(error / allowed);
            }
            println!(
                "fused={fused}: {} all {} gradients pass, worst tolerance fraction {worst:.5}",
                param.name, param.len
            );
        }
    }
}

#[test]
#[ignore = "requires Vulkan or Metal; set MEGANEURA_DEVICE_ID to select the adapter"]
fn production_extent_gradient_directions_match_finite_differences() {
    use ommatidia::neural::InitKind;

    let config = capacity_config();
    let context = gpu_context(false);
    let model = graph::build(config, [128, 128], 2).unwrap();
    let mut training = ommatidia::gpu::training_session(&model.graph, Arc::clone(&context));
    // Forward-only compilation of the scalar objective: finite differences do
    // not use autodiff or any backward kernel. This supplements, not replaces,
    // the small independent f64 loss/every-parameter-gradient test.
    let mut forward = ommatidia::gpu::inference_session(&model.graph, context);
    let mut rng = ommatidia::rng::Rng::new(71);
    let mut parameters = Vec::new();
    for param in &model.params {
        let scale = match param.kind {
            InitKind::Kaiming { fan_in } => (2.0 / fan_in as f32).sqrt(),
            InitKind::Zeros | InitKind::Constant(_) => 0.02,
        };
        let values: Vec<_> = (0..param.len).map(|_| scale * rng.normal()).collect();
        training.set_parameter(&param.name, &values);
        forward.set_parameter(&param.name, &values);
        parameters.push(values);
    }
    let (frame, target) = fixture_at_extent(config, [128, 128], 0, 19);
    let prepared = cpu::prepare(&frame, &State::default(), config);
    let prior = cpu::commit(
        &frame,
        &target.lobes,
        &vec![0.0; config.latent_channels as usize * frame.surfaces.len()],
        config,
    )
    .0;
    let (second, second_target) = fixture_at_extent(config, [128, 128], 1, 19);
    let second_prepared = cpu::prepare(&second, &prior, config);
    let feed = |session: &mut meganeura::Session| {
        // Re-upload so each forward probe explicitly receives identical inputs.
        graph::feed(session, "f0", &prepared, &target, 0);
        graph::feed_rgb(session, "f0", &frame, &target, config);
        graph::feed(session, "f1", &second_prepared, &second_target, 1);
        graph::feed_rgb(session, "f1", &second, &second_target, config);
    };
    feed(&mut training);
    training.step();
    training.wait();
    feed(&mut forward);
    forward.step();
    forward.wait();
    let base = training.read_loss();
    let inference_loss = forward.read_loss();
    assert!(base.is_finite() && inference_loss.is_finite());
    assert!((base - inference_loss).abs() <= 1e-5 * (1.0 + base.abs()));
    println!(
        "128x128 width={}: training loss={base}, forward loss={inference_loss}",
        config.channels
    );

    for (param, values) in model.params.iter().zip(&parameters) {
        let mut gradient = vec![0.0; param.len];
        training.read_param_grad(&param.name, &mut gradient);
        assert!(gradient.iter().all(|v| v.is_finite()));
        let norm = gradient
            .iter()
            .map(|&g| f64::from(g).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            norm > 1e-8,
            "{} has an uninformative gradient norm {norm}",
            param.name
        );
        for probe in 0..2 {
            // Gradient-aligned probes avoid loss-change cancellation. The
            // second also perturbs every coordinate in an independent direction.
            let random_scale = 0.5 / (param.len as f64).sqrt();
            let mut direction: Vec<f32> = gradient
                .iter()
                .map(|&g| {
                    let random = if probe == 0 {
                        0.0
                    } else if rng.uniform() < 0.5 {
                        -random_scale
                    } else {
                        random_scale
                    };
                    (f64::from(g) / norm + random) as f32
                })
                .collect();
            let length = direction
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt();
            for d in &mut direction {
                *d = (f64::from(*d) / length) as f32;
            }
            let expected: f64 = gradient
                .iter()
                .zip(&direction)
                .map(|(&g, &d)| f64::from(g) * f64::from(d))
                .sum();
            let mut finite = Vec::new();
            // Resolve at least ~128 loss ULPs at the smallest step, even for
            // the newly biased/latent tensors with small directional slopes.
            // Keep the convergence and derivative tolerances unchanged.
            let h_max = (512.0 * f64::from(f32::EPSILON) * f64::from(base.abs()) / expected.abs())
                .max(0.02) as f32;
            for h in [h_max, 0.5 * h_max, 0.25 * h_max] {
                let mut losses = Vec::new();
                for sign in [-1.0, 1.0] {
                    let shifted: Vec<_> = values
                        .iter()
                        .zip(&direction)
                        .map(|(&v, &d)| v + sign * h * d)
                        .collect();
                    forward.set_parameter(&param.name, &shifted);
                    feed(&mut forward);
                    forward.step();
                    forward.wait();
                    let loss = forward.read_loss();
                    assert!(loss.is_finite());
                    losses.push(f64::from(loss));
                }
                finite.push((losses[1] - losses[0]) / (2.0 * f64::from(h)));
            }
            let coarse = (4.0 * finite[1] - finite[0]) / 3.0;
            let fine = (4.0 * finite[2] - finite[1]) / 3.0;
            println!(
                "{} probe={probe}: gradient={expected:.8e}, finite={finite:?}, extrapolated={fine:.8e}",
                param.name
            );
            // Fixed before observing results. Require step-size convergence as
            // well as agreement; a noisy/nonconverged probe cannot pass.
            assert!(
                (fine - coarse).abs() <= 0.01 * expected.abs() + 1e-7,
                "{} probe {probe}: finite differences did not converge",
                param.name
            );
            assert!(
                (fine - expected).abs() <= 0.02 * expected.abs() + 1e-7,
                "{} probe {probe}: gradient mismatch",
                param.name
            );
        }
        forward.set_parameter(&param.name, values);
    }
}

#[test]
#[ignore = "requires Vulkan or Metal"]
fn two_frame_bptt_learns_and_reloads() {
    let context = gpu_context(false);
    let config = Config::default();
    let model = graph::build(config, [8, 8], 2).unwrap();
    let mut session = ommatidia::gpu::training_session(&model.graph, Arc::clone(&context));
    model.initialize(&mut session, 7);
    session.set_adam(1e-3, 0.9, 0.999, 1e-8);
    let mut old = State::default();
    for step in 0..2 {
        let (frame, target) = fixture(config, step, 19);
        let p = cpu::prepare(&frame, &old, config);
        graph::feed(&mut session, &format!("f{step}"), &p, &target, step);
        graph::feed_rgb(&mut session, &format!("f{step}"), &frame, &target, config);
        old = cpu::commit(
            &frame,
            &target.lobes,
            &vec![0.0; config.latent_channels as usize * frame.surfaces.len()],
            config,
        )
        .0;
    }
    let mut losses = Vec::new();
    for _ in 0..16 {
        session.step();
        session.wait();
        let loss = session.read_loss();
        assert!(loss.is_finite());
        losses.push(loss);
    }
    assert!(losses[15] < losses[0], "BPTT did not learn: {losses:?}");
    println!("two-frame BPTT {} -> {}", losses[0], losses[15]);
    let checkpoint =
        std::env::temp_dir().join(format!("ommatidia-v4-{}.safetensors", std::process::id()));
    session.save_checkpoint(&checkpoint).unwrap();
    let mut live = native::Native::new(Arc::clone(&context), config, [8, 8]).unwrap();
    live.sync_parameters(&session);
    let mut reloaded = native::Native::new(context, config, [8, 8]).unwrap();
    reloaded.session.load_checkpoint(&checkpoint).unwrap();
    for step in 0..8 {
        let (frame, _) = fixture(config, step, 71);
        let a = live.process(&frame).unwrap();
        let b = reloaded.process(&frame).unwrap();
        assert!(a.iter().all(|v| v.is_finite()));
        assert_eq!(a, b, "serialized weights changed frame {step}");
        assert_eq!(live.read_state().values, reloaded.read_state().values);
    }
    std::fs::remove_file(checkpoint).unwrap();
}
