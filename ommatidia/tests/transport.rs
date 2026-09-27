use ommatidia::transport::{self, State, cpu, graph, native};
use std::sync::Arc;
include!("fixtures/transport.rs");

fn gpu_context(timing: bool) -> Arc<blade_graphics::Context> {
    let device_id = std::env::var("MEGANEURA_DEVICE_ID")
        .ok()
        .map(|value| ommatidia::gpu::parse_device_id(&value))
        .transpose()
        .expect("invalid MEGANEURA_DEVICE_ID");
    let context = ommatidia::gpu::create_context(device_id, timing);
    println!("test adapter: {}", context.device_information().device_name);
    context
}

#[test]
fn contract_and_target_isolation() {
    let config = Config::default();
    for version in [1, 2] {
        assert!(Config { version, ..config }.validate([8, 8]).is_err());
    }
    let (frame, mut target) = fixture(config, 0, 1);
    let p = cpu::prepare(&frame, &[], config);
    target.lobes.fill(999.0);
    assert_eq!(p.features, cpu::prepare(&frame, &[], config).features);
    assert!(p.history.iter().all(|v| *v == 0.0));
    assert!(p.validity.iter().all(|v| *v == 0.0));
    let w = &p.prior;
    let n = frame.surfaces.len();
    for i in 0..2 * n {
        assert!(
            ((0..transport::CANDIDATES)
                .map(|k| w[k * 2 * n + i])
                .sum::<f32>()
                - 1.0)
                .abs()
                < 1e-5
        );
    }
    let mut s = frame.surfaces[0];
    let old = State {
        normal_depth: [0.0, 0.0, 1.0, 2.0],
        albedo_roughness: s.albedo_roughness,
        ..State::default()
    };
    assert!(!cpu::matches(&s, &old));
    s.motion[2] = 2.0;
    assert!(cpu::matches(&s, &old));
    let first = graph::build(config, frame.low, 1).unwrap();
    let recurrent = graph::build(config, frame.low, 2).unwrap();
    assert_eq!(
        first.params.len(),
        recurrent.params.len(),
        "unroll weights must be shared"
    );
}
#[test]
fn shader_parses() {
    naga::front::wgsl::parse_str(include_str!("../src/transport/prepare.wgsl")).unwrap();
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
fn lobe_supervision_detects_errors_that_cancel_in_rgb() {
    use meganeura::reference::{Feeds, evaluate_outputs};

    let config = Config {
        channels: 1,
        ..Default::default()
    };
    let model = graph::build(config, [4, 4], 1).unwrap();
    let n = 64;
    let mut feeds = Feeds::new();
    feeds.fill_random(&model.graph, 7, 0.0);
    feeds.set("f0.candidates", &vec![1.0; 5 * 6 * n]);
    let mut prior = vec![0.0; 6 * 2 * n];
    prior[..2 * n].fill(1.0);
    feeds.set("f0.prior", &prior);
    feeds.set("f0.rgb.albedo", &vec![0.5; 3 * n]);
    feeds.set("f0.rgb.target", &vec![1.5; 3 * n]);
    feeds.set("f0.target", &vec![1.0; 6 * n]);
    let loss = |feeds: &Feeds| evaluate_outputs(&model.graph, feeds).unwrap()[0].data[0];
    let mut weights = graph::LossWeights::default();
    feeds.set("loss.weights", &weights.values());
    assert!(loss(&feeds) < 1e-20);

    // Both (D=1, S=1) and (D=1.5, S=0.75) compose to RGB=1.5 at albedo=0.5.
    let mut wrong_lobes = vec![1.5; 6 * n];
    wrong_lobes[3 * n..].fill(0.75);
    feeds.set("f0.target", &wrong_lobes);
    assert!(loss(&feeds) > 1e-4);
    weights.lobes = 0.0;
    feeds.set("loss.weights", &weights.values());
    assert!(
        loss(&feeds) < 1e-20,
        "RGB-only objective should be blind to the decomposition"
    );
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
#[ignore = "requires Vulkan or Metal; set MEGANEURA_DEVICE_ID to select the adapter"]
fn inference_matches_reference_with_nonzero_head() {
    use meganeura::reference::{Feeds, evaluate_outputs};
    use ommatidia::neural::InitKind;

    let context = gpu_context(false);
    // Exercise the retained width and its capacity control through the actual
    // production session helper. A zero head would hide core precision errors.
    for channels in [16, 32] {
        let config = Config {
            channels,
            ..Config::default()
        };
        let model = graph::build(config, [8, 8], 0).unwrap();
        let mut session = ommatidia::gpu::inference_session(&model.graph, Arc::clone(&context));
        let mut feeds = Feeds::new();
        let mut rng = ommatidia::rng::Rng::new(71);
        for param in &model.params {
            let scale = match param.kind {
                InitKind::Kaiming { fan_in } => (2.0 / fan_in as f32).sqrt(),
                InitKind::Zeros => 0.02,
            };
            let values: Vec<_> = (0..param.len).map(|_| scale * rng.normal()).collect();
            feeds.set(&param.name, &values);
            session.set_parameter(&param.name, &values);
        }
        let mut previous = Vec::new();
        for step in 0..2 {
            let (frame, _) = fixture(config, step, 19);
            let p = cpu::prepare(&frame, &previous, config);
            for (name, values) in [
                ("f0.features", &p.features),
                ("f0.candidates", &p.candidates),
                ("f0.prior", &p.prior),
                ("f0.history", &p.history),
            ] {
                feeds.set(name, values);
                session.set_input(name, values);
            }
            let expected = evaluate_outputs(&model.graph, &feeds).unwrap();
            session.step();
            session.wait();
            let mut actual = vec![0.0; expected[0].len()];
            session.read_output_by_index(0, &mut actual);
            assert!(actual.iter().all(|v| v.is_finite()));
            let error = actual
                .iter()
                .zip(&expected[0].data)
                .map(|(&got, &want)| (f64::from(got) - want).abs() / (1.0 + want.abs()))
                .fold(0.0, f64::max);
            println!("inference width={channels}, frame={step}: normalized error {error}");
            assert!(error <= 1e-5, "inference/reference mismatch: {error}");
            // Identical, independently prepared inputs test cold and valid
            // history; the native recurrence tests cover GPU state evolution.
            let image = cpu::reconstruct(&p);
            previous = cpu::commit(&frame, &p, &image, config).0;
        }
    }
}

#[test]
#[ignore = "requires Vulkan or Metal; set MEGANEURA_DEVICE_ID to select the adapter"]
fn two_frame_training_matches_reference() {
    use meganeura::reference::{Feeds, gpu, gradients};
    use ommatidia::neural::InitKind;

    {
        let config = Config {
            // Shape-specific diagnostics can exercise the same graph at the
            // proposed width before changing the retained default.
            channels: std::env::var("OMMATIDIA_TEST_CHANNELS")
                .map(|v| v.parse().expect("OMMATIDIA_TEST_CHANNELS must be a u32"))
                .unwrap_or(Config::default().channels),
            ..Config::default()
        };
        let model = graph::build(config, [8, 8], 2).unwrap();
        let mut feeds = Feeds::new();
        let mut rng = ommatidia::rng::Rng::new(71);
        for param in &model.params {
            let values = match &param.kind {
                InitKind::Kaiming { fan_in } => (0..param.len)
                    .map(|_| rng.normal() * (2.0 / *fan_in as f32).sqrt())
                    .collect::<Vec<_>>(),
                // A nonzero head exposes gradients throughout the image pyramid.
                InitKind::Zeros => (0..param.len).map(|_| 0.02 * rng.normal()).collect(),
            };
            feeds.set(&param.name, &values);
        }
        let weights = graph::LossWeights::default();
        feeds.set("loss.weights", &weights.values());
        let mut previous = Vec::new();
        for step in 0..2 {
            let (frame, target) = fixture(config, step, 19);
            let p = cpu::prepare(&frame, &previous, config);
            let tag = format!("f{step}");
            feeds.set(&format!("{tag}.features"), &p.features);
            feeds.set(&format!("{tag}.candidates"), &p.candidates);
            feeds.set(&format!("{tag}.prior"), &p.prior);
            feeds.set(&format!("{tag}.target"), &target.lobes);
            let n = frame.surfaces.len();
            let width = frame.low[0] as usize * config.scale as usize;
            let mut albedo = vec![0.0; 3 * n];
            let mut emission = albedo.clone();
            let mut rgb = albedo.clone();
            for (i, surface) in frame.surfaces.iter().enumerate() {
                for c in 0..3 {
                    let j = config.index(frame.low, c, i % width, i / width);
                    albedo[j] = surface.albedo_roughness[c];
                    emission[j] = surface.emission[c];
                    rgb[j] = target.rgb[3 * i + c];
                }
            }
            feeds.set(&format!("{tag}.rgb.albedo"), &albedo);
            feeds.set(&format!("{tag}.rgb.emission"), &emission);
            feeds.set(&format!("{tag}.rgb.target"), &rgb);
            if step == 0 {
                assert!(p.validity.iter().all(|&v| v == 0.0));
                feeds.set(&format!("{tag}.history"), &p.history);
            } else {
                assert!(p.validity.contains(&0.0) && p.validity.contains(&1.0));
                for k in 0..4 {
                    feeds.set_u32(&format!("{tag}.warp{k}"), &p.indices[k]);
                    feeds.set(&format!("{tag}.coeff{k}"), &p.coefficients[k]);
                }
                let n = frame.surfaces.len();
                let mask: Vec<_> = (0..6 * n)
                    .map(|i| p.validity[i / (3 * n) * n + i % n])
                    .collect();
                feeds.set(&format!("{tag}.temporal_mask"), &mask);
            }
            let image = cpu::reconstruct(&p);
            previous = cpu::commit(&frame, &p, &image, config).0;
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
        println!("residual, finite differences\n{report}");
        assert!(report.passed(), "residual\n{report}");
        // Keep the production loss-only graph: extra outputs change buffer lifetimes.
        for (name, options) in gpu::Options::lowerings() {
            let report = gpu::check_training(&model.graph, &feeds, &options).unwrap();
            println!("residual, {name}\n{report}");
            assert!(report.passed(), "residual, {name}\n{report}");
        }
    }
}

#[test]
#[ignore = "requires Vulkan or Metal"]
fn native_history_caps_match_cpu_after_continuous_accumulation() {
    let context = gpu_context(false);
    let surface = Surface {
        normal_depth: [0.0, 0.0, 1.0, 3.0],
        albedo_roughness: [0.5, 0.5, 0.5, 1.0],
        motion: [0.0, 0.0, 3.0, 0.0],
        ..Default::default()
    };
    let ray = Ray {
        diffuse: [0.5, 0.25, 0.125, 1.0],
        specular: [2.0, 1.0, 0.5, 1.0],
        normal_depth: surface.normal_depth,
        albedo_roughness: surface.albedo_roughness,
    };
    let frame = Frame {
        low: [4, 8],
        jitter: [0.0; 2],
        rays: vec![ray; 32],
        surfaces: vec![surface; 128],
    };
    for diffuse_frames in [16.0, 32.0] {
        let config = Config {
            channels: 2,
            diffuse_frames,
            ..Config::default()
        };
        let mut native =
            native::Native::new(std::sync::Arc::clone(&context), config, frame.low).unwrap();
        let mut old = Vec::new();
        for step in 1..=40 {
            let prepared = cpu::prepare(&frame, &old, config);
            let (states, expected) =
                cpu::commit(&frame, &prepared, &cpu::reconstruct(&prepared), config);
            let actual = native.process(&frame).unwrap();
            for (a, b) in actual.iter().zip(&expected) {
                assert!((a - b).abs() / (1.0 + b.abs()) < 1e-5);
            }
            for state in native.read_state() {
                assert!((state.diffuse[3] - (step as f32).min(diffuse_frames)).abs() < 1e-5);
                assert!(
                    (state.specular[3] - (step as f32).min(config.specular_frames)).abs() < 1e-5
                );
            }
            old = states;
        }
    }
}

#[test]
#[ignore = "requires Vulkan or Metal"]
fn native_multiscale_recurrence_reset_and_hdr() {
    let context = gpu_context(false);
    let config = Config::default();
    let mut native = native::Native::new(context, config, [8, 8]).unwrap();
    let mut old = Vec::new();
    let mut worst = 0.0f32;
    for iteration in 0..24 {
        let step = iteration % 12;
        let (mut frame, _) = fixture(config, step, 71);
        if iteration >= 12 {
            // A sloped surface with parallel foreground/background layers.
            // Quantized normal lengths must not amplify spatial weights.
            for (i, s) in frame.surfaces.iter_mut().enumerate() {
                s.normal_depth[2] = 1.005;
                s.normal_depth[3] /= 1.0 + 0.045 * (i / 16) as f32;
            }
            for (i, r) in frame.rays.iter_mut().enumerate() {
                r.normal_depth[2] = 0.995;
                let y = (i / 8) as f32 * 2.0 + 0.5 + frame.jitter[1] * 2.0;
                r.normal_depth[3] /= 1.0 + 0.045 * y;
            }
        }
        if step == 0 || step == 8 {
            native.reset();
            old.clear();
        }
        if step == 6 {
            for s in &mut frame.surfaces {
                s.motion[3] = 1.0;
            }
        }
        let p = cpu::prepare(&frame, &old, config);
        let image = cpu::reconstruct(&p);
        let (states, expected) = cpu::commit(&frame, &p, &image, config);
        let actual = native.process(&frame).unwrap();
        let gpu_prepared = native.read_prepared(&frame, &old);
        assert_eq!(gpu_prepared.indices, p.indices);
        assert_eq!(gpu_prepared.coefficients, p.coefficients);
        for (name, gpu, cpu) in [
            ("candidates", &gpu_prepared.candidates, &p.candidates),
            ("history", &gpu_prepared.history, &p.history),
            ("prior", &gpu_prepared.prior, &p.prior),
        ] {
            for (a, b) in gpu.iter().zip(cpu) {
                assert!(
                    (a - b).abs() / (1.0 + b.abs()) < 0.005,
                    "{name}: {a} vs {b}"
                );
            }
        }
        for (a, b) in gpu_prepared
            .moments
            .iter()
            .flatten()
            .zip(p.moments.iter().flatten())
        {
            assert!(
                (a - b).abs() / (1.0 + b.abs()) < 0.005,
                "moments: {a} vs {b}"
            );
        }
        for (a, b) in gpu_prepared
            .ages
            .iter()
            .flatten()
            .zip(p.ages.iter().flatten())
        {
            assert!((a - b).abs() < 0.02, "ages: {a} vs {b}");
        }
        assert_eq!(native.read_history_validity(), p.validity);
        for (a, b) in native.read_features().iter().zip(&p.features) {
            assert!(
                (a - b).abs() < 0.005,
                "CPU/WGSL feature mismatch: {a} vs {b}"
            );
        }
        for (a, b) in actual.iter().zip(&expected) {
            assert!(a.is_finite());
            worst = worst.max((a - b).abs() / (1.0 + b.abs()));
        }
        let native_state = native.read_state();
        for (a, b) in native_state.iter().zip(&states) {
            assert!((a.diffuse[3] - b.diffuse[3]).abs() < 0.02);
        }
        old = states;
    }
    assert!(
        worst < 0.005,
        "CPU/native reconstruction discrepancy {worst}"
    );
    println!("24-frame flat/sloped CPU/native discrepancy {worst}");
    native.reset();
    let (mut frame, _) = fixture(config, 0, 1);
    frame.jitter = [0.0; 2];
    for r in &mut frame.rays {
        r.diffuse = [20000.0, 10000.0, 5000.0, 0.0];
        r.specular = [100.0, 200.0, 300.0, 0.0];
        r.normal_depth = [0.0, 0.0, 1.0, 1.0];
        r.albedo_roughness = [0.5, 0.5, 0.5, 0.5];
    }
    for s in &mut frame.surfaces {
        s.normal_depth = [0.0, 0.0, 1.0, 1.0];
        s.albedo_roughness = [0.5; 4];
        s.emission = [10.0, 20.0, 30.0, 0.0];
        s.motion = [0.0; 4];
    }
    let actual = native.process(&frame).unwrap();
    for v in actual.chunks_exact(3) {
        for c in 0..3 {
            let expected = [10110.0, 5220.0, 2830.0][c];
            assert!((v[c] - expected).abs() < 0.02 * expected);
        }
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
    let mut previous = Vec::new();
    for step in 0..2 {
        let (frame, target) = fixture(config, step, 19);
        let p = cpu::prepare(&frame, &previous, config);
        graph::feed(&mut session, &format!("f{step}"), &p, &target, step);
        graph::feed_rgb(&mut session, &format!("f{step}"), &frame, &target, config);
        let image = cpu::reconstruct(&p);
        previous = cpu::commit(&frame, &p, &image, config).0;
    }
    let mut first = 0.0;
    let mut last = 0.0;
    for i in 0..16 {
        session.step();
        session.wait();
        let loss = session.read_loss();
        assert!(loss.is_finite());
        if i == 0 {
            first = loss;
        }
        last = loss;
    }
    assert!(last < first, "BPTT did not learn: {first} -> {last}");
    println!("two-frame BPTT {first} -> {last}");
    let checkpoint = std::env::temp_dir().join(format!(
        "ommatidia-transport-{}.safetensors",
        std::process::id()
    ));
    session.save_checkpoint(&checkpoint).unwrap();
    let (frame, _) = fixture(config, 0, 71);
    let mut live = native::Native::new(Arc::clone(&context), config, [8, 8]).unwrap();
    live.sync_parameters(&session);
    let expected = live.process(&frame).unwrap();
    let mut native = native::Native::new(context, config, [8, 8]).unwrap();
    native.session.load_checkpoint(&checkpoint).unwrap();
    std::fs::remove_file(checkpoint).unwrap();
    let actual = native.process(&frame).unwrap();
    assert!(actual.iter().all(|v| v.is_finite()));
    assert_eq!(actual, expected, "serialized weights changed the image");
}
