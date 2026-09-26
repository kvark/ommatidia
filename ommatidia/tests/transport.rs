use ommatidia::transport::{self, State, cpu, graph, native};
use std::sync::Arc;
include!("fixtures/transport.rs");

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
fn two_frame_training_matches_reference() {
    use meganeura::reference::{Feeds, gpu, gradients};
    use ommatidia::neural::InitKind;

    {
        let config = Config::default();
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
fn native_multiscale_recurrence_reset_and_hdr() {
    let context = ommatidia::gpu::create_context(None, false);
    let config = Config::default();
    let mut native = native::Native::new(context, config, [8, 8]).unwrap();
    let mut old = Vec::new();
    let mut worst = 0.0f32;
    for step in 0..12 {
        let (mut frame, _) = fixture(config, step, 71);
        if step == 8 {
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
    println!("12-frame CPU/native discrepancy {worst}");
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
    let context = ommatidia::gpu::create_context(None, false);
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
