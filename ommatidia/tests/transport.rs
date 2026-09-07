use ommatidia::transport::{self, State, cpu, graph, native};
use std::sync::Arc;
include!("fixtures/transport.rs");

#[test]
fn contract_and_target_isolation() {
    let config = Config::default();
    let (frame, mut target) = fixture(config, 0, 1);
    let p = cpu::prepare(&frame, &[], config);
    target.lobes.fill(999.0);
    assert_eq!(p.features, cpu::prepare(&frame, &[], config).features);
    assert!(p.history.iter().all(|v| *v == 0.0));
    assert!(p.validity.iter().all(|v| *v == 0.0));
    let (_, w) = cpu::reconstruct(&p, &vec![1.0; p.prior.len()]);
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
        let (image, _) = cpu::reconstruct(&p, &vec![1.0; p.prior.len()]);
        let (states, expected) = cpu::commit(&frame, &p, &image, config);
        let actual = native.process(&frame).unwrap();
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
        let (image, _) = cpu::reconstruct(&p, &vec![1.0; p.prior.len()]);
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
    let mut native = native::Native::new(context, config, [8, 8]).unwrap();
    native.session.load_checkpoint(&checkpoint).unwrap();
    std::fs::remove_file(checkpoint).unwrap();
    let (frame, _) = fixture(config, 0, 71);
    assert!(
        native
            .process(&frame)
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    );
}
