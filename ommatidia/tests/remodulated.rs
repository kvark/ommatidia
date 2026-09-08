use ommatidia::transport::{cpu, graph};
use std::sync::Arc;
include!("fixtures/transport.rs");
#[test]
fn remodulated_loss_preserves_inference_parameters() {
    let c = Config::default();
    let a = graph::build(c, [8, 8], 0).unwrap();
    let b = graph::build_objective(c, [8, 8], 2, false, true).unwrap();
    assert_eq!(
        a.params
            .iter()
            .map(|p| (&p.name, p.len))
            .collect::<Vec<_>>(),
        b.params
            .iter()
            .map(|p| (&p.name, p.len))
            .collect::<Vec<_>>()
    );
    assert!(graph::build_objective(c, [8, 8], 0, false, true).is_err());
    assert!(a.graph.nodes().iter().all(|n| !matches!(&n.op,meganeura::graph::Op::Input{name} if name.contains("rgb.")||name.contains("target"))));
}
#[test]
#[ignore = "requires Vulkan or Metal"]
fn rgb_loss_matches_remodulation_and_reloads_in_native_runtime() {
    let c = Config::default();
    let (frame, target) = fixture(c, 0, 19);
    let p = cpu::prepare(&frame, &[], c);
    let context = ommatidia::gpu::create_context(None, false);
    let m = graph::build_objective(c, frame.low, 1, false, true).unwrap();
    let mut inf = ommatidia::gpu::inference_session(&m.graph, Arc::clone(&context));
    m.initialize(&mut inf, 7);
    graph::feed(&mut inf, "f0", &p, &target, 0);
    graph::feed_rgb(&mut inf, "f0", &frame, &target, c);
    inf.set_input("loss.weights", &[0.0, 1.0, 0.0, 0.0, 0.0]);
    inf.step();
    inf.wait();
    let (lobes, _) = cpu::reconstruct(&p, &vec![std::f32::consts::LN_2; p.prior.len()]);
    let rgb = cpu::commit(&frame, &p, &lobes, c).1;
    let expected = rgb
        .iter()
        .zip(&target.rgb)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / rgb.len() as f32;
    assert!(
        (inf.read_loss() - expected).abs() < 1e-6,
        "{} != {expected}",
        inf.read_loss()
    );
    let mut train = ommatidia::gpu::training_session(&m.graph, Arc::clone(&context));
    m.initialize(&mut train, 7);
    graph::feed(&mut train, "f0", &p, &target, 0);
    graph::feed_rgb(&mut train, "f0", &frame, &target, c);
    train.set_input("loss.weights", &[0.0, 1.0, 0.0, 0.0, 0.0]);
    for _ in 0..48 {
        train.set_adam(0.001, 0.9, 0.999, 1e-8);
        train.step();
        train.wait();
        assert!(train.read_loss().is_finite());
    }
    assert!(train.read_loss() < expected);
    let path = std::env::temp_dir().join(format!("remodulated-{}.safetensors", std::process::id()));
    train.save_checkpoint(&path).unwrap();
    let mut native = ommatidia::transport::native::Native::new(context, c, frame.low).unwrap();
    native.session.load_checkpoint(&path).unwrap();
    let rgb = native.process(&frame).unwrap();
    assert!(rgb.iter().all(|v| v.is_finite()));
    let native_mse = rgb
        .iter()
        .zip(&target.rgb)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / rgb.len() as f32;
    assert!(native_mse < expected);
    std::fs::remove_file(path).unwrap();
    println!("final-RGB MSE {expected} -> {native_mse}");
}
