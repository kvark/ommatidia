use ommatidia::transport::{cpu, graph, oracle};
use std::sync::Arc;
include!("fixtures/transport.rs");
#[test]
fn teacher_is_training_only_and_does_not_add_parameters() {
    let c = Config::default();
    assert!(graph::build_projected(c, [8, 8], 0, true).is_err());
    let a = graph::build(c, [8, 8], 1).unwrap();
    let b = graph::build_projected(c, [8, 8], 1, true).unwrap();
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
}
#[test]
#[ignore = "requires Vulkan or Metal"]
fn zero_teacher_matches_original_and_projected_colour_learns() {
    let c = Config::default();
    let (frame, target) = fixture(c, 0, 19);
    let p = cpu::prepare(&frame, &[], c);
    let candidates = oracle::Candidates {
        spatial: p.candidates.clone(),
        history: p.history.clone(),
        prior: p.prior.clone(),
        selected: Vec::new(),
    };
    let projection = oracle::reconstruct(&candidates, &target.lobes).unwrap();
    let context = ommatidia::gpu::create_context(None, false);
    let old = graph::build(c, [8, 8], 1).unwrap();
    let model = graph::build_projected(c, [8, 8], 1, true).unwrap();
    let mut original = ommatidia::gpu::inference_session(&old.graph, Arc::clone(&context));
    old.initialize(&mut original, 7);
    graph::feed(&mut original, "f0", &p, &target, 0);
    original.step();
    original.wait();
    let mut forward = ommatidia::gpu::inference_session(&model.graph, Arc::clone(&context));
    model.initialize(&mut forward, 7);
    graph::feed(&mut forward, "f0", &p, &target, 0);
    forward.set_input("f0.projected", &projection.lobes);
    forward.set_input("loss.projected_weight", &[0.0]);
    forward.step();
    forward.wait();
    assert!((original.read_loss() - forward.read_loss()).abs() < 1e-6);
    let mut train = ommatidia::gpu::training_session(&model.graph, Arc::clone(&context));
    model.initialize(&mut train, 7);
    graph::feed(&mut train, "f0", &p, &target, 0);
    train.set_input("f0.projected", &projection.lobes);
    train.set_input("loss.projected_weight", &[1.0]);
    train.set_input("loss.weights", &[0.0; 5]);
    let mut first = 0.0;
    let mut last = 0.0;
    for i in 0..32 {
        train.set_adam(0.001, 0.9, 0.999, 1e-8);
        train.step();
        train.wait();
        last = train.read_loss();
        if i == 0 {
            first = last;
        }
        assert!(last.is_finite());
    }
    assert!(last < first, "{first}->{last}");
    let path = std::env::temp_dir().join(format!("projected-{}.safetensors", std::process::id()));
    train.save_checkpoint(&path).unwrap();
    let runtime = graph::build(c, [8, 8], 0).unwrap();
    let mut restored = ommatidia::gpu::inference_session(&runtime.graph, context);
    restored.load_checkpoint(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    println!("projected colour loss {first} -> {last}");
}
