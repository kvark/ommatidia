//! Contracts for independently changing sampling and training objectives.
use ommatidia::{
    field,
    transport::{cpu, graph},
};
use std::sync::Arc;
include!("fixtures/transport.rs");

#[test]
fn stratification_preserves_intervals_and_target_isolation() {
    let bounds = field::Bounds {
        center: [0.0; 3],
        radius: 1.0,
    };
    let rays = [
        field::Ray {
            origin: [0.0, 0.0, -3.0],
            direction: [0.0, 0.0, 1.0],
        },
        field::Ray {
            origin: [3.0, 3.0, -3.0],
            direction: [0.0, 0.0, 1.0],
        },
    ];
    let (mid, dt) = field::data::ray_queries(bounds, &rays, 8).unwrap();
    let a = field::data::stratified_ray_queries(bounds, &rays, 8, 7).unwrap();
    let b = field::data::stratified_ray_queries(bounds, &rays, 8, 7).unwrap();
    let c = field::data::stratified_ray_queries(bounds, &rays, 8, 11).unwrap();
    assert_eq!(dt, a.1);
    assert_eq!(a.1, b.1);
    for (x, y) in a.0.iter().zip(&b.0) {
        assert_eq!(x.position, y.position);
    }
    assert!(a.0.iter().zip(&c.0).any(|(a, b)| a.position != b.position));
    assert!(a.0.iter().zip(&mid).any(|(a, b)| a.position != b.position));
    for i in 0..8 {
        let t = a.0[2 * i].position[2] + 3.0;
        assert!(t >= 2.0 + i as f32 * 0.25 - 1e-6);
        assert!(t <= 2.0 + (i + 1) as f32 * 0.25 + 1e-6);
        assert_eq!(a.0[2 * i + 1].position, rays[1].origin);
        assert_eq!(dt[2 * i + 1], 0.0);
    }
    assert!((dt.iter().sum::<f32>() - 2.0).abs() < 1e-6);
    let t = field::surface::Targets::new(bounds, &rays, 8, &[Some((Some(2.7), 20.0)), None], 1.0)
        .unwrap();
    assert_eq!(t.mass[4], 1.0);
    assert_eq!(t.mass.iter().sum::<f32>(), 1.0);
    assert!(field::data::stratified_ray_queries(bounds, &rays, 0, 7).is_err());
}

#[test]
fn loss_controls_are_validated_and_training_only() {
    let good = graph::LossWeights::default();
    assert!(good.validate().is_ok());
    assert!(
        graph::LossWeights {
            compressed: f32::NAN,
            ..good
        }
        .validate()
        .is_err()
    );
    assert!(
        graph::LossWeights {
            temporal: -0.1,
            ..good
        }
        .validate()
        .is_err()
    );
    let m = graph::build(Config::default(), [8, 8], 0).unwrap();
    assert!(m.graph.nodes().iter().all(|n| !matches!(&n.op,
        meganeura::graph::Op::Input { name } if name.starts_with("loss."))));
    let training = graph::build(Config::default(), [8, 8], 2).unwrap();
    assert_eq!(
        m.params
            .iter()
            .map(|p| (&p.name, p.len))
            .collect::<Vec<_>>(),
        training
            .params
            .iter()
            .map(|p| (&p.name, p.len))
            .collect::<Vec<_>>()
    );
}

fn feed(session: &mut meganeura::Session) {
    let config = Config::default();
    let mut previous = Vec::new();
    for slot in 0..2 {
        let (frame, target) = fixture(config, slot, 19);
        let p = cpu::prepare(&frame, &previous, config);
        graph::feed(session, &format!("f{slot}"), &p, &target, slot);
        session.set_input(
            &format!("f{slot}.loss_scale"),
            &vec![1.0; target.lobes.len()],
        );
        let (image, _) = cpu::reconstruct(&p, &vec![1.0; p.prior.len()]);
        previous = cpu::commit(&frame, &p, &image, config).0;
    }
}

#[test]
#[ignore = "requires Vulkan or Metal"]
fn objective_weights_are_linear_and_physical_only_training_learns() {
    let context = ommatidia::gpu::create_context(None, false);
    let model = graph::build(Config::default(), [8, 8], 2).unwrap();
    let mut forward = ommatidia::gpu::inference_session(&model.graph, Arc::clone(&context));
    model.initialize(&mut forward, 7);
    feed(&mut forward);
    let mut terms = [0.0; 5];
    for (i, term) in terms.iter_mut().enumerate() {
        let mut weights = [0.0; 5];
        weights[i] = 1.0;
        forward.set_input("loss.weights", &weights);
        forward.step();
        forward.wait();
        *term = forward.read_loss();
        assert!(term.is_finite() && *term >= 0.0);
    }
    graph::LossWeights::default().feed(&mut forward);
    forward.step();
    forward.wait();
    let expected = terms
        .into_iter()
        .zip([1.0, 0.1, 0.05, 0.01, 0.01])
        .map(|(t, w)| t * w)
        .sum::<f32>();
    assert!((forward.read_loss() - expected).abs() < 1e-6 + expected * 1e-5);
    println!("unweighted objective components: {terms:?}");
    let mut train = ommatidia::gpu::training_session(&model.graph, context);
    model.initialize(&mut train, 7);
    feed(&mut train);
    graph::LossWeights {
        compressed: 0.0,
        confidence: 0.0,
        temporal: 0.0,
        ..Default::default()
    }
    .feed(&mut train);
    train.set_adam(1e-3, 0.9, 0.999, 1e-8);
    let mut first = 0.0;
    let mut last = 0.0;
    for i in 0..16 {
        train.step();
        train.wait();
        last = train.read_loss();
        assert!(last.is_finite());
        if i == 0 {
            first = last;
        }
    }
    assert!(
        last < first,
        "physical-only objective did not learn: {first} -> {last}"
    );
    println!("physical-only objective {first} -> {last}");
}
