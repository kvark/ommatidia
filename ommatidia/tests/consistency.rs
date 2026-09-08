use ommatidia::field::{
    self, consistency,
    data::{Prepared, RenderShape, Targets},
    graph, visibility, *,
};
use std::sync::Arc;
fn config() -> Config {
    Config {
        extent: [8, 8],
        views: 2,
        channels: 4,
        hidden: 16,
        position_frequencies: 1,
        view_fusion: ViewFusion::VisibleRgb,
        ..Default::default()
    }
}
fn observations() -> Observations {
    Observations {
        bounds: Bounds {
            center: [0.0; 3],
            radius: 1.0,
        },
        views: (0..2)
            .map(|i| View {
                camera: Camera {
                    origin: [i as f32 * 0.1, 0.0, 3.0],
                    right: [1.0, 0.0, 0.0],
                    up: [0.0, 1.0, 0.0],
                    forward: [0.0, 0.0, -1.0],
                    tan_half_fov_y: 0.25,
                },
                rgb: vec![0.5; 8 * 8 * 3],
            })
            .collect(),
    }
}
#[test]
fn consistency_has_no_new_weights_or_target_position_inputs() {
    let c = config();
    let s = RenderShape {
        rays: 4,
        steps: 16,
        probes: 0,
    };
    let a = graph::build_surface_training(&c, s, 0).unwrap();
    let b = graph::build_consistent_training(&c, s, 0, 2, true).unwrap();
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
    assert!(
        graph::build_consistent_training(&c, RenderShape { steps: 18, ..s }, 0, 2, true).is_err()
    );
    assert!(graph::build_consistent_training(&c, s, 2, 2, true).is_err());
    let obs = observations();
    let a = consistency::Batch::new(&obs, &c, 16, 7).unwrap();
    let b = consistency::Batch::new(&obs, &c, 16, 7).unwrap();
    assert_eq!(a.pixels, b.pixels);
    assert_eq!(a.views, b.views);
    let inference = graph::build_diagnostics(&c, s).unwrap();
    assert!(inference.graph.nodes().iter().all(|n|!matches!(&n.op,meganeura::graph::Op::Input{name} if name.starts_with("target.") || name.starts_with("consistency."))));
}
#[test]
#[ignore = "requires Vulkan or Metal"]
fn consistency_matches_cpu_and_backpropagates_into_density_and_visibility() {
    let c = config();
    let obs = observations();
    let shape = RenderShape {
        rays: 4,
        steps: 16,
        probes: 0,
    };
    let batch = consistency::Batch::new(&obs, &c, 2, 7).unwrap();
    let mut rays = batch.rays.clone();
    rays.extend_from_slice(&batch.rays);
    let (q, dt) = field::data::ray_queries(obs.bounds, &rays, shape.steps).unwrap();
    let prepared = Prepared::new(&obs, &c, &q).unwrap();
    let context = ommatidia::gpu::create_context(None, false);
    let network = graph::build_consistent_training(&c, shape, 0, 2, false).unwrap();
    let mut train = ommatidia::gpu::training_session(&network.graph, Arc::clone(&context));
    network.initialize(&mut train, 7);
    let diagnostic = graph::build_diagnostics(&c, shape).unwrap();
    let mut inf = ommatidia::gpu::inference_session(&diagnostic.graph, Arc::clone(&context));
    diagnostic.initialize(&mut inf, 7);
    prepared.feed(&mut inf);
    inf.set_input("ray.deltas", &dt);
    inf.step();
    inf.wait();
    let mut mass = vec![0.0; (shape.steps + 1) * shape.rays];
    inf.read_output_by_index(1, &mut mass);
    let mut cpu = 0.0;
    for r in 0..2 {
        let mut source = vec![0.0; 64 * 17];
        inf.read_output_by_index(2 + batch.views[r], &mut source);
        let p = batch.pixels[r];
        let volume: Vec<_> = (0..=shape.steps)
            .map(|s| mass[s * shape.rays + r + 2])
            .collect();
        cpu += consistency::cdf_error(&source[p * 17..(p + 1) * 17], &volume).unwrap() / 2.0;
    }
    let labels: Vec<_> = (0..c.views)
        .map(|_| {
            Some(surface::Capture {
                version: 1,
                pixel_filter: "center".into(),
                ray_limit: 20.0,
                distance: vec![None; 64],
                emission: vec![[0.0; 3]; 64],
            })
        })
        .collect();
    let targets = Targets {
        rgb: inf.read_output(shape.rays * 3)[..6].to_vec(),
        emission: vec![0.0; q.len() * 3],
        emission_mask: vec![0.0; q.len() * 3],
        environment: [0.0; 3],
        environment_mask: [0.0; 3],
    };
    let mut forward = ommatidia::gpu::inference_session(&network.graph, Arc::clone(&context));
    network.initialize(&mut forward, 7);
    prepared.feed(&mut forward);
    forward.set_input("ray.deltas", &dt);
    targets.feed(&mut forward);
    visibility::Targets::new(&obs, &c, &labels, 0.0)
        .unwrap()
        .feed(&mut forward);
    batch.feed(&mut forward, 0.0).unwrap();
    forward.step();
    forward.wait();
    let zero = forward.read_loss();
    assert!(zero.abs() < 1e-6, "non-consistency losses must be zero");
    batch.feed(&mut forward, 1.0).unwrap();
    forward.step();
    forward.wait();
    let one = forward.read_loss();
    assert!(
        ((one - zero) as f64 - cpu).abs() < 2e-5,
        "GPU increment {} != CDF error {cpu}",
        one - zero
    );
    prepared.feed(&mut train);
    train.set_input("ray.deltas", &dt);
    targets.feed(&mut train);
    visibility::Targets::new(&obs, &c, &labels, 0.0)
        .unwrap()
        .feed(&mut train);
    batch.feed(&mut train, 1.0).unwrap();
    let mut old_density = [0.0];
    train.read_param("field.density.bias", &mut old_density);
    for step in 0..16 {
        train.set_adam(1e-3, 0.9, 0.999, 1e-8);
        train.step();
        train.wait();
        assert!(train.read_loss().is_finite());
        if step == 0 {
            let mut density = [0.0];
            train.read_param("field.density.bias", &mut density);
            assert_ne!(old_density, density, "isolated CDF loss must reach density");
            let mut visibility = vec![0.0; c.channels as usize * 17];
            train.read_param("field.visibility.weight", &mut visibility);
            assert!(visibility.iter().any(|v| v.abs() > 1e-6));
        }
    }
    assert!(train.read_loss() < one);
    let mut density = [0.0];
    train.read_param("field.density.bias", &mut density);
    assert_ne!(old_density, density);
    let mut weights = vec![0.0; c.channels as usize * 17];
    train.read_param("field.visibility.weight", &mut weights);
    assert!(weights.iter().any(|v| v.abs() > 1e-6));
    let path = std::env::temp_dir().join(format!("consistency-{}.safetensors", std::process::id()));
    train.save_checkpoint(&path).unwrap();
    inf.load_checkpoint(&path).unwrap();
    prepared.feed(&mut inf);
    inf.step();
    inf.wait();
    assert!(
        inf.read_output(shape.rays * 3)
            .iter()
            .all(|v| v.is_finite())
    );
    std::fs::remove_file(path).unwrap();
    println!(
        "CDF consistency CPU {cpu}; optimizer loss {one} -> {}",
        train.read_loss()
    );
}
