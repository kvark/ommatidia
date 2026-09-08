use ommatidia::field::{
    self,
    data::{Prepared, RenderShape, Targets},
    graph, visibility, *,
};
use std::sync::Arc;
fn fixture() -> (Config, Observations) {
    let c = Config {
        extent: [8, 8],
        views: 2,
        channels: 4,
        hidden: 16,
        position_frequencies: 1,
        view_fusion: ViewFusion::StereoRgb,
        ..Default::default()
    };
    let obs = Observations {
        bounds: Bounds {
            center: [0.0; 3],
            radius: 1.0,
        },
        views: (0..2)
            .map(|i| View {
                camera: Camera {
                    origin: [(i as f32 - 0.5) * 0.4, 0.0, 3.0],
                    right: [1.0, 0.0, 0.0],
                    up: [0.0, 1.0, 0.0],
                    forward: [0.0, 0.0, -1.0],
                    tan_half_fov_y: 0.2,
                },
                rgb: (0..64)
                    .flat_map(|p| [0.2 + p as f32 / 80.0, 0.2 + i as f32 * 0.3, 0.5])
                    .collect(),
            })
            .collect(),
    };
    (c, obs)
}
#[test]
fn stereo_schema_is_versioned_bounded_and_target_free() {
    let (c, obs) = fixture();
    let m = graph::build_points(&c, 2).unwrap();
    let old = Config {
        view_fusion: ViewFusion::VisibleRgb,
        ..c.clone()
    };
    let a = graph::build_points(&old, 2).unwrap();
    assert!(m.params.len() > a.params.len());
    for (a, b) in a.params.iter().zip(&m.params) {
        assert_eq!((&a.name, a.len), (&b.name, b.len));
    }
    assert!(m.graph.nodes().iter().all(
        |n| !matches!(&n.op,meganeura::graph::Op::Input{name} if name.starts_with("target."))
    ));
    let encoded = ron::to_string(&c).unwrap();
    assert!(encoded.contains("stereo-rgb"));
    assert_eq!(
        ron::from_str::<Config>(&encoded).unwrap().view_fusion,
        c.view_fusion
    );
    let q = [Query {
        position: [0.0; 3],
        direction: [0.0, 0.0, -1.0],
    }];
    let cache = Arc::new(field::stereo::Sweep::new(&obs, &c).unwrap());
    assert_eq!(
        Prepared::new(&obs, &c, &q).unwrap(),
        Prepared::with_stereo(&obs, &c, &q, Some(cache)).unwrap()
    );
    assert!(
        Config {
            extent: [2048, 2048],
            ..c
        }
        .validate()
        .is_err()
    );
}
#[test]
#[ignore = "requires Vulkan or Metal"]
fn stereo_gradients_view_symmetry_and_checkpoint_reload() {
    let (c, obs) = fixture();
    let context = ommatidia::gpu::create_context(None, false);
    let shape = RenderShape {
        rays: 2,
        steps: 4,
        probes: 0,
    };
    let model = graph::build_render(&c, shape, true).unwrap();
    let old = Config {
        view_fusion: ViewFusion::VisibleRgb,
        ..c.clone()
    };
    let control = graph::build_render(&old, shape, true).unwrap();
    let mut train = ommatidia::gpu::training_session(&model.graph, Arc::clone(&context));
    model.initialize(&mut train, 7);
    let mut other = ommatidia::gpu::inference_session(&control.graph, Arc::clone(&context));
    control.initialize(&mut other, 7);
    for p in &control.params {
        let mut a = vec![0.0; p.len];
        let mut b = a.clone();
        train.read_param(&p.name, &mut a);
        other.read_param(&p.name, &mut b);
        assert_eq!(a, b, "{}", p.name);
    }
    let rays = [
        obs.views[0].camera.ray([3.0, 3.0], c.extent),
        obs.views[1].camera.ray([4.0, 4.0], c.extent),
    ];
    let (q, dt) = field::data::ray_queries(obs.bounds, &rays, shape.steps).unwrap();
    Prepared::new(&obs, &c, &q).unwrap().feed(&mut train);
    train.set_input("ray.deltas", &dt);
    Targets {
        rgb: vec![0.7; 6],
        emission: vec![0.0; q.len() * 3],
        emission_mask: vec![0.0; q.len() * 3],
        environment: [0.1; 3],
        environment_mask: [0.0; 3],
    }
    .feed(&mut train);
    let labels = vec![
        Some(surface::Capture {
            version: 1,
            pixel_filter: "center".into(),
            ray_limit: 20.0,
            distance: vec![Some(3.0); 64],
            emission: vec![[0.0; 3]; 64]
        });
        2
    ];
    visibility::Targets::new(&obs, &c, &labels, 0.1)
        .unwrap()
        .feed(&mut train);
    let mut first = 0.0;
    for i in 0..32 {
        train.set_adam(0.001, 0.9, 0.999, 1e-8);
        train.step();
        train.wait();
        if i == 0 {
            first = train.read_loss();
        }
        assert!(train.read_loss().is_finite());
    }
    assert!(train.read_loss() < first);
    let mut match_weights = vec![0.0; 16];
    train.read_param("field.stereo.out.weight", &mut match_weights);
    assert!(match_weights.iter().any(|v| v.abs() > 1e-6));
    let path = std::env::temp_dir().join(format!("stereo-{}.safetensors", std::process::id()));
    train.save_checkpoint(&path).unwrap();
    let q = [
        Query {
            position: [0.0; 3],
            direction: [0.0, 0.0, -1.0],
        },
        Query {
            position: [0.0; 3],
            direction: [1.0, 0.0, 0.0],
        },
    ];
    let model = graph::build_points(&c, 2).unwrap();
    let mut inf = ommatidia::gpu::inference_session(&model.graph, context);
    inf.load_checkpoint(&path).unwrap();
    Prepared::new(&obs, &c, &q).unwrap().feed(&mut inf);
    inf.step();
    inf.wait();
    let density = inf.read_output(2);
    assert!((density[0] - density[1]).abs() < 1e-6);
    let mut rgb = vec![0.0; 6];
    inf.read_output_by_index(1, &mut rgb);
    let mut swapped = obs.clone();
    swapped.views.reverse();
    Prepared::new(&swapped, &c, &q).unwrap().feed(&mut inf);
    inf.step();
    inf.wait();
    let mut actual = vec![0.0; 6];
    inf.read_output_by_index(1, &mut actual);
    assert!(rgb.iter().zip(actual).all(|(a, b)| (a - b).abs() < 1e-4));
    std::fs::remove_file(path).unwrap();
    println!("stereo loss {first} -> {}", train.read_loss());
}
