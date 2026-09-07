use ommatidia::field::{
    data::{self, Prepared, RenderShape, Targets},
    graph, *,
};
use std::sync::Arc;
fn config() -> Config {
    Config {
        extent: [8, 8],
        views: 2,
        channels: 4,
        hidden: 16,
        position_frequencies: 1,
        ..Config::default()
    }
}
fn camera(x: f32) -> Camera {
    Camera {
        origin: [x, 0.0, 3.0],
        right: [1.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, -1.0],
        tan_half_fov_y: 0.5,
    }
}
fn observations(c: &Config) -> Observations {
    Observations {
        bounds: Bounds {
            center: [0.0; 3],
            radius: 1.0,
        },
        views: (0..c.views)
            .map(|v| View {
                camera: camera(v as f32 * 0.2),
                rgb: (0..c.extent[0] * c.extent[1] * 3)
                    .map(|i| 0.1 + (i % 7) as f32 * 0.07)
                    .collect(),
            })
            .collect(),
    }
}
#[test]
fn projection_roundtrip_and_missing_evidence() {
    let c = config();
    let camera = camera(0.0);
    camera.validate().unwrap();
    for pixel in [[0.0, 0.0], [3.5, 3.5], [7.0, 7.0]] {
        let ray = camera.ray(pixel, c.extent);
        let x = std::array::from_fn(|i| ray.origin[i] + 2.5 * ray.direction[i]);
        let back = camera.project(x, c.extent).unwrap();
        assert!((back[0] - pixel[0]).abs() < 1e-5 && (back[1] - pixel[1]).abs() < 1e-5);
    }
    let q = [Query {
        position: [0.0, 0.0, 4.0],
        direction: [0.0, 0.0, -1.0],
    }];
    let inputs = Prepared::new(&observations(&c), &c, &q).unwrap();
    assert_eq!(inputs.coverage, [0.0]);
    assert_eq!(inputs.inverse_count, [1.0]);
    assert!(inputs.weights.iter().flatten().flatten().all(|v| *v == 0.0));
    let mut bad = c.clone();
    bad.extent = [0, 8];
    assert!(bad.validate().is_err());
    assert!(
        Prepared::new(
            &observations(&c),
            &c,
            &[Query {
                position: [0.0; 3],
                direction: [0.0; 3]
            }]
        )
        .is_err()
    );
}
#[test]
fn task_heads_share_the_actual_core_and_exclude_truth_inputs() {
    let c = config();
    let m = graph::build_points(&c, 2).unwrap();
    let transport = ommatidia::transport::graph::build(
        ommatidia::transport::Config {
            channels: c.channels,
            ..Default::default()
        },
        c.extent,
        0,
    )
    .unwrap();
    let core = |model: &graph::Network| {
        model
            .params
            .iter()
            .filter(|p| p.name.starts_with("core."))
            .map(|p| (p.name.clone(), p.len))
            .collect::<Vec<_>>()
    };
    assert_eq!(core(&m), core(&transport));
    assert!(!core(&m).is_empty());
    for n in m.graph.nodes() {
        if let meganeura::graph::Op::Input { name } = &n.op {
            assert!(
                !name.starts_with("target.")
                    && !name.contains("gbuffer")
                    && !name.contains("velocity")
                    && !name.contains("lights")
            );
        }
    }
    let mut more = c.clone();
    more.views = 3;
    let other = graph::build_points(&more, 2).unwrap();
    assert_eq!(
        m.params.len(),
        other.params.len(),
        "view encoder weights must be tied"
    );
}
#[test]
#[ignore = "requires Vulkan or Metal"]
fn field_direction_isolation_pooling_and_render_parity() {
    let c = config();
    let obs = observations(&c);
    let model = graph::build_points(&c, 2).unwrap();
    let context = ommatidia::gpu::create_context(None, false);
    let mut session = ommatidia::gpu::inference_session(&model.graph, Arc::clone(&context));
    model.initialize(&mut session, 4);
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
    Prepared::new(&obs, &c, &q).unwrap().feed(&mut session);
    session.step();
    session.wait();
    let density = session.read_output(2);
    assert!(
        (density[0] - density[1]).abs() < 1e-6,
        "view-dependent geometry"
    );
    let mut original = vec![0.0; 6];
    session.read_output_by_index(1, &mut original);
    let mut swapped = obs.clone();
    swapped.views.reverse();
    Prepared::new(&swapped, &c, &q).unwrap().feed(&mut session);
    session.step();
    session.wait();
    let mut actual = vec![0.0; 6];
    session.read_output_by_index(1, &mut actual);
    assert!(
        original
            .iter()
            .zip(&actual)
            .all(|(a, b)| (a - b).abs() < 1e-5),
        "view order changes the field"
    );
    let shape = RenderShape {
        rays: 2,
        steps: 4,
        probes: 0,
    };
    let m = graph::build_render(&c, shape, false).unwrap();
    let mut s = ommatidia::gpu::inference_session(&m.graph, context);
    m.initialize(&mut s, 4);
    let rays = [
        camera(0.0).ray([3.5, 3.5], c.extent),
        Ray {
            origin: [100.0, 100.0, 3.0],
            direction: [0.0, 0.0, -1.0],
        },
    ];
    let (q, dt) = data::ray_queries(obs.bounds, &rays, shape.steps).unwrap();
    Prepared::new(&obs, &c, &q).unwrap().feed(&mut s);
    s.set_input("ray.deltas", &dt);
    s.step();
    s.wait();
    let mut density = vec![0.0; 8];
    let mut rgb = vec![0.0; 24];
    let mut env = vec![0.0; 3];
    s.read_output_by_index(1, &mut density);
    s.read_output_by_index(2, &mut rgb);
    s.read_output_by_index(4, &mut env);
    let mut expected = [0.0; 6];
    for r in 0..2 {
        let mut t = 1.0;
        for k in 0..4 {
            let i = k * 2 + r;
            let keep = (-density[i] * dt[i]).exp();
            for c in 0..3 {
                expected[r * 3 + c] += t * (1.0 - keep) * rgb[i * 3 + c];
            }
            t *= keep;
        }
        for c in 0..3 {
            expected[r * 3 + c] += t * env[c];
        }
    }
    let actual = s.read_output(6);
    assert!(
        expected
            .iter()
            .zip(&actual)
            .all(|(a, b)| (a - b).abs() < 1e-5)
    );
    println!("field volume integration matches CPU; miss ray uses predicted environment");
}
#[test]
#[ignore = "requires Vulkan or Metal"]
fn field_training_reaches_encoder_and_light_heads_and_reloads() {
    let c = config();
    let obs = observations(&c);
    let shape = RenderShape {
        rays: 2,
        steps: 4,
        probes: 2,
    };
    let m = graph::build_render(&c, shape, true).unwrap();
    let context = ommatidia::gpu::create_context(None, false);
    let mut s = ommatidia::gpu::training_session(&m.graph, Arc::clone(&context));
    m.initialize(&mut s, 9);
    let light = Lighting {
        environment: [0.1; 3],
        emitters: Vec::new(),
        probes: vec![
            EmissionProbe {
                position: [0.0; 3],
                radiance: [2.0, 1.0, 0.5],
            },
            EmissionProbe {
                position: [0.5, 0.0, 0.0],
                radiance: [0.0; 3],
            },
        ],
    };
    let rays = [
        camera(0.0).ray([3.0, 3.0], c.extent),
        camera(0.0).ray([4.0, 4.0], c.extent),
    ];
    let (mut q, dt) = data::ray_queries(obs.bounds, &rays, shape.steps).unwrap();
    let (emission, emission_mask) = data::append_probes(&mut q, &light, 2, 1);
    let inputs = Prepared::new(&obs, &c, &q).unwrap();
    let mut targets = Targets {
        rgb: vec![0.2; 6],
        emission,
        emission_mask,
        environment: [0.1; 3],
        environment_mask: [1.0; 3],
    };
    let snapshot = inputs.clone();
    targets.environment = [0.2; 3];
    assert_eq!(
        snapshot,
        Prepared::new(&obs, &c, &q).unwrap(),
        "labels leaked into features"
    );
    inputs.feed(&mut s);
    s.set_input("ray.deltas", &dt);
    targets.feed(&mut s);
    s.set_adam(0.003, 0.9, 0.999, 1e-8);
    let tracked = [
        "core.level0.a",
        "field.emission.weight",
        "field.density.weight",
    ];
    let before = s.read_params(&tracked);
    let mut first = 0.0;
    let mut last = 0.0;
    for i in 0..24 {
        s.step();
        s.wait();
        last = s.read_loss();
        assert!(last.is_finite());
        if i == 0 {
            first = last;
        }
    }
    assert!(
        last < first,
        "field did not fit synthetic batch: {first}->{last}"
    );
    let after = s.read_params(&tracked);
    for (a, b) in before.iter().zip(&after) {
        assert!(b.iter().all(|v| v.is_finite()));
        assert!(
            a.iter().zip(b).any(|(a, b)| (a - b).abs() > 1e-7),
            "missing gradient to a field branch"
        );
    }
    let path = std::env::temp_dir().join(format!("field-{}.safetensors", std::process::id()));
    s.save_checkpoint(&path).unwrap();
    let inf = graph::build_render(&c, shape, false).unwrap();
    let mut inference = ommatidia::gpu::inference_session(&inf.graph, context);
    inference.load_checkpoint(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    inputs.feed(&mut inference);
    inference.set_input("ray.deltas", &dt);
    inference.step();
    inference.wait();
    assert!(inference.read_output(6).iter().all(|v| v.is_finite()));
    println!(
        "field synthetic loss {first:.6}->{last:.6}; encoder, density and emission weights updated"
    );
}
