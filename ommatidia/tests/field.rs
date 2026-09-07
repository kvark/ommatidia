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

#[test]
fn incident_manifest_versions_and_runtime_isolation() {
    let record = SceneRecord {
        scene_seed: 7,
        lighting_seed: None,
        bounds: Bounds {
            center: [0.0; 3],
            radius: 1.0,
        },
        lighting: Lighting {
            environment: [1.0; 3],
            emitters: vec![],
            probes: vec![],
        },
        incident: None,
    };
    let mut manifest = Manifest {
        version: 1,
        rgb_space: "scene-linear-renderer-units".into(),
        static_scene: true,
        extent: [8, 8],
        scenes: vec![record],
        records: vec![],
    };
    manifest.validate(0).unwrap();
    let old = ron::to_string(&manifest).unwrap();
    assert!(!old.contains("incident"));
    let mut old: Manifest = ron::from_str(&old).unwrap();
    old.validate(0).unwrap();
    let probe = incident::Probe {
        origin: [0.0; 3],
        direction: [0.0, 1.0, 0.0],
        proposal: incident::Proposal::UniformHemisphere,
        direct: incident::Estimate {
            mean: [1.0; 3],
            variance_of_mean: [0.0; 3],
        },
        indirect: incident::Estimate {
            mean: [0.0; 3],
            variance_of_mean: [0.0; 3],
        },
        total_variance_of_mean: [0.0; 3],
    };
    old.scenes[0].incident = Some(incident::Capture {
        version: 1,
        integrator: "blade-canonical-point-ray-v1".into(),
        max_bounces: 8,
        batches: 4,
        paths_per_batch: 4,
        ray_epsilon: 0.01,
        ray_distance: 200.0,
        radiance_ceiling: 1e6,
        probes: vec![probe],
    });
    assert!(old.validate(0).is_err());
    old.version = 2;
    old.validate(0).unwrap();
    old.scenes[0].incident.as_mut().unwrap().batches = 1;
    assert!(old.validate(0).is_err());
    manifest.version = 4;
    assert!(manifest.validate(0).is_err());
    let obs = ron::to_string(&observations(&config())).unwrap();
    assert!(!obs.contains("incident"));
    let shape = RenderShape {
        rays: 2,
        steps: 4,
        probes: 0,
    };
    let inf = graph::build_incident(&config(), shape).unwrap();
    let old = graph::build_render(&config(), shape, false).unwrap();
    assert_eq!(
        inf.params
            .iter()
            .map(|p| (&p.name, p.len))
            .collect::<Vec<_>>(),
        old.params
            .iter()
            .map(|p| (&p.name, p.len))
            .collect::<Vec<_>>()
    );
    for n in inf.graph.nodes() {
        if let meganeura::graph::Op::Input { name } = &n.op {
            assert!(!name.starts_with("target."));
        }
    }
    assert!(graph::build_training(&config(), shape, 2).is_err());
}

#[test]
#[ignore = "requires Vulkan or Metal"]
fn incident_integral_is_the_same_field_and_preserves_the_split() {
    let c = config();
    let obs = observations(&c);
    let shape = RenderShape {
        rays: 2,
        steps: 4,
        probes: 0,
    };
    let context = ommatidia::gpu::create_context(None, false);
    let image = graph::build_render(&c, shape, false).unwrap();
    let incoming = graph::build_incident(&c, shape).unwrap();
    let mut a = ommatidia::gpu::inference_session(&image.graph, Arc::clone(&context));
    let mut b = ommatidia::gpu::inference_session(&incoming.graph, context);
    image.initialize(&mut a, 4);
    incoming.initialize(&mut b, 4);
    let rays = [
        Ray {
            origin: [0.0; 3],
            direction: [1.0, 0.0, 0.0],
        },
        Ray {
            origin: [10.0; 3],
            direction: [0.0, 1.0, 0.0],
        },
    ];
    let (q, dt) = data::ray_queries(obs.bounds, &rays, shape.steps).unwrap();
    let prepared = Prepared::new(&obs, &c, &q).unwrap();
    for s in [&mut a, &mut b] {
        prepared.feed(s);
        s.set_input("ray.deltas", &dt);
        s.step();
        s.wait();
    }
    let total = b.read_output(6);
    let mut direct = vec![0.0; 6];
    let mut indirect = direct.clone();
    b.read_output_by_index(1, &mut direct);
    b.read_output_by_index(2, &mut indirect);
    let mut density = vec![0.0; 8];
    let mut emission = vec![0.0; 24];
    let mut env = [0.0; 3];
    a.read_output_by_index(1, &mut density);
    a.read_output_by_index(3, &mut emission);
    a.read_output_by_index(4, &mut env);
    for r in 0..2 {
        let mut trans = 1.0;
        let mut expected = [0.0; 3];
        for k in 0..4 {
            let i = 2 * k + r;
            let keep = (-density[i] * dt[i]).exp();
            for (c, value) in expected.iter_mut().enumerate() {
                *value += trans * (1.0 - keep) * emission[3 * i + c];
            }
            trans *= keep;
        }
        for (c, value) in expected.iter_mut().enumerate() {
            *value += trans * env[c];
            assert!((*value - direct[3 * r + c]).abs() < 1e-5);
        }
    }
    assert!(
        total
            .iter()
            .zip(a.read_output(6))
            .all(|(x, y)| (x - y).abs() < 1e-6)
    );
    assert!(
        total
            .iter()
            .zip(direct.iter().zip(&indirect))
            .all(|(t, (d, i))| (*t - d - i).abs() < 1e-6)
    );
    assert!(
        direct
            .iter()
            .chain(&indirect)
            .all(|v| v.is_finite() && *v >= 0.0)
    );
    assert_eq!(&indirect[3..], &[0.0; 3]);
    println!(
        "incident field: split sums to camera radiance; CPU visibility/emission integral matches"
    );
}

#[test]
#[ignore = "requires Vulkan or Metal"]
fn incident_supervision_backpropagates_through_visibility_and_reloads() {
    let c = config();
    let obs = observations(&c);
    let shape = RenderShape {
        rays: 2,
        steps: 4,
        probes: 0,
    };
    let context = ommatidia::gpu::create_context(None, false);
    let m = graph::build_training(&c, shape, 1).unwrap();
    let inf = graph::build_render(&c, shape, false).unwrap();
    let mut s = ommatidia::gpu::training_session(&m.graph, Arc::clone(&context));
    let mut baseline = ommatidia::gpu::inference_session(&inf.graph, Arc::clone(&context));
    m.initialize(&mut s, 9);
    inf.initialize(&mut baseline, 9);
    let rays = [
        camera(0.0).ray([3.0, 3.0], c.extent),
        Ray {
            origin: [0.0; 3],
            direction: [1.0, 0.0, 0.0],
        },
    ];
    let (q, dt) = data::ray_queries(obs.bounds, &rays, shape.steps).unwrap();
    let inputs = Prepared::new(&obs, &c, &q).unwrap();
    inputs.feed(&mut baseline);
    baseline.set_input("ray.deltas", &dt);
    baseline.step();
    baseline.wait();
    let pixels = baseline.read_output(6);
    let targets = Targets {
        rgb: pixels[..3].to_vec(),
        emission: vec![0.0; 24],
        emission_mask: vec![0.0; 24],
        environment: [0.0; 3],
        environment_mask: [0.0; 3],
    };
    let mut p = incident::Probe {
        origin: [0.0; 3],
        direction: [1.0, 0.0, 0.0],
        proposal: incident::Proposal::UniformHemisphere,
        direct: incident::Estimate {
            mean: [0.05; 3],
            variance_of_mean: [0.0; 3],
        },
        indirect: incident::Estimate {
            mean: [0.6; 3],
            variance_of_mean: [0.01; 3],
        },
        total_variance_of_mean: [0.01; 3],
    };
    let snapshot = inputs.clone();
    p.indirect.mean = [0.7; 3];
    assert_eq!(snapshot, Prepared::new(&obs, &c, &q).unwrap());
    inputs.feed(&mut s);
    s.set_input("ray.deltas", &dt);
    targets.feed(&mut s);
    incident::Targets::new(&[&p], c.exposure)
        .unwrap()
        .weighted(1.0)
        .unwrap()
        .feed(&mut s);
    s.set_adam(0.003, 0.9, 0.999, 1e-8);
    let tracked = [
        "core.level0.a",
        "field.density.weight",
        "field.emission.weight",
        "field.appearance.out.weight",
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
    assert!(last < first, "incident batch loss: {first}->{last}");
    for (a, b) in before.iter().zip(s.read_params(&tracked)) {
        assert!(b.iter().all(|v| v.is_finite()));
        assert!(a.iter().zip(b).any(|(a, b)| (a - b).abs() > 1e-7));
    }
    let path = std::env::temp_dir().join(format!("incident-{}.safetensors", std::process::id()));
    s.save_checkpoint(&path).unwrap();
    let model = graph::build_incident(&c, shape).unwrap();
    let mut loaded = ommatidia::gpu::inference_session(&model.graph, context);
    loaded.load_checkpoint(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    inputs.feed(&mut loaded);
    loaded.set_input("ray.deltas", &dt);
    loaded.step();
    loaded.wait();
    assert!(loaded.read_output(6).iter().all(|v| v.is_finite()));
    println!(
        "incident loss {first}->{last}; gradients reach core, density, emission and scattered radiance; reload passes"
    );
}

#[test]
fn surface_manifest_and_query_contract() {
    let c = config();
    let labels = surface::Capture {
        version: 1,
        pixel_filter: "center".into(),
        ray_limit: 200.0,
        distance: vec![Some(3.0); 64],
        emission: vec![[0.0; 3]; 64],
    };
    labels.validate(c.extent).unwrap();
    let mut m = Manifest {
        version: 2,
        rgb_space: "scene-linear-renderer-units".into(),
        static_scene: true,
        extent: c.extent,
        scenes: vec![SceneRecord {
            scene_seed: 1,
            lighting_seed: None,
            bounds: observations(&c).bounds,
            lighting: Lighting {
                environment: [1.0; 3],
                emitters: vec![],
                probes: vec![],
            },
            incident: None,
        }],
        records: vec![ViewRecord {
            sample: 0,
            scene: 0,
            camera: camera(0.0),
            surface: Some(labels),
        }],
    };
    assert!(m.validate(1).is_err());
    m.version = 3;
    m.validate(1).unwrap();
    let decoded: Manifest = ron::from_str(&ron::to_string(&m).unwrap()).unwrap();
    decoded.validate(1).unwrap();
    m.records[0].surface.as_mut().unwrap().distance[0] = None;
    m.records[0].surface.as_mut().unwrap().emission[0] = [1.0; 3];
    assert!(m.validate(1).is_err());
    let shape = RenderShape {
        rays: 2,
        steps: 8,
        probes: 0,
    };
    let a = graph::build_training(&c, shape, 0).unwrap();
    let b = graph::build_surface_training(&c, shape, 0).unwrap();
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
    for model in [
        graph::build_diagnostics(&c, shape).unwrap(),
        graph::build_points(&c, 2).unwrap(),
    ] {
        for n in model.graph.nodes() {
            if let meganeura::graph::Op::Input { name } = &n.op {
                assert!(!name.starts_with("target."));
            }
        }
    }
}

#[test]
#[ignore = "requires Vulkan or Metal"]
fn surface_termination_supervision_learns_and_preserves_inference() {
    let c = config();
    let obs = observations(&c);
    let shape = RenderShape {
        rays: 2,
        steps: 8,
        probes: 0,
    };
    let rays = [
        camera(0.0).ray([3.0, 3.0], c.extent),
        camera(0.0).ray([4.0, 3.0], c.extent),
    ];
    let (q, dt) = data::ray_queries(obs.bounds, &rays, shape.steps).unwrap();
    let inputs = Prepared::new(&obs, &c, &q).unwrap();
    let context = ommatidia::gpu::create_context(None, false);
    let inf = graph::build_diagnostics(&c, shape).unwrap();
    let net = graph::build_surface_training(&c, shape, 0).unwrap();
    let mut baseline = ommatidia::gpu::inference_session(&inf.graph, Arc::clone(&context));
    inf.initialize(&mut baseline, 7);
    inputs.feed(&mut baseline);
    baseline.set_input("ray.deltas", &dt);
    baseline.step();
    baseline.wait();
    let rgb = baseline.read_output(6);
    let mut before_mass = vec![0.0; 18];
    baseline.read_output_by_index(1, &mut before_mass);
    for r in 0..2 {
        assert!(((0..9).map(|s| before_mass[2 * s + r]).sum::<f32>() - 1.0).abs() < 1e-5);
    }
    let mut train = ommatidia::gpu::training_session(&net.graph, Arc::clone(&context));
    net.initialize(&mut train, 7);
    inputs.feed(&mut train);
    train.set_input("ray.deltas", &dt);
    Targets {
        rgb,
        emission: vec![0.0; 48],
        emission_mask: vec![0.0; 48],
        environment: [0.0; 3],
        environment_mask: [0.0; 3],
    }
    .feed(&mut train);
    // Both rays terminate near the same front surface. No density target is assigned behind it.
    let labels = [Some((Some(2.6), 200.0)); 2];
    let target = surface::Targets::new(obs.bounds, &rays, 8, &labels, 1.0).unwrap();
    assert_eq!(target.valid, 2);
    target.feed(&mut train);
    let tracked = ["core.level0.a", "field.density.weight"];
    let before = train.read_params(&tracked);
    train.set_adam(0.003, 0.9, 0.999, 1e-8);
    let mut first = 0.0;
    let mut last = 0.0;
    for k in 0..48 {
        train.step();
        train.wait();
        last = train.read_loss();
        assert!(last.is_finite());
        if k == 0 {
            first = last;
        }
    }
    assert!(last < 0.75 * first, "termination loss {first}->{last}");
    for (a, b) in before.iter().zip(train.read_params(&tracked)) {
        assert!(b.iter().all(|v| v.is_finite()));
        assert!(a.iter().zip(b).any(|(a, b)| (a - b).abs() > 1e-7));
    }
    let path = std::env::temp_dir().join(format!("surface-{}.safetensors", std::process::id()));
    train.save_checkpoint(&path).unwrap();
    baseline.load_checkpoint(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    inputs.feed(&mut baseline);
    baseline.set_input("ray.deltas", &dt);
    baseline.step();
    baseline.wait();
    let mut after = vec![0.0; 18];
    baseline.read_output_by_index(1, &mut after);
    let nll = |w: &[f32]| {
        -w.iter()
            .zip(&target.mass)
            .map(|(p, t)| t * (p + 1e-8).ln())
            .sum::<f32>()
    };
    assert!(nll(&after) < nll(&before_mass));
    println!(
        "termination NLL {} -> {}; training loss {first}->{last}",
        nll(&before_mass),
        nll(&after)
    );
}
