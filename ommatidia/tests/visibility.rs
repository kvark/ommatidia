use ommatidia::field::{
    self,
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
fn observations(c: &Config) -> Observations {
    Observations {
        bounds: Bounds {
            center: [0.0; 3],
            radius: 1.0,
        },
        views: (0..c.views)
            .map(|i| View {
                camera: Camera {
                    origin: [i as f32 * 0.1, 0.0, 3.0],
                    right: [1.0, 0.0, 0.0],
                    up: [0.0, 1.0, 0.0],
                    forward: [0.0, 0.0, -1.0],
                    tan_half_fov_y: 0.5,
                },
                rgb: vec![if i == 0 { 2.0 } else { 6.0 }; 8 * 8 * 3],
            })
            .collect(),
    }
}
fn captures(c: &Config) -> Vec<Option<surface::Capture>> {
    (0..c.views)
        .map(|_| {
            Some(surface::Capture {
                version: 1,
                pixel_filter: "center".into(),
                ray_limit: 20.0,
                distance: vec![None; 64],
                emission: vec![[0.0; 3]; 64],
            })
        })
        .collect()
}
#[test]
fn source_labels_are_target_only_and_mask_uncertified_space() {
    let c = config();
    let obs = observations(&c);
    let q = [Query {
        position: [0.0; 3],
        direction: [0.0, 0.0, -1.0],
    }];
    let before = Prepared::new(&obs, &c, &q).unwrap();
    let mut labels = captures(&c);
    let t = visibility::Targets::new(&obs, &c, &labels, 0.1).unwrap();
    assert!(t.valid > 0);
    assert!((t.mass.iter().flatten().sum::<f32>() - 0.1).abs() < 1e-5);
    for v in labels.iter_mut().flatten() {
        v.distance.fill(Some(1.0));
    }
    let masked = visibility::Targets::new(&obs, &c, &labels, 0.1).unwrap();
    assert_eq!(masked.valid, 0);
    assert_eq!(before, Prepared::new(&obs, &c, &q).unwrap());
    assert!(visibility::Targets::new(&obs, &c, &[None, None], 0.0).is_err());
    let m = graph::build_points(&c, 1).unwrap();
    assert!(m.graph.nodes().iter().all(
        |n| !matches!(&n.op, meganeura::graph::Op::Input {name} if name.starts_with("target."))
    ));
}
#[test]
#[ignore = "requires Vulkan or Metal"]
fn predicted_occlusion_closes_copy_and_preserves_direction_isolation() {
    let mut c = config();
    c.views = 1;
    let obs = observations(&c);
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
    let m = graph::build_points(&c, q.len()).unwrap();
    let mut s =
        ommatidia::gpu::inference_session(&m.graph, ommatidia::gpu::create_context(None, false));
    m.initialize(&mut s, 7);
    s.set_parameter("field.source.gate.bias", &[20.0]);
    s.set_parameter("field.emission.weight", &vec![0.0; c.hidden as usize * 3]);
    s.set_parameter("field.emission.bias", &[-20.0; 3]);
    s.set_parameter(
        "field.appearance.out.weight",
        &vec![0.0; c.hidden as usize * 3],
    );
    s.set_parameter("field.appearance.out.bias", &[(0.2f32.exp() - 1.0).ln(); 3]);
    let mut bias = vec![-30.0; visibility::BINS + 1];
    bias[0] = 30.0;
    s.set_parameter("field.visibility.bias", &bias);
    Prepared::new(&obs, &c, &q).unwrap().feed(&mut s);
    s.step();
    s.wait();
    let mut rgb = vec![0.0; 6];
    s.read_output_by_index(1, &mut rgb);
    assert!(
        rgb.iter().all(|v| (*v - 0.2).abs() < 1e-4),
        "occluded source regained authority: {rgb:?}"
    );
    let density = s.read_output(2);
    assert!((density[0] - density[1]).abs() < 1e-6);
    bias[0] = -30.0;
    bias[visibility::BINS] = 30.0;
    s.set_parameter("field.visibility.bias", &bias);
    s.step();
    s.wait();
    s.read_output_by_index(1, &mut rgb);
    assert!(
        rgb.iter().all(|v| (*v - 2.0).abs() < 1e-4),
        "clear source lost radiance: {rgb:?}"
    );
}
#[test]
#[ignore = "requires Vulkan or Metal"]
fn visibility_trains_reloads_and_is_view_permutation_invariant() {
    let c = config();
    let obs = observations(&c);
    let context = ommatidia::gpu::create_context(None, false);
    let shape = RenderShape {
        rays: 2,
        steps: 4,
        probes: 0,
    };
    let m = graph::build_render(&c, shape, true).unwrap();
    let mut s = ommatidia::gpu::training_session(&m.graph, Arc::clone(&context));
    m.initialize(&mut s, 7);
    let rays = [
        obs.views[0].camera.ray([3.0, 3.0], c.extent),
        obs.views[1].camera.ray([4.0, 4.0], c.extent),
    ];
    let (q, dt) = field::data::ray_queries(obs.bounds, &rays, shape.steps).unwrap();
    Prepared::new(&obs, &c, &q).unwrap().feed(&mut s);
    s.set_input("ray.deltas", &dt);
    Targets {
        rgb: vec![1.0; 6],
        emission: vec![0.0; q.len() * 3],
        emission_mask: vec![0.0; q.len() * 3],
        environment: [0.1; 3],
        environment_mask: [1.0; 3],
    }
    .feed(&mut s);
    visibility::Targets::new(&obs, &c, &captures(&c), 0.1)
        .unwrap()
        .feed(&mut s);
    let mut first = 0.0;
    let mut last = 0.0;
    for step in 0..32 {
        s.set_adam(0.001, 0.9, 0.999, 1e-8);
        s.step();
        s.wait();
        last = s.read_loss();
        if step == 0 {
            first = last;
        }
        assert!(last.is_finite());
    }
    assert!(last < first, "{first} -> {last}");
    let mut weight = vec![0.0; c.channels as usize * (visibility::BINS + 1)];
    s.read_param("field.visibility.weight", &mut weight);
    assert!(weight.iter().any(|v| v.abs() > 1e-6));
    let path = std::env::temp_dir().join(format!("visible-rgb-{}.safetensors", std::process::id()));
    s.save_checkpoint(&path).unwrap();
    let points = graph::build_points(&c, q.len()).unwrap();
    let mut inf = ommatidia::gpu::inference_session(&points.graph, context);
    inf.load_checkpoint(&path).unwrap();
    Prepared::new(&obs, &c, &q).unwrap().feed(&mut inf);
    inf.step();
    inf.wait();
    let mut rgb = vec![0.0; q.len() * 3];
    inf.read_output_by_index(1, &mut rgb);
    let mut swap = obs.clone();
    swap.views.reverse();
    Prepared::new(&swap, &c, &q).unwrap().feed(&mut inf);
    inf.step();
    inf.wait();
    let mut other = rgb.clone();
    inf.read_output_by_index(1, &mut other);
    assert!(rgb.iter().zip(other).all(|(a, b)| (a - b).abs() < 1e-4));
    std::fs::remove_file(path).unwrap();
    println!("visibility loss {first} -> {last}");
}

#[test]
#[ignore = "requires Vulkan or Metal"]
fn source_distributions_normalize_and_common_initialization_matches() {
    let c = config();
    let obs = observations(&c);
    let queries = [Query {
        position: [0.0; 3],
        direction: [0.0, 0.0, -1.0],
    }];
    let context = ommatidia::gpu::create_context(None, false);
    let model = graph::build_points(&c, queries.len()).unwrap();
    let control = graph::build_points(
        &Config {
            view_fusion: ViewFusion::LateRgb,
            ..c.clone()
        },
        queries.len(),
    )
    .unwrap();
    let mut session = ommatidia::gpu::inference_session(&model.graph, Arc::clone(&context));
    let mut other = ommatidia::gpu::inference_session(&control.graph, context);
    model.initialize(&mut session, 11);
    control.initialize(&mut other, 11);
    for p in &control.params {
        let mut actual = vec![0.0; p.len];
        let mut expected = vec![0.0; p.len];
        session.read_param(&p.name, &mut actual);
        other.read_param(&p.name, &mut expected);
        assert_eq!(
            actual, expected,
            "shared initialization changed: {}",
            p.name
        );
    }
    Prepared::new(&obs, &c, &queries)
        .unwrap()
        .feed(&mut session);
    session.step();
    session.wait();
    let bins = visibility::BINS + 1;
    for view in 0..c.views {
        let mut mass = vec![0.0; (c.extent[0] * c.extent[1]) as usize * bins];
        session.read_output_by_index(4 + view, &mut mass);
        assert!(mass.iter().all(|v| (*v - 1.0 / bins as f32).abs() < 1e-6));
        assert!(
            mass.chunks_exact(bins)
                .all(|p| (p.iter().sum::<f32>() - 1.0).abs() < 1e-6)
        );
    }
}
