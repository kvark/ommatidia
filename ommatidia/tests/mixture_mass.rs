use meganeura::graph::Op;
use ommatidia::transport::{self, cpu, graph};
include!("fixtures/transport.rs");

#[test]
#[ignore = "requires Vulkan or Metal"]
fn extreme_logits_preserve_candidate_mass_and_cpu_reconstruction() {
    let c = Config::default();
    let (frame, _) = fixture(c, 0, 19);
    let p = cpu::prepare(&frame, &[], c);
    let n = frame.surfaces.len();
    let mut model = graph::build(c, frame.low, 0).unwrap();
    let head = model
        .graph
        .nodes()
        .iter()
        .find(|node| matches!(&node.op, Op::Parameter { name } if name == "head.lobe_candidates"))
        .unwrap()
        .id;
    let id = model
        .graph
        .nodes()
        .iter()
        .find(|node| node.inputs.get(1) == Some(&head) && matches!(node.op, Op::Conv2d { .. }))
        .unwrap()
        .id;
    let node = &mut model.graph.nodes_mut()[id as usize];
    node.op = Op::Parameter {
        name: "probe.logits".into(),
    };
    node.inputs.clear();
    let multiplier = model
        .graph
        .nodes()
        .iter()
        .find(|node| node.inputs == [id] && matches!(node.op, Op::Softplus { .. }))
        .unwrap()
        .id;
    let mut outputs = model.graph.outputs().to_vec();
    outputs.push(multiplier);
    model.graph.set_outputs(outputs);
    let mut session = ommatidia::gpu::inference_session(
        &model.graph,
        ommatidia::gpu::create_context(None, false),
    );
    for (name, values) in [
        ("f0.features", &p.features),
        ("f0.candidates", &p.candidates),
        ("f0.prior", &p.prior),
        ("f0.history", &p.history),
    ] {
        if session.has_input(name) {
            session.set_input(name, values);
        }
    }
    for offset in [
        -1000.0f32, -30.0, -20.0, -18.5, -17.0, -16.0, 0.0, 30.0, 1000.0,
    ] {
        let logits: Vec<_> = (0..p.prior.len())
            .map(|i| offset + (i % 11) as f32 * 0.1)
            .collect();
        session.set_parameter("probe.logits", &logits);
        session.step();
        session.wait();
        let mut weights = vec![0.0; p.prior.len()];
        let mut multipliers = weights.clone();
        session.read_output_by_index(1, &mut weights);
        session.read_output_by_index(2, &mut multipliers);
        for i in 0..2 * n {
            let total: f32 = (0..transport::CANDIDATES)
                .map(|k| weights[k * 2 * n + i])
                .sum();
            assert!(
                (total - 1.0).abs() < 2e-6,
                "lost candidate mass at logit offset {offset}: {total}"
            );
        }
        // An independent mathematical reference, not the GPU multipliers:
        // forwarding those to the scalar model hid negative-tail cancellation.
        let reference: Vec<_> = logits
            .iter()
            .map(|&x| {
                let x = x as f64;
                (x.max(0.0) + (-x.abs()).exp().ln_1p()) as f32
            })
            .collect();
        for (&actual, &truth) in multipliers.iter().zip(&reference) {
            if truth >= transport::MIN_MULTIPLIER {
                assert!(
                    (actual - truth).abs() <= 1e-5 * truth,
                    "softplus negative-tail mismatch at offset {offset}: {actual} != {truth}"
                );
            }
        }
        let (expected, expected_weights) = cpu::reconstruct(&p, &reference);
        let mut image = vec![0.0; expected.len()];
        session.read_output_by_index(0, &mut image);
        assert!(
            image
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() < 2e-5 * (1.0 + b.abs()))
        );
        assert!(
            weights
                .iter()
                .zip(expected_weights)
                .all(|(a, b)| (a - b).abs() < 2e-6)
        );
        println!("normalized extreme selector offset {offset}");
    }
}
