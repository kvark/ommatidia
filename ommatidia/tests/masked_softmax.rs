use meganeura::Graph;
use ommatidia::transport::{graph, mixture, native};
use std::sync::Arc;
include!("fixtures/transport.rs");

fn reference(z: &[f32], p: &[f32], stride: usize) -> Vec<f32> {
    let mut out = vec![0.0; z.len()];
    for i in 0..stride {
        let center = (0..6)
            .filter(|&k| p[k * stride + i] > 0.0)
            .map(|k| z[k * stride + i] as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        let mut weights = [0.0; 6];
        for k in 0..6 {
            if p[k * stride + i] > 0.0 {
                weights[k] = p[k * stride + i] as f64
                    * ((z[k * stride + i] as f64 - center) * mixture::SOFTMAX_GAIN as f64).exp();
                total += weights[k];
            }
        }
        for k in 0..6 {
            out[k * stride + i] = (weights[k] / total) as f32;
        }
    }
    out
}
#[test]
#[ignore = "requires Vulkan or Metal"]
fn masked_softmax_preserves_prior_mask_shift_and_mathematical_weights() {
    let mut g = Graph::new();
    let z = g.parameter("logits", &[12]);
    let p = g.input("prior", &[12]);
    let w = mixture::build(&mut g, z, p, 2, 1);
    g.set_outputs(vec![w]);
    let mut s = ommatidia::gpu::inference_session(&g, ommatidia::gpu::create_context(None, false));
    let prior = [
        0.02, 0.0, 0.06, 0.2, 0.12, 0.0, 0.3, 0.3, 0.5, 0.0, 0.0, 0.5,
    ];
    s.set_input("prior", &prior);
    let mut actual = vec![0.0; 12];
    let base = [
        0.0, -0.5, 0.5, 0.75, -0.25, -1.0, 1.5, 0.0, -1.0, 1.0, 1e6, 0.5,
    ];
    let expected = reference(&base, &prior, 2);
    for offset in [-1e6f32, -1000.0, 0.0, 1000.0, 1e6] {
        let logits: Vec<_> = base.iter().map(|v| v + offset).collect();
        s.set_parameter("logits", &logits);
        s.step();
        s.wait();
        s.read_output_by_index(0, &mut actual);
        for ((&v, &truth), &p) in actual.iter().zip(&expected).zip(&prior) {
            assert!(
                v.is_finite() && (v - truth).abs() < 2e-6,
                "{offset}: {v} != {truth}"
            );
            if p == 0.0 {
                assert_eq!(v, 0.0);
            }
        }
        for i in 0..2 {
            assert!(((0..6).map(|k| actual[k * 2 + i]).sum::<f32>() - 1.0).abs() < 2e-6);
        }
    }
    s.set_parameter("logits", &[0.0; 12]);
    s.step();
    s.wait();
    s.read_output_by_index(0, &mut actual);
    assert!(actual.iter().zip(prior).all(|(a, p)| (a - p).abs() < 2e-6));
}
#[test]
#[ignore = "requires Vulkan or Metal"]
fn masked_softmax_backward_matches_independent_analytic_reference() {
    let mut g = Graph::new();
    let z = g.parameter("logits", &[12]);
    let p = g.input("prior", &[12]);
    let weights = mixture::build(&mut g, z, p, 2, 1);
    let target = g.input("target", &[12]);
    let loss = g.mse_loss(weights, target);
    g.set_outputs(vec![loss, weights]);
    let mut s = ommatidia::gpu::training_session(&g, ommatidia::gpu::create_context(None, false));
    let prior = [
        0.02, 0.0, 0.06, 0.2, 0.12, 0.0, 0.3, 0.3, 0.5, 0.0, 0.0, 0.5,
    ];
    let logits = [
        -0.8, 1000.0, 0.4, -0.1, 0.8, -1000.0, -0.4, 0.5, 0.0, -3.0, 2000.0, 0.8,
    ];
    let target = [0.1, 0.0, 0.2, 0.3, 0.3, 0.0, 0.1, 0.1, 0.3, 0.0, 0.0, 0.6];
    let w = reference(&logits, &prior, 2);
    s.set_parameter("logits", &logits);
    s.set_input("prior", &prior);
    s.set_input("target", &target);
    s.set_adam(0.0, 0.9, 0.999, 1e-8);
    s.step();
    s.wait();
    let mut grad = [0.0; 12];
    s.read_param_grad("logits", &mut grad);
    for i in 0..2 {
        let mean = (0..6)
            .map(|k| {
                let j = k * 2 + i;
                w[j] as f64 * 2.0 * (w[j] - target[j]) as f64 / 12.0
            })
            .sum::<f64>();
        for k in 0..6 {
            let j = k * 2 + i;
            let expected = mixture::SOFTMAX_GAIN as f64
                * w[j] as f64
                * (2.0 * (w[j] - target[j]) as f64 / 12.0 - mean);
            assert!(
                (grad[j] as f64 - expected).abs() < 2e-7,
                "{j}: {} != {expected}",
                grad[j]
            );
            if prior[j] == 0.0 {
                assert_eq!(grad[j], 0.0);
            }
        }
        assert!((0..6).map(|k| grad[k * 2 + i]).sum::<f32>().abs() < 2e-7);
    }
    assert!(grad.iter().any(|v| v.abs() > 1e-4));
}
#[test]
#[ignore = "requires Vulkan or Metal"]
fn modes_share_initial_parameters_and_zero_head_native_recurrence() {
    let context = ommatidia::gpu::create_context(None, false);
    let old = Config::default();
    let new = Config {
        version: 2,
        mixture: mixture::Mode::MaskedSoftmax,
        ..old
    };
    let mut a = native::Native::new(Arc::clone(&context), old, [8, 8]).unwrap();
    let mut b = native::Native::new(context, new, [8, 8]).unwrap();
    assert_eq!(a.network.params.len(), b.network.params.len());
    for (p, q) in a.network.params.iter().zip(&b.network.params) {
        assert_eq!((&p.name, p.len), (&q.name, q.len));
        let mut av = vec![0.0; p.len];
        let mut bv = vec![0.0; p.len];
        a.session.read_param(&p.name, &mut av);
        b.session.read_param(&q.name, &mut bv);
        assert_eq!(av, bv);
    }
    for index in 0..4 {
        let (frame, _) = fixture(old, index, 19);
        let first = a.process(&frame).unwrap();
        let second = b.process(&frame).unwrap();
        assert!(
            first
                .iter()
                .zip(second)
                .all(|(a, b)| (a - b).abs() < 2e-5 * (1.0 + a.abs()))
        );
    }
    a.reset();
    b.reset();
    let (frame, _) = fixture(old, 0, 19);
    let first = a.process(&frame).unwrap();
    let second = b.process(&frame).unwrap();
    assert!(
        first
            .iter()
            .zip(second)
            .all(|(a, b)| (a - b).abs() < 2e-5 * (1.0 + a.abs()))
    );
    let model = graph::build(new, [8, 8], 2).unwrap();
    assert_eq!(model.params.len(), b.network.params.len());
}
