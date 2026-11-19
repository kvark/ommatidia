//! Replay a complete checkpoint transaction and inspect the last cursor's warp gradients.
//! Pass model.safetensors, report.json and the original ordered training captures.
//! Use --synthetic-only report.json to exercise just the collision and overflow cases.
use meganeura::{Session, compile::ShaderEntry};
use ommatidia::transport::Config;
use ommatidia_train::{
    Result,
    checkpoint::Checkpoint,
    corpus::Corpus,
    sampler::{Prefetch, learning_rate},
    training::Trainer,
};
use serde::Serialize;
use std::{path::PathBuf, sync::Arc};

#[derive(Default, Serialize)]
struct ErrorStats {
    destinations: usize,
    nonzero_contributors_max: usize,
    unequal_to_ordered_f32: usize,
    outside_f32_roundoff_bound: usize,
    max_absolute_error_to_f64: f64,
}

struct Scatter {
    ids: Vec<u32>,
    values: Vec<f32>,
    result: Vec<f32>,
}

impl Scatter {
    fn check(&self, stats: &mut ErrorStats) {
        let n = self.result.len();
        let mut ordered = vec![0.0f32; n];
        let mut accurate = vec![0.0f64; n];
        let mut magnitude = vec![0.0f64; n];
        let mut count = vec![0usize; n];
        for (&id, &value) in self.ids.iter().zip(&self.values) {
            assert!(value.is_finite());
            let j = id as usize;
            if j < n && value != 0.0 {
                ordered[j] += value;
                accurate[j] += f64::from(value);
                magnitude[j] += f64::from(value).abs();
                count[j] += 1;
            }
        }
        for (j, &gpu) in self.result.iter().enumerate() {
            assert!(gpu.is_finite());
            stats.destinations += 1;
            stats.nonzero_contributors_max = stats.nonzero_contributors_max.max(count[j]);
            stats.unequal_to_ordered_f32 += usize::from(gpu != ordered[j]);
            let error = (f64::from(gpu) - accurate[j]).abs();
            let bound = 2.0 * count[j] as f64 * f64::from(f32::EPSILON) * magnitude[j];
            stats.outside_f32_roundoff_bound += usize::from(error > bound + 1e-37);
            stats.max_absolute_error_to_f64 = stats.max_absolute_error_to_f64.max(error);
        }
    }
}

fn warps(session: &Session) -> Vec<(String, Scatter)> {
    let plan = session.plan();
    plan.dispatches
        .iter()
        .filter(|d| {
            d.shader == ShaderEntry::ScatterAddAtomic && d.params[2] == 1 && d.params[3] != 3
        })
        .map(|d| {
            let source_len = d.params[1] as usize;
            let read = |buffer, n| {
                let mut values = vec![0.; n];
                session.read_buffer(buffer, &mut values);
                values
            };
            let name = &plan
                .input_buffers
                .iter()
                .find(|(_, b)| *b == d.input_buffers[0])
                .expect("named warp indices")
                .0;
            assert!(name.contains(".warp"));
            let ids = read(d.input_buffers[0], source_len)
                .into_iter()
                .map(f32::to_bits)
                .collect();
            let mut values = read(d.input_buffers[1], source_len);
            if matches!(d.params[3], 1 | 2) {
                for (value, scale) in values.iter_mut().zip(read(d.input_buffers[2], source_len)) {
                    *value *= scale;
                }
            }
            (
                name.clone(),
                Scatter {
                    ids,
                    values,
                    result: read(d.output_buffer, d.params[0] as usize),
                },
            )
        })
        .collect()
}

fn synthetic(context: Arc<blade_graphics::Context>) -> ErrorStats {
    const N: usize = 2048;
    let mut graph = meganeura::Graph::new();
    let index = graph.input_u32("index", &[N]);
    let value = graph.input("value", &[N, 1]);
    let out = graph.scatter_add(index, value, N);
    graph.set_outputs(vec![out]);
    let mut session = ommatidia::gpu::inference_session(&graph, context);
    // Distribute contributors across workgroups. Remaining sources are out of bounds.
    let mut ids = vec![u32::MAX; N];
    let mut values = vec![17.; N];
    let mut i = 0;
    for (dst, degree) in [0, 1, 2, 7, 8, 9, 33].into_iter().enumerate() {
        for _ in 0..degree {
            let row = (i * 271) % N;
            ids[row] = dst as u32;
            values[row] = [1e4, 0.0001, -1e4, 0.0002, 0.0003][i % 5];
            i += 1;
        }
    }
    let invalid = (i * 271) % N;
    ids[invalid] = N as u32;
    let zero = ((i + 1) * 271) % N;
    ids[zero] = 0;
    values[zero] = 0.0;
    session.set_input_u32("index", &ids);
    session.set_input("value", &values);
    let mut sample = Scatter {
        ids,
        values,
        result: vec![0.; N],
    };
    let mut stats = ErrorStats::default();
    for _ in 0..16 {
        session.step();
        session.wait_read_output(0, &mut sample.result);
        sample.check(&mut stats);
    }
    stats
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1).map(PathBuf::from);
    let path = args.next().ok_or("expected model.safetensors")?;
    let output = args.next().ok_or("expected output JSON path")?;
    let device = std::env::var("MEGANEURA_DEVICE_ID")
        .ok()
        .map(|s| ommatidia::gpu::parse_device_id(&s))
        .transpose()?;
    let ctx = ommatidia::gpu::create_context(device, false);
    let synthetic = synthetic(Arc::clone(&ctx));
    if path.as_os_str() == "--synthetic-only" {
        std::fs::write(output, serde_json::to_vec_pretty(&synthetic)?)?;
        assert_eq!(synthetic.nonzero_contributors_max, 33);
        assert_eq!(synthetic.outside_f32_roundoff_bound, 0);
        return Ok(());
    }
    let saved: Checkpoint =
        serde_json::from_slice(&std::fs::read(path.with_file_name("trainer.json"))?)?;
    let config = Config::parse(&saved.settings.model)?;
    let corpus = Arc::new(Corpus::open(&args.collect::<Vec<_>>(), config)?);
    let mut t = Trainer::new(
        Arc::clone(&ctx),
        config,
        saved.sampler.input_extent(),
        saved.settings.unroll,
        saved.settings.batch,
        saved.settings.margin,
    )?;
    // Preserve the normal graph optimizations, but retain intermediate buffers for readback.
    t.session = meganeura::train::build(
        &t.network.graph,
        meganeura::SessionConfig {
            mode: meganeura::Mode::Training,
            gpu: Some(ctx),
            runtime: meganeura::SessionOptions {
                no_alias: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .0;
    t.session.set_grad_accumulate(saved.settings.batch as u32);
    t.session.set_grad_clip_norm(1.0);
    let batch = Prefetch::new(Arc::clone(&corpus), config, saved.sampler.clone()).receive()?;
    let rate = learning_rate(
        saved.settings.peak_rate,
        saved.step + 1,
        saved.settings.steps,
    );
    struct Replay {
        forward: (Vec<f32>, Vec<f32>),
        gradients: Vec<f32>,
        taps: Vec<(String, Scatter)>,
    }
    let mut baseline: Option<Replay> = None;
    let mut errors = ErrorStats::default();
    let mut first_changing_warp = None;
    let mut max_grad_delta = 0.0f32;
    let mut forward_changes = 0;
    let mut warp_names = vec![];
    for _ in 0..16 {
        Checkpoint::restore(&path, &mut t, &saved.settings, &corpus.identities())?;
        t.step(&batch, rate, saved.settings.weights)?;
        let forward = (t.last_microbatch_losses(), t.read_states());
        let gradients: Vec<f32> = t
            .network
            .params
            .iter()
            .flat_map(|p| {
                let mut values = vec![0.; p.len];
                t.session.read_param_grad(&p.name, &mut values);
                values
            })
            .collect();
        assert!(gradients.iter().all(|v| v.is_finite()) && gradients.iter().any(|&v| v != 0.0));
        let taps = warps(&t.session);
        assert!(!taps.is_empty());
        for (_, tap) in &taps {
            tap.check(&mut errors);
        }
        if let Some(first) = &baseline {
            forward_changes += usize::from(forward != first.forward);
            for (x, y) in gradients.iter().zip(&first.gradients) {
                max_grad_delta = max_grad_delta.max((x - y).abs());
            }
            for (index, ((_, tap), (_, initial))) in taps.iter().zip(&first.taps).enumerate() {
                if tap.ids == initial.ids
                    && tap.values == initial.values
                    && tap.result != initial.result
                {
                    first_changing_warp =
                        Some(first_changing_warp.map_or(index, |old: usize| old.min(index)));
                }
            }
        } else {
            warp_names = taps.iter().map(|(name, _)| name.clone()).collect();
            baseline = Some(Replay {
                forward,
                gradients,
                taps,
            });
        }
    }
    let report = serde_json::json!({
        "repetitions":16, "synthetic":synthetic, "last_cursor_warps":warp_names,
        "measured_warps":errors, "first_warp_with_same_inputs_and_different_output":first_changing_warp.map(|i| &warp_names[i]),
        "changed_forward_losses_or_carries":forward_changes, "max_parameter_gradient_delta":max_grad_delta
    });
    std::fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    assert_eq!(errors.outside_f32_roundoff_bound, 0);
    assert_eq!(synthetic.outside_f32_roundoff_bound, 0);
    println!("{report}");
    Ok(())
}
