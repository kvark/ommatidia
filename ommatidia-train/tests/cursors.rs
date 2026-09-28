use ommatidia::transport::{graph, native};
use ommatidia_train::{
    checkpoint::{Checkpoint, Settings, model_text},
    sampler::{Batch, Sampler, learning_rate},
    training::Trainer,
};
use std::sync::Arc;
include!("../../ommatidia/tests/fixtures/transport.rs");

fn context() -> Arc<blade_graphics::Context> {
    let id = std::env::var("MEGANEURA_DEVICE_ID")
        .ok()
        .map(|s| ommatidia::gpu::parse_device_id(&s).unwrap());
    ommatidia::gpu::create_context(id, false)
}
fn batch(sampler: &mut Sampler, config: Config) -> Batch {
    let windows = sampler.next_windows();
    let frames = windows
        .iter()
        .map(|w| {
            (0..sampler.unroll)
                .map(|i| {
                    let (mut frame, mut target) =
                        fixture(config, w.start + i, w.sequence as u64 + 31);
                    for r in &mut frame.rays {
                        for v in r.diffuse[..3].iter_mut().chain(&mut r.specular[..3]) {
                            *v *= w.gain;
                        }
                    }
                    for s in &mut frame.surfaces {
                        for v in &mut s.emission[..3] {
                            *v *= w.gain;
                        }
                    }
                    for v in target.lobes.iter_mut().chain(&mut target.rgb) {
                        *v *= w.gain;
                    }
                    (frame, target)
                })
                .collect()
        })
        .collect();
    Batch {
        windows,
        frames,
        next_sampler: sampler.clone(),
        decode_seconds: 0.0,
    }
}
fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance,
            "{i}: {a} vs {b}"
        );
    }
}

#[test]
#[ignore = "requires Vulkan or Metal"]
fn gpu_cursor_carry_matches_24_frame_causal_inference_and_packing() {
    let context = context();
    let config = Config {
        channels: 2,
        ..Config::default()
    };
    let mut trainer = Trainer::new(Arc::clone(&context), config, [8; 2], 4, 2, 4).unwrap();
    // Inference mode exercises the actual transfer/pack/carry path without AD.
    trainer.session =
        ommatidia::gpu::inference_session(&trainer.network.graph, Arc::clone(&context));
    trainer.initialize(7, [0.5; 6]);
    let mut rng = ommatidia::rng::Rng::new(11);
    for p in &trainer.network.params {
        if p.name.starts_with("head.") && p.name.ends_with("weight") {
            trainer.session.set_parameter(
                &p.name,
                &(0..p.len).map(|_| 0.02 * rng.normal()).collect::<Vec<_>>(),
            );
        }
    }
    let mut native = (0..2)
        .map(|_| native::Native::new(Arc::clone(&context), config, [8; 2]).unwrap())
        .collect::<Vec<_>>();
    for model in &mut native {
        trainer.share_parameters(&mut model.session).unwrap();
    }
    let mut sampler = Sampler::new(8, 2, [8; 2], 4, [8; 2], 3, 64).unwrap();
    for step in 0..6 {
        let mut batch = batch(&mut sampler, config);
        for (frame, _) in batch.frames.iter_mut().flatten() {
            frame.surfaces[0].motion = [-2.0, 0.0, 0.0, 0.0];
            frame.surfaces[1].motion = [-1.25, 0.0, 0.0, 0.0];
        }
        // Explicit, uninterrupted 24-frame streams with one cut in stream 1.
        for (c, w) in batch.windows.iter_mut().enumerate() {
            w.reset = step == 0 || (c == 1 && step == 3);
        }
        for (c, model) in native.iter_mut().enumerate() {
            if batch.windows[c].reset {
                model.reset();
            }
            for (frame, _) in &batch.frames[c] {
                model.advance(frame).unwrap();
            }
        }
        trainer
            .run_batch(&batch, None, graph::LossWeights::default())
            .unwrap();
        let state = trainer.read_states();
        for (name, expected) in graph::LOSS_MASK_NAMES
            .iter()
            .zip(graph::loss_masks(config, [8; 2], [0; 4]).unwrap())
        {
            let buffer = trainer
                .session
                .plan()
                .input_buffers
                .iter()
                .find(|(key, _)| key == name)
                .unwrap()
                .1;
            let mut actual = vec![0.0; expected.len()];
            trainer.session.read_buffer(buffer, &mut actual);
            assert_eq!(actual, expected, "GPU loss mask {name}");
        }
        let stride = state.len() / 2;
        for (c, model) in native.iter().enumerate() {
            close(
                &state[c * stride..(c + 1) * stride],
                &model.read_state().values,
                1e-5,
            );
        }
        // Out-of-crop right-edge taps cannot read the neighboring full image.
        let frame = &batch.frames[1][3].0;
        let prepared = native[1].read_prepared(frame);
        assert_eq!(prepared.validity[config.index([8; 2], 0, 0, 0)], 0.0);
        assert_eq!(prepared.validity[config.index([8; 2], 0, 1, 0)], 1.0);
        for k in 0..4 {
            assert!(prepared.indices[k].iter().all(|i| *i < stride as u32));
        }
        // Targets use a separate shader and cannot enter observation features.
        // Readback here is test-only, not part of the production update loop.
        for (slot, (frame, target)) in batch.frames[1].iter().enumerate() {
            let read = |name: &str, len| {
                let key = format!("f{slot}.{name}");
                let buffer = trainer
                    .session
                    .plan()
                    .input_buffers
                    .iter()
                    .find(|(name, _)| name == &key)
                    .unwrap()
                    .1;
                let mut result = vec![0.0; len];
                trainer.session.read_buffer(buffer, &mut result);
                result
            };
            assert_eq!(read("target", target.lobes.len()), target.lobes);
            let n = frame.surfaces.len();
            let mut rgb = vec![0.0; 3 * n];
            let mut albedo = rgb.clone();
            let mut emission = rgb.clone();
            for (i, s) in frame.surfaces.iter().enumerate() {
                for c in 0..3 {
                    let j = config.index([8; 2], c, i % 16, i / 16);
                    rgb[j] = target.rgb[3 * i + c];
                    albedo[j] = s.albedo_roughness[c];
                    emission[j] = s.emission[c];
                }
            }
            assert_eq!(read("rgb.target", 3 * n), rgb);
            assert_eq!(read("rgb.albedo", 3 * n), albedo);
            assert_eq!(read("rgb.emission", 3 * n), emission);
        }
        assert_eq!(trainer.session.adam_step_count(), 0);
    }
}

#[test]
#[ignore = "requires Vulkan or Metal"]
fn resume_restores_adam_schedule_rng_and_gpu_state() {
    let context = context();
    let config = Config {
        channels: 2,
        ..Config::default()
    };
    let mut trainer = Trainer::new(Arc::clone(&context), config, [8; 2], 4, 2, 4).unwrap();
    trainer.initialize(7, [0.5; 6]);
    let settings = Settings {
        steps: 4,
        batch: 2,
        unroll: 4,
        crop: [8; 2],
        margin: 4,
        peak_rate: 0.003,
        seed: 7,
        weights: graph::LossWeights::default(),
        model: model_text(config).unwrap(),
    };
    let mut sampler = Sampler::new(7, 2, [8; 2], 4, [8; 2], 3, 64).unwrap();
    for step in 1..=2 {
        trainer
            .step(
                &batch(&mut sampler, config),
                learning_rate(settings.peak_rate, step, 4),
                settings.weights,
            )
            .unwrap();
    }
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory =
        std::env::temp_dir().join(format!("ommatidia-resume-{}-{nonce}", std::process::id()));
    Checkpoint::save(
        &directory,
        &mut trainer,
        settings.clone(),
        Vec::new(),
        sampler.clone(),
        [0.5; 6],
    )
    .unwrap();
    for step in 3..=4 {
        let (loss, _) = trainer
            .step(
                &batch(&mut sampler, config),
                learning_rate(settings.peak_rate, step, 4),
                settings.weights,
            )
            .unwrap();
        let microbatch = trainer.last_microbatch_losses();
        assert_eq!(microbatch.len(), 2);
        assert_eq!(loss, microbatch.iter().sum::<f32>() / 2.0);
        assert!(trainer.read_states().iter().all(|v| v.is_finite()));
    }
    let mut restored = Trainer::new(Arc::clone(&context), config, [8; 2], 4, 2, 4).unwrap();
    let checkpoint = directory.join("model.safetensors");
    let mut wrong = settings.clone();
    wrong.steps = 5;
    assert!(Checkpoint::restore(&checkpoint, &mut restored, &wrong, &[]).is_err());
    let saved = Checkpoint::restore(&checkpoint, &mut restored, &settings, &[]).unwrap();
    assert_eq!(restored.session.adam_step_count(), 2);
    let mut resumed_sampler = saved.sampler;
    for step in 3..=4 {
        restored
            .step(
                &batch(&mut resumed_sampler, config),
                learning_rate(settings.peak_rate, step, 4),
                settings.weights,
            )
            .unwrap();
    }
    assert_eq!(sampler.next_windows(), resumed_sampler.next_windows());
    assert_eq!(restored.session.adam_step_count(), 4);
    let names = trainer
        .network
        .params
        .iter()
        .map(|p| p.name.as_str())
        .collect::<Vec<_>>();
    for (a, b) in trainer
        .session
        .read_params(&names)
        .iter()
        .zip(restored.session.read_params(&names))
    {
        close(a, &b, 1e-6);
    }
    close(&trainer.read_states(), &restored.read_states(), 1e-5);
    let metadata_path = directory.join("trainer.json");
    let metadata = std::fs::read(&metadata_path).unwrap();
    let mut old: serde_json::Value = serde_json::from_slice(&metadata).unwrap();
    old["schema"] = 1.into();
    std::fs::write(&metadata_path, serde_json::to_vec(&old).unwrap()).unwrap();
    let error = Checkpoint::restore(&checkpoint, &mut restored, &settings, &[])
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("training-loss schema differs"), "{error}");
    std::fs::write(&metadata_path, metadata).unwrap();
    // A changed state file cannot silently turn resume into a warm start.
    std::fs::write(directory.join("state.f32"), [0; 4]).unwrap();
    assert!(Checkpoint::restore(&checkpoint, &mut restored, &settings, &[]).is_err());
    for name in [
        "model.safetensors",
        "model.transport.ron",
        "trainer.json",
        "state.f32",
    ] {
        std::fs::remove_file(directory.join(name)).unwrap();
    }
    std::fs::remove_dir(directory).unwrap();
}
