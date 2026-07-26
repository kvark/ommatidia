//! The network has to survive meganeura's compiler and run on a real device.
//!
//! These need a GPU, so they are ignored by default:
//!
//! ```sh
//! cargo test -p ommatidia --test gpu_model -- --ignored --nocapture
//! ```

use ommatidia::model::{ModelConfig, Objective, build};
use ommatidia::rng::Rng;
use ommatidia::{Plane, PlaneSet};

/// Small enough to compile quickly, structurally the full network.
fn config() -> ModelConfig {
    ModelConfig {
        scale: 2,
        tile: 16,
        batch: 1,
        cond_planes: PlaneSet::new().with(Plane::Color).with(Plane::Depth),
        base_channels: 16,
        level_multipliers: vec![1, 2],
        blocks_per_level: 1,
        num_groups: 8,
        time_input_dim: 16,
        time_embed_dim: 32,
        ..ModelConfig::default()
    }
}

fn filled(rng: &mut Rng, len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|_| rng.normal() * scale).collect()
}

#[test]
#[ignore = "requires a GPU"]
fn inference_session_runs() {
    let config = config();
    let model = build(&config, false).expect("build");
    let mut session = meganeura::build_inference_session(&model.graph);
    model.initialize(&mut session, 1);

    let mut rng = Rng::new(7);
    session.set_input("cond", &filled(&mut rng, config.cond_len(), 0.5));
    session.set_input("x_t", &filled(&mut rng, config.target_len(), 1.0));
    session.set_input("t_emb", &filled(&mut rng, config.time_len(), 1.0));

    session.step();
    session.wait();

    let out: Vec<f32> = session.read_output(config.target_len());
    assert_eq!(out.len(), config.target_len());
    assert!(
        out.iter().all(|v| v.is_finite()),
        "output has non-finite values"
    );
    // The head is zero-initialised, so an untrained network predicts nothing.
    assert!(
        out.iter().all(|&v| v == 0.0),
        "a zero-initialised head should predict exactly zero"
    );
}

/// The real check: gradients flow all the way back and the loss comes down on
/// a batch the network is allowed to memorise.
#[test]
#[ignore = "requires a GPU"]
fn training_overfits_one_batch() {
    let config = config();
    let model = build(&config, true).expect("build");
    let mut session = meganeura::build_session(&model.graph);
    model.initialize(&mut session, 1);

    let mut rng = Rng::new(3);
    let cond = filled(&mut rng, config.cond_len(), 0.5);
    let x_t = filled(&mut rng, config.target_len(), 1.0);
    let t_emb = filled(&mut rng, config.time_len(), 1.0);
    let target = filled(&mut rng, config.target_len(), 1.0);

    session.set_input("cond", &cond);
    session.set_input("x_t", &x_t);
    session.set_input("t_emb", &t_emb);
    session.set_input("target", &target);
    session.set_adam(2e-3, 0.9, 0.999, 1e-8);

    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for step in 0..60 {
        session.step();
        session.wait();
        let loss = session.read_loss();
        assert!(loss.is_finite(), "loss went non-finite at step {step}");
        if step == 0 {
            first = loss;
        }
        last = loss;
        if step % 10 == 0 {
            println!("step {step:>3}: loss {loss:.6}");
        }
    }

    println!("loss {first:.6} -> {last:.6}");
    assert!(
        last < first * 0.5,
        "loss barely moved: {first:.6} -> {last:.6}"
    );
}

#[test]
#[ignore = "requires a GPU"]
fn direct_objective_runs_without_a_timestep() {
    let mut config = config();
    config.objective = Objective::Direct;
    let model = build(&config, true).expect("build");
    let mut session = meganeura::build_session(&model.graph);
    model.initialize(&mut session, 1);

    let mut rng = Rng::new(5);
    session.set_input("cond", &filled(&mut rng, config.cond_len(), 0.5));
    session.set_input("target", &filled(&mut rng, config.target_len(), 1.0));
    session.set_adam(2e-3, 0.9, 0.999, 1e-8);

    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for step in 0..40 {
        session.step();
        session.wait();
        let loss = session.read_loss();
        assert!(loss.is_finite(), "loss went non-finite at step {step}");
        if step == 0 {
            first = loss;
        }
        last = loss;
    }
    println!("direct loss {first:.6} -> {last:.6}");
    assert!(
        last < first * 0.5,
        "loss barely moved: {first:.6} -> {last:.6}"
    );
}
