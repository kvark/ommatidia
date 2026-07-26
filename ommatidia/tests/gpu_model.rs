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

/// What one reconstruction costs at frame resolution.
///
/// The project's whole premise is a real-time budget, so the cost per output
/// pixel is the number that decides whether any of the rest matters. Timing
/// needs no trained weights — the dispatch sequence is the same either way.
///
/// ```sh
/// cargo test -p ommatidia --release --test gpu_model -- --ignored frame_cost --nocapture
/// ```
#[test]
#[ignore = "requires a GPU"]
fn frame_cost_at_realistic_extents() {
    // Square, because ModelConfig carries one extent for both axes. A 16:9
    // frame of the same area is the honest comparison.
    for tile in [256u32, 512, 768, 1024] {
        let config = ModelConfig {
            scale: 2,
            tile,
            batch: 1,
            objective: Objective::Direct,
            ..config()
        };
        let Ok(model) = build(&config, false) else {
            println!("{tile}^2: rejected by validate()");
            continue;
        };
        let mut session = meganeura::build_inference_session(&model.graph);
        model.initialize(&mut session, 1);

        let mut rng = Rng::new(1);
        session.set_input("cond", &filled(&mut rng, config.cond_len(), 0.5));

        // Warm up, then time a run of steps.
        for _ in 0..3 {
            session.step();
            session.wait();
        }
        const RUNS: u32 = 10;
        let started = std::time::Instant::now();
        for _ in 0..RUNS {
            session.step();
            session.wait();
        }
        let per_frame = started.elapsed().as_secs_f64() / RUNS as f64;

        let out_pixels = (tile * config.scale) as f64 * (tile * config.scale) as f64;
        let ns_per_pixel = per_frame * 1e9 / out_pixels;
        // What that rate implies for the two extents anyone would ask about.
        let at_1080p = ns_per_pixel * 1920.0 * 1080.0 / 1e6;
        let at_4k = ns_per_pixel * 3840.0 * 2160.0 / 1e6;
        println!(
            "{tile}^2 -> {}^2: {:.2} ms/frame, {ns_per_pixel:.3} ns/output pixel \
             => {at_1080p:.1} ms at 1080p, {at_4k:.1} ms at 4K",
            tile * config.scale,
            per_frame * 1e3,
        );
    }
}

/// Where the latency actually goes, as the network shrinks.
///
/// The frame cost above says the default backbone is roughly two orders of
/// magnitude off a real-time budget, so the useful question is not whether it
/// is slow but what has to go. Cost is reported per output pixel so the
/// configurations compare directly.
#[test]
#[ignore = "requires a GPU"]
fn frame_cost_by_model_size() {
    const TILE: u32 = 512;
    let shapes: [(u32, usize, usize, &str); 6] = [
        (64, 3, 2, "default"),
        (48, 3, 1, "narrower, one block"),
        (32, 3, 1, "half width"),
        (32, 2, 1, "half width, two levels"),
        (16, 2, 1, "quarter width, two levels"),
        (16, 1, 1, "quarter width, flat"),
    ];
    for (base, levels, blocks, label) in shapes {
        let config = ModelConfig {
            scale: 2,
            tile: TILE,
            batch: 1,
            base_channels: base,
            level_multipliers: (0..levels).map(|i| 1 << i).collect(),
            blocks_per_level: blocks,
            num_groups: 8,
            objective: Objective::Direct,
            ..config()
        };
        let Ok(model) = build(&config, false) else {
            println!("{label}: rejected by validate()");
            continue;
        };
        let params: usize = model.params.iter().map(|p| p.len).sum();
        let mut session = meganeura::build_inference_session(&model.graph);
        model.initialize(&mut session, 1);
        let mut rng = Rng::new(1);
        session.set_input("cond", &filled(&mut rng, config.cond_len(), 0.5));

        for _ in 0..3 {
            session.step();
            session.wait();
        }
        const RUNS: u32 = 10;
        let started = std::time::Instant::now();
        for _ in 0..RUNS {
            session.step();
            session.wait();
        }
        let per_frame = started.elapsed().as_secs_f64() / RUNS as f64;
        let out_pixels = (TILE * config.scale) as f64 * (TILE * config.scale) as f64;
        let at_1080p = per_frame * 1e9 / out_pixels * 1920.0 * 1080.0 / 1e6;
        println!(
            "{label:<28} base {base:>3}, {levels} levels, {blocks} blocks: \
             {params:>9} params, {:>7.1} ms/frame => {at_1080p:>7.1} ms at 1080p",
            per_frame * 1e3,
        );
    }
}
