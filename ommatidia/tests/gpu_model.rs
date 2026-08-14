//! The network has to survive meganeura's compiler and run on a real device.
//!
//! These need a GPU, so they are ignored by default:
//!
//! ```sh
//! OMMATIDIA_TEST_DEVICE_ID=0x744c \
//!   cargo test -p ommatidia --test gpu_model -- --ignored --nocapture
//! ```

use ommatidia::model::{ModelConfig, Objective, ReconstructionBase, build, build_for_extent};
use ommatidia::rng::Rng;
use ommatidia::{Plane, PlaneSet};

/// Tests have no host context to inherit. Adapter choice is therefore a test
/// runner input, parsed here at the executable boundary rather than by the
/// Ommatidium library.
fn context(timing: bool) -> std::sync::Arc<blade_graphics::Context> {
    let value = std::env::var("OMMATIDIA_TEST_DEVICE_ID").expect(
        "GPU measurements require OMMATIDIA_TEST_DEVICE_ID; the default adapter may not be the one intended",
    );
    let device_id =
        ommatidia::gpu::parse_device_id(&value).expect("invalid OMMATIDIA_TEST_DEVICE_ID");
    ommatidia::gpu::create_context(Some(device_id), timing)
}

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
        objective: Objective::Diffusion,
        reconstruction_base: ReconstructionBase::Bilinear,
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
    let mut session = ommatidia::gpu::inference_session(&model.graph, context(false));
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
    let mut session = ommatidia::gpu::training_session(&model.graph, context(false));
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
    let mut session = ommatidia::gpu::training_session(&model.graph, context(false));
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
    ommatidia::gpu::warn_if_busy();
    // Use real display aspect ratios. The runtime graph supports rectangular
    // extents even though training crops are square.
    for extent in [[640u32, 360], [960, 540], [1280, 720]] {
        let config = ModelConfig {
            scale: 2,
            tile: 64,
            batch: 1,
            cond_planes: PlaneSet::new()
                .with(Plane::Color)
                .with(Plane::Depth)
                .with(Plane::Normal)
                .with(Plane::DiffuseAlbedo)
                .with(Plane::SpecularF0)
                .with(Plane::Roughness),
            ..ModelConfig::default()
        };
        let Ok(model) = build_for_extent(&config, false, extent) else {
            println!("{}x{}: rejected by validate()", extent[0], extent[1]);
            continue;
        };
        let mut session = ommatidia::gpu::inference_session(&model.graph, context(false));
        model.initialize(&mut session, 1);

        let mut rng = Rng::new(1);
        session.set_input(
            "cond",
            &filled(&mut rng, config.cond_len_for_extent(extent), 0.5),
        );

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

        let out_width = extent[0] * config.scale;
        let out_height = extent[1] * config.scale;
        let out_pixels = out_width as f64 * out_height as f64;
        let ns_per_pixel = per_frame * 1e9 / out_pixels;
        // What that rate implies for the two extents anyone would ask about.
        let at_1080p = ns_per_pixel * 1920.0 * 1080.0 / 1e6;
        let at_4k = ns_per_pixel * 3840.0 * 2160.0 / 1e6;
        println!(
            "{}x{} -> {out_width}x{out_height}: {:.1} ms/frame, {:.1} GFLOP, {ns_per_pixel:.3} ns/output pixel \
             => {at_1080p:.0} ms at 1080p, {at_4k:.0} ms at 4K",
            extent[0],
            extent[1],
            per_frame * 1e3,
            config.flops(out_pixels as usize),
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
    ommatidia::gpu::warn_if_busy();
    const INPUT: [u32; 2] = [960, 540];
    // The same shapes the quality sweep trains, so the two tables line up.
    let shapes: [(u32, usize, usize, &str); 5] = [
        (64, 3, 2, "reference"),
        (32, 3, 1, "base 32, 3 levels"),
        (24, 3, 1, "base 24, 3 levels"),
        (16, 2, 1, "base 16, 2 levels"),
        (8, 2, 1, "base 8, 2 levels"),
    ];
    for (base, levels, blocks, label) in shapes {
        let config = ModelConfig {
            scale: 2,
            tile: 64,
            batch: 1,
            base_channels: base,
            level_multipliers: (0..levels).map(|i| 1 << i).collect(),
            blocks_per_level: blocks,
            num_groups: 8,
            cond_planes: PlaneSet::new()
                .with(Plane::Color)
                .with(Plane::Depth)
                .with(Plane::Normal)
                .with(Plane::DiffuseAlbedo)
                .with(Plane::SpecularF0)
                .with(Plane::Roughness),
            objective: Objective::Direct,
            ..ModelConfig::default()
        };
        let Ok(model) = build_for_extent(&config, false, INPUT) else {
            println!("{label}: rejected by validate()");
            continue;
        };
        let params: usize = model.params.iter().map(|p| p.len).sum();
        let mut session = ommatidia::gpu::inference_session(&model.graph, context(false));
        model.initialize(&mut session, 1);
        let mut rng = Rng::new(1);
        session.set_input(
            "cond",
            &filled(&mut rng, config.cond_len_for_extent(INPUT), 0.5),
        );

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
        println!(
            "{label:<20} {params:>8} params, {:>7.1} GFLOP/1080p, \
             {:>7.1} ms at 960x540 -> 1920x1080",
            config.flops(1920 * 1080),
            per_frame * 1e3,
        );
    }
}

/// Which adapter a session actually lands on.
///
/// The test runner supplies the adapter explicitly and the session reports its
/// chosen device. On a machine with an integrated and a discrete GPU that is
/// the difference between a benchmark and a fiction, so it is worth printing
/// rather than assuming.
#[test]
#[ignore = "requires a GPU"]
fn reports_the_selected_adapter() {
    let model = build(&config(), false).expect("build");
    let session = ommatidia::gpu::inference_session(&model.graph, context(false));
    println!(
        "session landed on: {}",
        session.device_information().device_name
    );
}
