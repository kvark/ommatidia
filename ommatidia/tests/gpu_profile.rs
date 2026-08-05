//! Where the frame time actually goes.
//!
//! `frame_cost_by_model_size` in `gpu_model` established that cutting the
//! network by a factor of four hundred in parameters bought only a factor of
//! eight in time, so the cost is not in the weights. This finds out what it is
//! in, because the answer decides where the fix belongs: slow kernels are
//! meganeura's problem, launch overhead is an architecture problem here.
//!
//! The discriminator is the sum of the GPU pass timings against the wall clock
//! of an unprofiled step. If the passes account for the wall time, the kernels
//! are slow. If they do not, the gap is submission and synchronisation.
//!
//! ```sh
//! OMMATIDIA_DEVICE_ID=0x744c \
//!   cargo test -p ommatidia --release --test gpu_profile -- --ignored --nocapture
//! ```
//!
//! Timestamps and the rewrite policy are typed options here, not environment
//! variables — meganeura's core stopped reading the environment, so a client
//! that wants either has to ask.

use ommatidia::model::{ModelConfig, Objective, build};
use ommatidia::rng::Rng;
use ommatidia::{Plane, PlaneSet};

fn config(tile: u32) -> ModelConfig {
    ModelConfig {
        scale: 2,
        tile,
        batch: 1,
        cond_planes: PlaneSet::new()
            .with(Plane::Color)
            .with(Plane::Depth)
            .with(Plane::Normal)
            .with(Plane::DiffuseAlbedo)
            .with(Plane::SpecularF0)
            .with(Plane::Roughness),
        base_channels: 64,
        level_multipliers: vec![1, 2, 4],
        blocks_per_level: 2,
        num_groups: 8,
        objective: Objective::Direct,
        ..ModelConfig::default()
    }
}

/// Build an inference session with timestamps enabled and an explicit
/// rewrite policy.
///
/// Meganeura's core no longer reads the environment, so both of the things
/// this test needs — timestamp query pools, and whether the Winograd rewrite
/// fires — arrive as typed options. That also makes the Winograd comparison a
/// single self-contained run instead of two invocations with a variable
/// flipped between them.
fn profiling_session(
    graph: &meganeura::Graph,
    context: std::sync::Arc<blade_graphics::Context>,
    winograd: bool,
) -> meganeura::Session {
    meganeura::train::build(
        graph,
        meganeura::SessionConfig {
            mode: meganeura::Mode::Inference,
            gpu: Some(context),
            optimize: meganeura::OptimizeConfig {
                no_winograd: !winograd,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .0
}

/// GroupNorm launches `batch * num_groups` workgroups regardless of how big
/// the image is, so at batch 1 it runs on a handful of the device's compute
/// units whatever the resolution. Raising the group count is a direct test of
/// that: it changes the parallelism and nothing else about the work.
///
/// The normalisation is not equivalent across these, so this is a timing probe
/// rather than a configuration anyone would train.
#[test]
#[ignore = "requires a GPU"]
fn group_norm_parallelism_is_independent_of_image_size() {
    ommatidia::gpu::warn_if_busy();
    let context = ommatidia::gpu::context(true);
    for groups in [8u32, 32, 64] {
        let mut config = config(512);
        config.num_groups = groups;
        let model = build(&config, false).expect("build");
        let mut session = profiling_session(&model.graph, context.clone(), true);
        model.initialize(&mut session, 1);
        let mut rng = Rng::new(1);
        let cond: Vec<f32> = (0..config.cond_len()).map(|_| rng.normal() * 0.5).collect();
        session.set_input("cond", &cond);

        for _ in 0..3 {
            session.step();
            session.wait();
        }
        const RUNS: u32 = 6;
        let started = std::time::Instant::now();
        for _ in 0..RUNS {
            session.step();
            session.wait();
        }
        let wall = started.elapsed().as_secs_f64() / RUNS as f64;
        println!(
            "num_groups {groups:>3} => {:>4} workgroups per GroupNorm, {:.1} ms/frame",
            groups,
            wall * 1e3
        );
    }
}

#[test]
#[ignore = "requires a GPU"]
fn where_the_frame_time_goes() {
    ommatidia::gpu::warn_if_busy();
    let context = ommatidia::gpu::context(true);

    const TILE: u32 = 512;
    let config = config(TILE);
    let model = build(&config, false).expect("build");
    let mut session = profiling_session(&model.graph, context.clone(), true);
    model.initialize(&mut session, 1);

    let dispatches = session.plan().dispatches.len();
    // Every group boundary is a global barrier: it drains the pipeline and
    // forbids the next dispatch overlapping the previous one. A chain-shaped
    // network puts one dispatch in most groups.
    let groups = session.num_groups();
    let mut rng = Rng::new(1);
    let cond: Vec<f32> = (0..config.cond_len()).map(|_| rng.normal() * 0.5).collect();
    session.set_input("cond", &cond);

    // Unprofiled wall clock, which is what a caller actually pays.
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
    let wall = started.elapsed().as_secs_f64() / RUNS as f64;

    println!(
        "\n{TILE}^2 -> {}^2, {} parameters, {dispatches} dispatches in {groups} barrier groups \
         ({:.2} dispatches per group)",
        TILE * config.scale,
        model.params.iter().map(|p| p.len).sum::<usize>(),
        dispatches as f64 / groups as f64,
    );
    println!("unprofiled wall clock: {:.2} ms/frame", wall * 1e3);
    println!(
        "if launch bound, that is {:.1} us per dispatch\n",
        wall * 1e6 / dispatches as f64
    );

    // Profiled: one pass per dispatch, with timestamps. The extra passes cost
    // something themselves, so this total runs above the wall clock above —
    // what matters is the breakdown and the share the kernels account for.
    // Blade exposes a submission's timestamps only once the encoder is
    // restarted, so the timings trail the step that produced them. Stepping a
    // few times and dumping after each makes the lag visible rather than
    // guessed at.
    session.set_profiling(true);
    for round in 0..4 {
        session.step();
        session.wait();
        println!("after profiled step {round}:");
        session.dump_gpu_timings();
    }
}

/// Winograd against the direct convolution path, in one run.
///
/// The rewrite fires only above `in_channels * out_channels >= 4096`, a
/// heuristic with no term for the image size, so a workload with modest
/// channels over a large frame sits right at its boundary. This measures
/// which side it belongs on rather than arguing about it.
///
/// The two arms are **interleaved**, not run one after the other. Anything
/// else sharing the GPU drifts over the seconds a sequential comparison takes,
/// and a drift between the arms lands entirely on whichever ran second — which
/// is enough to reverse the verdict. Alternating rounds spreads it across both.
#[test]
#[ignore = "requires a GPU"]
fn winograd_earns_its_place() {
    ommatidia::gpu::warn_if_busy();
    let context = ommatidia::gpu::context(false);
    let config = config(512);
    let model = build(&config, false).expect("build");

    let mut sessions: Vec<meganeura::Session> = [true, false]
        .into_iter()
        .map(|winograd| {
            let mut session = profiling_session(&model.graph, context.clone(), winograd);
            model.initialize(&mut session, 1);
            let mut rng = Rng::new(1);
            let cond: Vec<f32> = (0..config.cond_len()).map(|_| rng.normal() * 0.5).collect();
            session.set_input("cond", &cond);
            for _ in 0..3 {
                session.step();
                session.wait();
            }
            session
        })
        .collect();

    const ROUNDS: usize = 8;
    let mut totals = [0.0f64; 2];
    for _ in 0..ROUNDS {
        for (arm, session) in sessions.iter_mut().enumerate() {
            let started = std::time::Instant::now();
            session.step();
            session.wait();
            totals[arm] += started.elapsed().as_secs_f64();
        }
    }

    let on = totals[0] / ROUNDS as f64 * 1e3;
    let off = totals[1] / ROUNDS as f64 * 1e3;
    println!(
        "winograd on : {on:>8.1} ms/frame, {} dispatches",
        sessions[0].plan().dispatches.len()
    );
    println!(
        "winograd off: {off:>8.1} ms/frame, {} dispatches",
        sessions[1].plan().dispatches.len()
    );
    println!("winograd is {:.2}x the direct path", off / on);

    // Not asserted as a ratio: the margin depends on the device, and a busy
    // one compresses every ratio toward one. Printed so a regression in
    // either path shows up in the log.
    assert!(on.is_finite() && off.is_finite());
}
