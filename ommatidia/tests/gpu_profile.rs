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
//! OMMATIDIA_TEST_DEVICE_ID=0x744c \
//!   cargo test -p ommatidia --release --test gpu_profile -- --ignored --nocapture
//! ```
//!
//! Timestamps and the rewrite policy are typed options here, not environment
//! variables — meganeura's core stopped reading the environment, so a client
//! that wants either has to ask.

use ommatidia::model::{ModelConfig, Objective, build};
use ommatidia::rng::Rng;
use ommatidia::{FrameInputs, Plane, PlaneSet, ReconstructionBase, Upscaler};

fn context(timing: bool) -> std::sync::Arc<blade_graphics::Context> {
    let value = std::env::var("OMMATIDIA_TEST_DEVICE_ID").expect(
        "profiling requires OMMATIDIA_TEST_DEVICE_ID; silently profiling the default adapter is not trustworthy",
    );
    let device_id =
        ommatidia::gpu::parse_device_id(&value).expect("invalid OMMATIDIA_TEST_DEVICE_ID");
    ommatidia::gpu::create_context(Some(device_id), timing)
}

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

/// The shape of the checkpoint published for Blade integration. Keep the
/// detailed pass trace on the model users actually run; the larger reference
/// shape remains useful for the focused kernel comparisons below.
fn deployment_config(tile: u32) -> ModelConfig {
    ModelConfig {
        base_channels: 24,
        blocks_per_level: 1,
        ..config(tile)
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
    let context = context(true);
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
    let context = context(true);

    // Measure the shape users mean by 1080p: 960x540 input reconstructed 2x
    // to 1920x1080 output. The old 720^2 proxy had the same pixel count but
    // made the public result needlessly ambiguous.
    const INPUT: [u32; 2] = [960, 540];
    let config = if let Some(requested) = std::env::var_os("OMMATIDIA_PROFILE_CHECKPOINT") {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let requested = std::path::PathBuf::from(requested);
        let checkpoint =
            if requested.is_absolute() || requested.with_extension("safetensors").is_file() {
                requested
            } else {
                manifest.join("..").join(requested)
            };
        let (mut config, _) = ommatidia::checkpoint::load_config(&checkpoint)
            .expect("load OMMATIDIA_PROFILE_CHECKPOINT sidecar");
        config.tile = 64;
        config.batch = 1;
        config
    } else {
        deployment_config(64)
    };
    let model = ommatidia::model::build_for_extent(&config, false, INPUT).expect("build");
    let mut session = profiling_session(&model.graph, context.clone(), true);
    model.initialize(&mut session, 1);

    let dispatches = session.plan().dispatches.len();
    // Every group boundary is a global barrier: it drains the pipeline and
    // forbids the next dispatch overlapping the previous one. A chain-shaped
    // network puts one dispatch in most groups.
    let groups = session.num_groups();
    let mut rng = Rng::new(1);
    let cond: Vec<f32> = (0..config.cond_len_for_extent(INPUT))
        .map(|_| rng.normal() * 0.5)
        .collect();
    session.set_input("cond", &cond);

    // Unprofiled wall clock, which is what a caller actually pays.
    for _ in 0..3 {
        session.step();
        session.wait();
    }
    const RUNS: usize = 20;
    let mut wall_samples_ms: Vec<f64> = (0..RUNS)
        .map(|_| {
            let started = std::time::Instant::now();
            session.step();
            session.wait();
            started.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    wall_samples_ms.sort_by(f64::total_cmp);
    let wall_ms = (wall_samples_ms[RUNS / 2 - 1] + wall_samples_ms[RUNS / 2]) * 0.5;

    println!(
        "\n{}x{} -> {}x{}, {} parameters, {dispatches} dispatches in {groups} barrier groups \
         ({:.2} dispatches per group)",
        INPUT[0],
        INPUT[1],
        INPUT[0] * config.scale,
        INPUT[1] * config.scale,
        model.params.iter().map(|p| p.len).sum::<usize>(),
        dispatches as f64 / groups as f64,
    );
    println!("unprofiled wall median: {wall_ms:.2} ms/frame");
    println!(
        "if launch bound, that is {:.1} us per dispatch\n",
        wall_ms * 1e3 / dispatches as f64
    );

    // Meganeura's structured profiler retains repeated per-dispatch samples,
    // plan metadata, allocation totals, and any driver pipeline statistics it
    // can query. Its one-pass-per-dispatch instrumentation is deliberately
    // separate from the normal wall-clock measurement above.
    let profile = meganeura::profiler::capture_session_profile(
        &mut session,
        |_| {},
        meganeura::profiler::CaptureOptions {
            samples: 5,
            unprofiled_median_ms: Some(wall_ms),
            include_pipeline_statistics: true,
        },
    )
    .expect("capture deployment profile");

    println!(
        "profile instrumentation: {:.1}% of normal wall time; timestamps cover {:.1}% of profiled wall time",
        profile.measurement.instrumentation_wall_ratio.unwrap() * 100.0,
        profile
            .measurement
            .timestamped_gpu_share_of_profiled_wall_pct,
    );
    let overhead = profile.measurement.instrumentation_wall_ratio.unwrap();
    assert!(
        overhead <= 1.20,
        "per-dispatch instrumentation changed wall time by {:.1}%; this trace is too perturbative for attribution",
        (overhead - 1.0) * 100.0,
    );
    assert!(
        profile
            .measurement
            .timestamped_gpu_share_of_profiled_wall_pct
            >= 90.0,
        "GPU timestamps explain less than 90% of profiled wall time; do not attribute the missing time to kernels",
    );

    println!("profiled GPU total by kernel family:");
    for family in &profile.families {
        println!(
            "  {:>24}: {:>2} dispatches, {:>6.2} ms ({:>5.1}%)",
            family.family,
            family.dispatch_count,
            family.dispatch_median_sum_ms,
            family.share_of_dispatch_median_sum_pct,
        );
    }

    let output = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../target/ommatidia-deployment-profile.json");
    meganeura::profiler::save_session_profile_json(&output, &profile)
        .expect("save deployment profile");
    println!("structured trace: {}", output.display());
}

/// Measure the host-visible cost of the real integration path: pack the host
/// textures, submit Meganeura's model, unpack to the 1080p output, and wait for
/// completion. The structured trace above intentionally measures the model in
/// isolation; this number prevents that useful diagnostic from being mistaken
/// for the complete post-process latency.
#[test]
#[ignore = "requires a GPU and a trained checkpoint"]
fn end_to_end_1080p_runtime_cost() {
    ommatidia::gpu::warn_if_busy();
    let context = context(false);
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let requested = std::env::var_os("OMMATIDIA_PROFILE_CHECKPOINT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| manifest.join("../runs/raw-restir-b24"));
    let checkpoint = if requested.is_absolute() || requested.with_extension("safetensors").is_file()
    {
        requested
    } else {
        manifest.join("..").join(requested)
    };
    assert!(
        checkpoint.with_extension("safetensors").is_file(),
        "set OMMATIDIA_PROFILE_CHECKPOINT to a trained checkpoint stem"
    );

    const INPUT: [u32; 2] = [960, 540];
    let mut upscaler =
        Upscaler::from_checkpoint_for_extent(context.clone(), &checkpoint, INPUT, 1, 1000)
            .expect("build runtime");
    let input_size = blade_graphics::Extent {
        width: INPUT[0],
        height: INPUT[1],
        depth: 1,
    };
    let (output_width, output_height) = upscaler.output_extent();
    let output_size = blade_graphics::Extent {
        width: output_width,
        height: output_height,
        depth: 1,
    };

    let input_texture = context.create_texture(blade_graphics::TextureDesc {
        name: "profile-input",
        format: blade_graphics::TextureFormat::Rgba32Float,
        size: input_size,
        dimension: blade_graphics::TextureDimension::D2,
        array_layer_count: 1,
        mip_level_count: 1,
        usage: blade_graphics::TextureUsage::RESOURCE | blade_graphics::TextureUsage::COPY,
        sample_count: 1,
        external: None,
    });
    let input_view = context.create_texture_view(
        input_texture,
        blade_graphics::TextureViewDesc {
            name: "profile-input",
            format: blade_graphics::TextureFormat::Rgba32Float,
            dimension: blade_graphics::ViewDimension::D2,
            subresources: &blade_graphics::TextureSubresources::default(),
        },
    );
    let hr_input_texture = context.create_texture(blade_graphics::TextureDesc {
        name: "profile-hr-input",
        format: blade_graphics::TextureFormat::Rgba32Float,
        size: output_size,
        dimension: blade_graphics::TextureDimension::D2,
        array_layer_count: 1,
        mip_level_count: 1,
        usage: blade_graphics::TextureUsage::RESOURCE | blade_graphics::TextureUsage::COPY,
        sample_count: 1,
        external: None,
    });
    let hr_input_view = context.create_texture_view(
        hr_input_texture,
        blade_graphics::TextureViewDesc {
            name: "profile-hr-input",
            format: blade_graphics::TextureFormat::Rgba32Float,
            dimension: blade_graphics::ViewDimension::D2,
            subresources: &blade_graphics::TextureSubresources::default(),
        },
    );
    let output_texture = context.create_texture(blade_graphics::TextureDesc {
        name: "profile-output",
        format: Upscaler::OUTPUT_FORMAT,
        size: output_size,
        dimension: blade_graphics::TextureDimension::D2,
        array_layer_count: 1,
        mip_level_count: 1,
        usage: blade_graphics::TextureUsage::STORAGE | blade_graphics::TextureUsage::COPY,
        sample_count: 1,
        external: None,
    });
    let output_view = context.create_texture_view(
        output_texture,
        blade_graphics::TextureViewDesc {
            name: "profile-output",
            format: Upscaler::OUTPUT_FORMAT,
            dimension: blade_graphics::ViewDimension::D2,
            subresources: &blade_graphics::TextureSubresources::default(),
        },
    );
    let mut inputs = FrameInputs::color_only(input_view, input_view);
    if upscaler.config().reconstruction_base == ReconstructionBase::HighResolutionGuided
        || upscaler.config().demodulate
    {
        inputs = inputs.with_high_resolution_gbuffer(hr_input_view, hr_input_view, hr_input_view);
    }
    let mut encoder = context.create_command_encoder(blade_graphics::CommandEncoderDesc {
        name: "ommatidia-end-to-end-profile",
        buffer_count: 2,
        manual_barriers: false,
    });
    encoder.start();
    encoder.init_texture(input_texture);
    encoder.init_texture(hr_input_texture);
    encoder.init_texture(output_texture);
    let initialized = context.submit(&mut encoder);
    assert!(context.wait_for(&initialized, 30_000).unwrap());

    let run = |upscaler: &mut Upscaler, encoder: &mut blade_graphics::CommandEncoder| {
        encoder.start();
        let started = std::time::Instant::now();
        upscaler.upscale(encoder, &inputs, output_view);
        let completed = context.submit(encoder);
        assert!(context.wait_for(&completed, 30_000).unwrap());
        started.elapsed().as_secs_f64() * 1e3
    };
    // The discrete card leaves its low-power state slowly enough that three
    // frames can still put the median on a clock ramp. A sustained real-time
    // workload is the quantity this test names, so warm for long enough to
    // reach the steady clock before collecting samples.
    const WARMUP_RUNS: usize = 20;
    for _ in 0..WARMUP_RUNS {
        run(&mut upscaler, &mut encoder);
    }
    const RUNS: usize = 40;
    let mut samples: Vec<f64> = (0..RUNS)
        .map(|_| run(&mut upscaler, &mut encoder))
        .collect();
    samples.sort_by(f64::total_cmp);
    let median = (samples[RUNS / 2 - 1] + samples[RUNS / 2]) * 0.5;
    let p90 = samples[(RUNS * 9 / 10).min(RUNS - 1)];
    println!(
        "end to end: {}x{} -> {}x{} median {median:.2} ms, p90 {p90:.2} ms, range {:.2}..{:.2} ms over {RUNS} samples after {WARMUP_RUNS} warmups (pack + model + unpack + submissions)",
        INPUT[0],
        INPUT[1],
        output_width,
        output_height,
        samples[0],
        samples[RUNS - 1],
    );

    let median_of = |mut samples: Vec<f64>| {
        samples.sort_by(f64::total_cmp);
        (samples[RUNS / 2 - 1] + samples[RUNS / 2]) * 0.5
    };
    let pack_ms = median_of(
        (0..RUNS)
            .map(|_| {
                encoder.start();
                let started = std::time::Instant::now();
                upscaler.pack(&mut encoder, &inputs);
                let completed = context.submit(&mut encoder);
                assert!(context.wait_for(&completed, 30_000).unwrap());
                started.elapsed().as_secs_f64() * 1e3
            })
            .collect(),
    );
    let model_ms = median_of(
        (0..RUNS)
            .map(|_| {
                let started = std::time::Instant::now();
                upscaler.run();
                upscaler.session().wait();
                started.elapsed().as_secs_f64() * 1e3
            })
            .collect(),
    );
    let unpack_ms = median_of(
        (0..RUNS)
            .map(|_| {
                encoder.start();
                let started = std::time::Instant::now();
                upscaler.unpack(&mut encoder, &inputs, output_view);
                let completed = context.submit(&mut encoder);
                assert!(context.wait_for(&completed, 30_000).unwrap());
                started.elapsed().as_secs_f64() * 1e3
            })
            .collect(),
    );
    println!(
        "isolated stages (each with its own completion wait): pack {pack_ms:.2} ms, model {model_ms:.2} ms, unpack {unpack_ms:.2} ms"
    );

    let mut host_phases = vec![[0.0f64; 4]; RUNS];
    for phases in &mut host_phases {
        encoder.start();
        let mut started = std::time::Instant::now();
        upscaler.pack(&mut encoder, &inputs);
        context.submit(&mut encoder);
        phases[0] = started.elapsed().as_secs_f64() * 1e3;

        started = std::time::Instant::now();
        upscaler.run();
        phases[1] = started.elapsed().as_secs_f64() * 1e3;

        started = std::time::Instant::now();
        encoder.start();
        upscaler.unpack(&mut encoder, &inputs, output_view);
        let completed = context.submit(&mut encoder);
        phases[2] = started.elapsed().as_secs_f64() * 1e3;

        started = std::time::Instant::now();
        assert!(context.wait_for(&completed, 30_000).unwrap());
        phases[3] = started.elapsed().as_secs_f64() * 1e3;
    }
    let phase_median = |index| median_of(host_phases.iter().map(|sample| sample[index]).collect());
    println!(
        "integrated host phases: pack/submit {:.2} ms, model step call {:.2} ms, unpack/submit {:.2} ms, final wait {:.2} ms",
        phase_median(0),
        phase_median(1),
        phase_median(2),
        phase_median(3),
    );

    upscaler.destroy();
    drop(upscaler);
    context.destroy_texture_view(output_view);
    context.destroy_texture(output_texture);
    context.destroy_texture_view(input_view);
    context.destroy_texture(input_texture);
    context.destroy_texture_view(hr_input_view);
    context.destroy_texture(hr_input_texture);
    context.destroy_command_encoder(&mut encoder);
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
    let context = context(false);
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
