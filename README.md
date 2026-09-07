# ommatidia

[![check](https://github.com/kvark/ommatidia/actions/workflows/check.yml/badge.svg)](https://github.com/kvark/ommatidia/actions/workflows/check.yml)

Portable neural denoising and 2× reconstruction from sparse path samples.

The network runs through [meganeura](https://github.com/kvark/meganeura) on
Vulkan and Metal, so there is no CUDA, no vendor SDK, and no Python anywhere in
the pipeline. Training data comes from [blade](https://github.com/kvark/blade):
a sparse low-resolution path trace provides the primary input and a converged
high-resolution path trace provides ground truth. ReSTIR+SVGF is a comparison
control only; the product does not assume sample reuse or a prior denoiser.

> **Status:** research prototype. The published v0.3.1 model is spatial;
> temporal 2× training, evaluation and Rust inference are experimental. Current
> recurrent results do not consistently beat the deterministic guide on unseen
> scenes. See the [architecture review](docs/reconstruction-review-2026-09-06.md)
> for the implemented next path and the remaining validation gates.

![Ommatidium architecture: sparse path samples and G-buffer feed a convolutional reconstructor which gathers current samples and optionally mixes a motion-reprojected previous reconstruction only where surface validation accepts it](docs/architecture.svg)

The transformer investigation and the resulting complexity/temporal decision are
documented in [`docs/architecture-decision.md`](docs/architecture-decision.md).

[Download the checkpoint](https://huggingface.co/mad-bot/ommatidia) ·
[Training and validation data](https://huggingface.co/datasets/mad-bot/ommatidia)

The Hugging Face `v0.3.1` revision contains the tuned HR-guided b8 checkpoint used
in the comparison archive; `v0.2.0` retains the low-resolution-only checkpoint. Repeated
960×540 → 1920×1080 runs on a Radeon RX 7900 XT span 8.46–9.30 ms median;
the v0.3.1 trace measured 8.83 ms median and 8.90 ms p90, including pack,
model, unpack, and submissions. Its isolated split was 0.79 ms pack, 7.22 ms
network, and 0.88 ms unpack. Ray tracing, the optional output-resolution
primary-surface pass, and display post-processing are excluded. The
latest experimental temporal b16/r3 checkpoint measures 15.77 ms median and
16.05 ms p90 end to end on the same GPU: 1.73 ms pack, 11.70 ms network, and
2.82 ms unpack when each stage is waited independently, with 142.4 MiB of
recurrent history. Timestamp instrumentation adds 1.3% to ordinary wall time
and covers 93.8% of it; the uninstrumented model median is 11.66 ms.
It is not a release replacement yet: a fresh eight-family real-mesh audit is
0.53 dB behind its own deterministic guide. The exact controls and rejected
capacity/data fixes are in the
[`DLSS 4/5 follow-up`](docs/results/dlss-4-5-probes-2026-09-04.md).
A subsequent equal-frame-exposure control found that causal rollout alone
reduces useful history and regresses both procedural and real-mesh quality; it
is retained as training infrastructure, not promoted as a checkpoint. See the
[`rollout result`](docs/results/causal-rollout-2026-09-04.md).

Upscaling is real today, but narrowly scoped: the published checkpoint and the
current training recipe are **2×**. Runtime frames may be rectangular (the
measured path is 960×540 to 1920×1080), while training uses square crops. There
is no trained 1× denoise-only model or dynamic quality mode. Temporal 2×
training, evaluation, and Rust inference exist; changing the scale means
training a checkpoint whose output head has the corresponding sub-pixel gather
channels.

## Research direction

The [September architecture review](docs/reconstruction-review-2026-09-06.md)
separates verified correctness defects from untested quality hypotheses. The
new opt-in path combines linear-radiance fusion, candidate-aware gates, and a
local residual backbone without image-wide normalization. It requires new
weights; the published checkpoint and default loading behavior are unchanged.

See [design and contracts](docs/design.md),
[measured results and comparison images](docs/results-overview.md), and
[the experiment sequence](docs/reconstruction-review-2026-09-06.md#experiment-order).
No DLSS-parity claim is established by the current evaluation.

## Layout

| crate | what it does |
|---|---|
| `ommatidia` | the model, the dataset format, the diffusion schedule, and the host-facing `Upscaler` |
| `ommatidia-capi` | versioned C ABI; checkpoint discovery today, borrowed-Vulkan inference in progress |
| `ommatidia-data` | renders training pairs with blade |
| `ommatidia-train` | trains a checkpoint and evaluates it |

Both siblings are path dependencies, so a checkout expects `../blade` and
`../meganeura` beside it. The workspace `[patch]` section forces meganeura's
`blade-graphics` and blade's own to resolve to the same crate — without that,
the `Context` a host renderer owns is not the type meganeura's session accepts.

## Try it

```sh
# Render a training set. Add `--device-id ID` only if there are several adapters.
cargo run --release -p ommatidia-data -- \
    --out data/train.omd \
    --samples 2400 --lr 128x128 --scale 2 \
    --input-frames 4 --canonical-frames 1024 --hr-gbuffer

# Train, then reconstruct a crop and write input/nearest/predicted/reference PNGs.
cargo run --release -p ommatidia-train -- \
    --data data/train.omd --steps 8000 \
    --lr 3e-4 --lr-final 1e-5 --eval-every 1000 --checkpoint-every 1000 \
    --out runs/first --eval-out runs/first-eval
```

External glTF catalogs (ABO objects, HSSD interiors, Aria digital twins) expand
the clean corpus without putting the same mesh on both sides of a split. See
[`docs/catalog.md`](docs/catalog.md).

Direct regression in one forward pass is the default and the main line;
`--objective diffusion` shares the same backbone and is kept for comparison,
not because it wins — see the status note above. A checkpoint of either loads
into the same runtime, and `--eval-only` re-scores a finished one without
retraining it, which is how the historical sampler-step sweep was measured.

The generator now defaults to one sparse path per input pixel and 4,096
reference paths per output pixel, both using the same eight-bounce depth
(with Russian roulette after the fourth bounce). `--input-bounces 3`
explicitly reproduces historical truncated-input captures. Every new capture
writes a `.transport.json` sidecar; copied-reference depth remains unverified. `--restir-input` and `--svgf-input` exist only for matched Blade
baselines; SVGF datasets are tagged and need the trainer's explicit
`--allow-filtered-input` override.

Passing `--checkpoint runs/first --preview runs/live` to the generator also
runs that checkpoint from the live `RayTracer` texture views and writes
`*-predicted.png`, exercising the same shared-context path as a host renderer.
For input-sample-count or estimator ablations, `--reference-from existing.omd`
copies the already-rendered high-resolution records. It verifies every copied
sample's G-buffer against the newly rendered scene and camera, so a mismatched
seed cannot silently pair unrelated input and ground truth.

Each sample stores the colour alongside the renderer's own depth, normals,
albedo, specular reflectance, and roughness. With `--hr-gbuffer`, it also stores
those surfaces at output resolution and, for moving sequences, exact
output-resolution motion. That is the structural advantage a renderer has over
photographic super-resolution—it can provide exact silhouettes rather than ask
the upscaler to infer them. Input-resolution planes come from sparse shading;
output-resolution planes may require a separate primary-surface pass in a pure
path tracer.
The sparse radiance and low-resolution G-buffer must describe the same primary
ray. If the path tracer jitters a ray inside the pixel while the G-buffer stays
at its centre, silhouette pixels carry one surface's light beside another
surface's depth and normal, and no denoiser can recover the missing
correspondence. Blade captures therefore disable its internal primary-ray
jitter; an application that jitters its whole camera/G-buffer consistently can
leave that application-level jitter in place.
The trainer takes the plane set from the file header, so `--color-only` gives
the other arm of that ablation without regenerating anything.

## Using it from Blade

Render at the model's input resolution, then hand Ommatidium the input textures
and the graphics context the application already owns. The primary integration
uses the application's independent sparse ray/path colour plus Blade's primary
surface G-buffer. The network executes on the same device and queue—no second
context, external-memory import, cross-device copy, or direct-model CPU
readback.

```rust
path_tracer.render_sparse(&mut encoder, sparse_path_color);
renderer.fill_gbuffer(&mut encoder, debug_config);

let low = renderer.get_surface_size();
let mut upscaler = ommatidia::Upscaler::from_checkpoint_for_extent(
    context.clone(),
    "runs/first",
    [low.width, low.height],
    /* sampler steps = */ 1,
    /* timesteps = */ 1000,
)?;

let inputs = ommatidia::FrameInputs::from_color_and_blade_gbuffer(
    sparse_path_color,
    renderer.view_gbuffer(),
).with_blade_high_resolution_gbuffer(
    high_res_renderer.view_gbuffer(), // after the host's output-resolution primary pass
);

upscaler.upscale(
    &mut encoder,
    &inputs,
    output_view, // Rgba16Float, at upscaler.output_extent()
);

// Record display after the unpack dispatch. Blade applies its normal tone
// mapper to the neural result instead of its internal radiance.
renderer.post_proc_external(
    &mut display_pass,
    output_view,
    debug_config,
    post_proc_config,
    &[],
    &[],
);
```

### Resolution contract

Runtime frames are not limited to the checkpoint's square training tile.
`from_checkpoint_for_extent` accepts a rectangular input extent and produces
exactly `scale` times that width and height. Each input axis must be divisible
by `2^(levels - 1)` and must remain at least two texels after that
downsampling. For the published three-level, 2× checkpoint this means any
width and height divisible by 4 and no smaller than 8; 960×540 → 1920×1080
is one valid example.

The scale is part of the checkpoint, so a 2× checkpoint cannot be requested
at an arbitrary scale. Meganeura also prepares the graph for one input extent:
create or cache one `Upscaler` per size, and use a fresh instance or call
`reset_history` after a resize. All low-resolution inputs must match the
selected input extent. The output and any required high-resolution G-buffer
planes must match the scaled extent. Device texture limits, memory, and work
that grows roughly with input pixel count are the practical upper bounds.

Raw Vulkan/C integration is planned as a user-space C ABI, not a Vulkan
extension. The ownership, synchronization, and release contract is in
[`docs/integration.md`](docs/integration.md).

The ABI 1.1 checkpoint-inspection slice is available now in
[`include/ommatidia.h`](include/ommatidia.h): it links from plain C and
inspects a checkpoint's exact graph/resource contract without enumerating a
GPU. [`examples/c/inspect.c`](examples/c/inspect.c) is the conformance example.
GPU execution is intentionally not exported yet; adding an entry point that
secretly created a second device would violate the integration contract.

Note the sampler still walks the chain on the host, one roundtrip per step, so
a diffusion checkpoint is far from a frame budget. A direct one is a single
forward pass, which is the other reason it is the main line.

## A long run

`scripts/curriculum.sh` drives one unattended, serialising its runs so they do
not contend, and waiting rather than failing if the device is short of memory.
`scripts/curve.py` lines the resulting runs up by step. Calibrate the step rate
first: it is set by whatever else is using the GPU rather than by model size,
and the cosine schedule needs the total step count up front.

## Tests

```sh
cargo test                                  # everything that needs no GPU
cargo test -- --ignored                     # the GPU tests
```

The GPU tests are worth knowing about: `gpu_runtime` checks that the pack and
unpack shaders reproduce the CPU batching value for value and SSIM-checks a
deterministic non-zero-network PNG on LavaPipe. That is the one
contract in the system that fails silently — the network trains against the CPU
path, so if the shaders drift, training keeps looking perfect and the renderer
produces garbage.
