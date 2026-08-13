# ommatidia

[![check](https://github.com/kvark/ommatidia/actions/workflows/check.yml/badge.svg)](https://github.com/kvark/ommatidia/actions/workflows/check.yml)

Neural frame reconstruction from sparse samples — a portable DLSS replacement.

The network runs through [meganeura](https://github.com/kvark/meganeura) on
Vulkan and Metal, so there is no CUDA, no vendor SDK, and no Python anywhere in
the pipeline. Training data comes from [blade](https://github.com/kvark/blade):
a sparse low-resolution path trace provides the primary input and a converged
high-resolution path trace provides ground truth. ReSTIR+SVGF is a comparison
control only; the product does not assume sample reuse or a prior denoiser.

> **Status:** early. The published v0.1 checkpoint is historical and was
> trained on raw ReSTIR. The current path is trained from independent sparse
> paths and uses no sample reuse. Temporal history is designed but not yet
> implemented; see [`docs/temporal.md`](docs/temporal.md).

![Ommatidium architecture: sparse path trace and G-buffer through GPU packing, a multi-scale low-resolution reconstructor, sub-pixel unpacking, and future compact temporal feature feedback](docs/architecture.svg)

The transformer investigation and the resulting complexity/temporal decision are
documented in [`docs/architecture-decision.md`](docs/architecture-decision.md).

[Download the checkpoint](https://huggingface.co/mad-bot/ommatidia) ·
[Training and validation data](https://huggingface.co/datasets/mad-bot/ommatidia)

The Hugging Face v0.1 checkpoint is retained for release provenance; it predates
the independent-path training contract and is not the checkpoint used for the
results below. The current guided b8 checkpoint reconstructs an actual
960×540 → 1920×1080 frame in 7.76 ms median (7.94 ms p90) on an idle Radeon
RX 7900 XT, including pack, model, unpack, and submissions but excluding ray
tracing and display post-processing. An isolated trace attributes 0.76 ms to
the guided pack, 6.99 ms to the network, and 0.12 ms to unpacking.

Upscaling is real today, but narrowly scoped: the published checkpoint and the
current training recipe are **2×**. Runtime frames may be rectangular (the
measured path is 960×540 to 1920×1080), while training uses square crops. There
is no trained 1× denoise-only model, dynamic quality mode, or temporal
supersampling yet; changing the scale means training a checkpoint whose output
head has the corresponding `3 × scale²` channels.

## Results

The path-tracing-first data path produces independent sparse paths and a
4,096-spp reference. It does not run historical weights on an
out-of-distribution input.

The original model was evaluated against the wrong deterministic base. On 76
crops from a separate 128-scene validation set, texel-aligned bilinear scores
26.46 dB / 0.5864 SSIM. A depth/normal/albedo-guided prefilter at input
resolution raises the zero-network reconstruction to 34.08 dB / 0.9473; the
b8 learned residual reaches 34.12 dB / 0.9474. A b24 control reaches 34.15 dB /
0.9476 but costs 2.67× as much end-to-end time. The filter is part of
Ommatidium's reconstruction contract and runs inside the existing pack
stage—there is still no ReSTIR or SVGF upstream.
The complete controlled setup and trace are recorded in the
[`independent-path result`](docs/results/path-trace-guided-2026-08-13.md).

A compute-matched shifted-window transformer tied the convolutional network in
quality and ran 7.5% slower. Its experimental Meganeura window primitive was
removed again, deleting roughly 440 lines rather than carrying unused graph,
autodiff, compiler, and shader surface. The evidence and future temporal gate
are in the [`architecture decision`](docs/architecture-decision.md).

The primary comparison is now ordered by the actual product path. Every image
is a matched 2× reconstruction of the same held-out scene; ReSTIR+SVGF is a
separate control, not Ommatidium's input.

| Sparse paths (4 spp, 64×64 crop) | Bilinear 2× | Ommatidium 2× | ReSTIR+SVGF control, bilinear 2× | Canonical (4,096 spp, 128×128 crop) |
|---|---|---|---|---|
| ![Independent sparse path input](runs/eval-path-trace-4spp-guided-validation/input.png) | ![Bilinear sparse path upsampling](runs/eval-path-trace-4spp-guided-validation/bilinear.png) | ![Ommatidium guided neural reconstruction](runs/eval-path-trace-4spp-guided-validation/predicted.png) | ![Matched Blade ReSTIR plus SVGF control](runs/eval-path-control-svgf/bilinear.png) | ![Matched converged canonical path trace](runs/eval-path-trace-4spp-guided-validation/reference.png) |

### Why the ReSTIR control is darker

Blade's linear-HDR regression shows that no-reuse ReSTIR carries 99.1% and
pairwise reuse 99.0% of a **transport-matched direct** canonical reference.
The remaining dark faces are not a reservoir normalization loss: Blade's
real-time mode stops at the first non-emissive hit, while the clean canonical
target traces secondary paths that bring indirect fill back from the floor and
neighboring objects. NVIDIA likewise treats multi-bounce reuse as a separate
[ReSTIR GI](https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing)
algorithm. The table above therefore labels ReSTIR+SVGF as a direct-light
comparison control and never presents raw ReSTIR as Ommatidium source data.

The checked-in July experiments below used SVGF-filtered inputs. They found the
architecture and kernel optimizations, but their quality figures describe an
upscaler stacked after SVGF, not its replacement. Dataset v2 records that
provenance and the trainer refuses filtered data unless
`--allow-filtered-input` is explicit. Findings worth knowing before reading
further:

- **The direct objective beats the diffusion one**, by 5.12 dB against 1.54,
  with the diffusion arm given half again as much training. Worse for
  diffusion, its sampler makes its own output worse the more it is used —
  +3.31 dB at one step, +1.55 at twenty. Conditioning on the renderer's
  G-buffer leaves little for a sampler to explore, so iterating only
  accumulates the model's own error.
- **The G-buffer is worth half a decibel** over colour alone, steadily, for
  channels the renderer produced anyway.
- **The deployment shape is semi-realtime, but not a 2 ms upscaler.** Two kernel fixes
  in meganeura — parallelising GroupNorm over the image, and making the
  Winograd transforms read contiguously — were worth 5.4x with the weights
  untouched. The rest was not needing the large network at all: a 649k
  parameter model matches the 6.5M one once it is trained out. With the much
  stronger guided base, a 74k-parameter b8 model stays within 0.03 dB of b24
  and runs in 7.76 ms end to end at 960×540 → 1920×1080.
- **Compare shapes at convergence, not at a fixed step count.** A sweep that
  gave every shape 5000 steps ranked them almost exactly wrong, because the
  large ones were the undertrained ones.

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
# Render a training set. Pick the adapter explicitly if there are several.
cargo run --release -p ommatidia-data -- \
    --device-id 0x744c --out data/train.omd \
    --samples 2400 --lr 128x128 --scale 2 \
    --input-frames 1 --canonical-frames 1024

# Train, then reconstruct a crop and write input/nearest/predicted/reference PNGs.
cargo run --release -p ommatidia-train -- \
    --device-id 0x744c --data data/train.omd --steps 8000 \
    --lr 3e-4 --lr-final 1e-5 --eval-every 1000 --checkpoint-every 1000 \
    --out runs/first --eval-out runs/first-eval
```

Direct regression in one forward pass is the default and the main line;
`--objective diffusion` shares the same backbone and is kept for comparison,
not because it wins — see the status note above. A checkpoint of either loads
into the same runtime, and `--eval-only` re-scores a finished one without
retraining it, which is how the sampler-step sweep above was measured.

The generator defaults to one three-bounce sparse path per input pixel and
4,096 eight-bounce paths per reference pixel (with Russian roulette after the
fourth bounce). `--restir-input` and `--svgf-input` exist only for matched Blade
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
albedo, specular reflectance, and roughness. That is the structural advantage a
renderer has over photographic super-resolution — it knows where the
silhouettes are rather than having to infer them — and at input resolution it
costs nothing, since the renderer filled those targets on its way to shading.
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

upscaler.upscale(
    &mut encoder,
    &ommatidia::FrameInputs::from_color_and_blade_gbuffer(
        sparse_path_color,
        renderer.view_gbuffer(),
    ),
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

Raw Vulkan/C integration is planned as a user-space C ABI, not a Vulkan
extension. The ownership, synchronization, and release contract is in
[`docs/integration.md`](docs/integration.md).

The first ABI 1.0 slice is available now in
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
