# ommatidia

[![check](https://github.com/kvark/ommatidia/actions/workflows/check.yml/badge.svg)](https://github.com/kvark/ommatidia/actions/workflows/check.yml)

Neural frame reconstruction from sparse samples — a portable DLSS replacement.

The network runs through [meganeura](https://github.com/kvark/meganeura) on
Vulkan and Metal, so there is no CUDA, no vendor SDK, and no Python anywhere in
the pipeline. Training data comes from [blade](https://github.com/kvark/blade):
a sparse low-resolution path trace provides the primary input and a converged
high-resolution path trace provides ground truth. ReSTIR+SVGF is a comparison
control only; the product does not assume sample reuse or a prior denoiser.

> **Status:** early. Spatial reconstruction is one learned gather over the raw
> path samples. The latest experimental recurrent model is trained on matched
> **1-spp, four-frame** sequences: it is **+1.59 dB per frame over deterministic
> accumulation and reduces motion-compensated 16x16-block fluctuation about
> 9x**, while retaining 52% rather than 34% of reference detail. The training
> and evaluation path is implemented; the native API deliberately rejects this
> checkpoint until GPU reprojection, validity, and history ownership land. The
> published v0.3.1 checkpoint therefore remains the spatial runtime default.
> See the [`temporal result`](docs/results/temporal-validity-1spp-2026-08-22.md)
> and [`runtime plan`](docs/temporal.md).

![Ommatidium architecture: sparse path samples and G-buffer feed a convolutional reconstructor which gathers current samples and optionally mixes a motion-reprojected previous reconstruction only where surface validation accepts it](docs/architecture.svg)

The transformer investigation and the resulting complexity/temporal decision are
documented in [`docs/architecture-decision.md`](docs/architecture-decision.md).

[Download the checkpoint](https://huggingface.co/mad-bot/ommatidia) ·
[Training and validation data](https://huggingface.co/datasets/mad-bot/ommatidia)

The Hugging Face `v0.3.1` revision contains the tuned HR-guided b8 checkpoint used
below; `v0.2.0` retains the low-resolution-only checkpoint. Repeated
960×540 → 1920×1080 runs on a Radeon RX 7900 XT span 8.46–9.30 ms median;
the v0.3.1 trace measured 8.83 ms median and 8.90 ms p90, including pack,
model, unpack, and submissions. Its isolated split was 0.79 ms pack, 7.22 ms
network, and 0.88 ms unpack. Ray tracing,
the optional output-resolution primary-surface pass, temporal history, and
display post-processing are excluded. The range is reported because amdgpu's load
counter became intermittently unavailable during the final trace; the harness
now reports that condition rather than silently calling the device idle.

Upscaling is real today, but narrowly scoped: the published checkpoint and the
current training recipe are **2×**. Runtime frames may be rectangular (the
measured path is 960×540 to 1920×1080), while training uses square crops. There
is no trained 1× denoise-only model or dynamic quality mode. Temporal 2×
training/evaluation exists, but it is not in the native runtime yet; changing
the scale means training a checkpoint whose output head has the corresponding
sub-pixel gather channels.

## Results

The path-tracing-first data path produces independent sparse paths and a
4,096-spp reference. It does not run historical weights on an
out-of-distribution input.

On 128 non-overlapping crops from a separate seed-10000 validation set,
texel-aligned bilinear scores 26.51 dB / 0.5776 SSIM. The v0.3 filter reaches
34.61 dB / 0.9543 before its learned correction. A held-out sweep tuned the
same fixed-cost filter to 34.72 dB / 0.9574; the matching b8 residual reaches
34.74 dB / 0.9575. The
network still runs wholly at low resolution; the extra guide is contained in
Ommatidium's existing unpack dispatch, with no ReSTIR or SVGF upstream and no
new Meganeura graph operation or shader group. Filter coefficients are stored
in the checkpoint, so the runtime continues to interpret v0.3.0 exactly as it
was trained while v0.3.1 opts into the tuned profile.
A matched 1-spp arm reaches 31.74 dB / 0.9215 SSIM with the HR guide, but a b8
network trained specifically on that distribution adds less than 0.005 dB.
That closes the “larger static network” branch: temporal history must supply
new samples before more model capacity is justified.
On the current common 128-crop score, a perfect static-history oracle makes
that value concrete: the tuned guide rises from 32.17 dB at one accumulated
sample to 33.76 at two, 34.72 at four, 35.50 at eight, and 35.89 at sixteen.
Four aligned 1-spp frames are worth +2.55 dB before motion and rejection
losses—over one hundred times the b8 static residual's gain.
The first moving-camera oracle measured the important failure mode:
motion-only accumulation ghosts and falls from 31.17 to 29.09 dB, whereas
depth/normal/albedo rejection reaches 32.33 dB / 0.9301 SSIM. Only 2.7% of
history pixels are rejected. Validity is therefore an explicit input to the
temporal model, not something a larger backbone should have to infer.
The sequence-aware trainer now preserves whole-sequence splits and tests a
safe temporal model: accumulated colour remains the deterministic base, while
current RGB, confidence, and the exact guided base are ordinary U-Net input
channels. Predicting three low-resolution colour corrections reaches 32.28 dB
on a separate moving-sequence set versus 32.26 dB for rejected history alone;
a 4×-compute b16 control reaches 32.30 dB. The small learned increment is real
but not release-worthy, so the spatial v0.3.1 runtime remains the default while
motion diversity and temporal losses are expanded. See the
[`temporal model result`](docs/results/temporal-low-color-2026-08-14.md).
The follow-up
[`curved-motion gates`](docs/results/temporal-motion-gates-2026-08-15.md)
add a motion-compensated stability metric. Curved-motion training plus one
history-deviation channel improves the independent set by 1.27 dB spatially
and 0.57 dB temporally at essentially the same b8 cost; the next larger step is
joint sequence training rather than a wider or transformer backbone.
The subsequent
[`object-motion gate`](docs/results/temporal-object-motion-2026-08-15.md)
animates independent Blade objects and scores moving pixels directly. A mixed
camera/object 4,000-step b8 run reaches 34.52 dB on object-only motion and
34.21 dB on camera-only motion, improving the prior checkpoint by 0.06 and
0.02 dB at identical inference cost while retaining temporal stability. A
velocity-channel ablation was worse and its code was removed.
The complete controlled setup and trace are recorded in the
[`independent-path result`](docs/results/path-trace-guided-2026-08-13.md).

A compute-matched shifted-window transformer tied the convolutional network in
quality and ran 7.5% slower. Its experimental Meganeura window primitive was
removed again, deleting roughly 440 lines rather than carrying unused graph,
autodiff, compiler, and shader surface. The evidence and future temporal gate
are in the [`architecture decision`](docs/architecture-decision.md).

The primary comparison is ordered by the actual product path. Every image is a
matched **2×** reconstruction of the same held-out scene; ReSTIR+SVGF is a
separate control, not Ommatidium's input. OIDN denoises at input resolution and
then receives the same texel-centre 2× bilinear reconstruction—it is not being
presented as an OIDN upscaler.

| scene | Sparse paths (4 spp, 128×128) | Ommatidium 2× | OIDN High + bilinear 2× | ReSTIR+SVGF control + bilinear 2× | Canonical (4,096 spp, 256×256) |
|---|---|---|---|---|---|
| canopy shadow | ![Sparse canopy-shadow input](docs/comparison-suite/canopy-shadow/input.png) | ![Ommatidium canopy-shadow reconstruction](docs/comparison-suite/canopy-shadow/ommatidium.png) | ![OIDN High canopy-shadow denoise](docs/comparison-suite/canopy-shadow/oidn-input-high.png) | ![ReSTIR plus SVGF canopy-shadow control](docs/comparison-suite/canopy-shadow/restir-svgf.png) | ![Canonical canopy-shadow reference](docs/comparison-suite/canopy-shadow/canonical.png) |
| local light | ![Sparse local-light input](docs/comparison-suite/local-light/input.png) | ![Ommatidium local-light reconstruction](docs/comparison-suite/local-light/ommatidium.png) | ![OIDN High local-light denoise](docs/comparison-suite/local-light/oidn-input-high.png) | ![ReSTIR plus SVGF local-light control](docs/comparison-suite/local-light/restir-svgf.png) | ![Canonical local-light reference](docs/comparison-suite/local-light/canonical.png) |
| hard shadow | ![Sparse hard-shadow input](docs/comparison-suite/hard-shadow/input.png) | ![Ommatidium hard-shadow reconstruction](docs/comparison-suite/hard-shadow/ommatidium.png) | ![OIDN High hard-shadow denoise](docs/comparison-suite/hard-shadow/oidn-input-high.png) | ![ReSTIR plus SVGF hard-shadow control](docs/comparison-suite/hard-shadow/restir-svgf.png) | ![Canonical hard-shadow reference](docs/comparison-suite/hard-shadow/canonical.png) |

The six-scene suite confirms both the progress and the shortcoming visible in
those images. Ommatidium averages 29.26 dB, 1.21 dB above OIDN High, retains 77%
of canonical detail rather than OIDN's 49%, and has far lower relative error in
dark regions. OIDN leads SSIM (0.873 versus 0.841) because it is smoother.
Ommatidium still carries only 93.5% of canonical mean luminance and leads OIDN
by just 0.36 dB on the new block-average low-frequency score: the broad mottling
is real. The complete images, per-scene CSV, speed trace, rejected first fixes,
and reproduction command are in the
[`curated OIDN result`](docs/results/curated-oidn-2026-08-22.md) and
[`benchmark harness`](benchmarks/README.md).

### Temporal history removes the broad fluctuation

The remaining spatial mottling is not recoverable from a different loss over
the same noisy frame. The recurrent experiment instead mixes the previous
reconstruction after the current-frame gather, using current-to-previous motion
and explicit per-sub-pixel surface validity. Rejected history hard-closes the
gate; it can no longer become accidental black radiance.

On 756 crops from 63 unseen four-frame sequences, with a fresh 1-spp path each
frame, the model reaches 27.62 dB versus 26.04 dB for accumulated HR guidance.
Its motion-compensated temporal delta is +4.14 dB better, and its 16x16-block
delta is +9.54 dB better: broad frame-to-frame fluctuation falls from 0.000578
to 0.000064, about ninefold. Quality improves with recurrent age rather than
drifting. These are training/evaluation results; native temporal deployment is
the next implementation boundary.

| accumulated HR guide | recurrent Ommatidium | 1,024-spp reference |
|---|---|---|
| ![Accumulated guide with broad horizontal illumination bands](docs/temporal-low-frequency/hr-guided.png) | ![Temporal Ommatidium output with the broad bands suppressed](docs/temporal-low-frequency/predicted.png) | ![High-sample reference for the temporal validation crop](docs/temporal-low-frequency/reference.png) |

The full data recipe, radius gate, metrics, rejected initialization, and 4-spp
control are in the
[`1-spp temporal result`](docs/results/temporal-validity-1spp-2026-08-22.md).

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

### Reconstruction is one operation

The learned residual over a deterministic filter was worth 0.02 dB, and the
frame was visibly soft. Neither was a limit of the network. Asking a
least-squares model for the residual of a filter asks it to predict that
filter's error, which is dominated by the noise the renderer happened to draw
and whose conditional mean is almost exactly zero — so a 74k model and a 649k
model agreed, and the conclusion drawn was that spatial reconstruction had
saturated.

An oracle that only picks which of the *already shipped* filter footprints to
use per texel is worth +0.83 to +2.23 dB at 4 spp, bracketed by forcing the
choice constant over 4×4 and 16×16 blocks so it cannot exploit the noise draw.
The tuned global filter width is the right one for 14.7% of texels.

`Prediction::SubpixelKernel` therefore has the network emit gather weights over
the input samples, one set per output sub-pixel, with nothing filtered
beforehand and no base to correct. The output is a convex combination of
measured radiance, so it cannot overshoot, invent energy, or emit the black
pixels a rejected bilateral gather produces. Measured on the same held-out set:

| reconstruction | PSNR | SSIM | relMSE | detail |
|---|---:|---:|---:|---:|
| texel-centre bilinear | 26.51 dB | 0.5776 | 0.08941 | 394% |
| HR guide 5×5 (v0.3.1 base) | 34.87 dB | 0.9579 | 0.01043 | 63% |
| kernel b8 r2 | 35.38 dB | 0.9338 | 0.00815 | 99% |
| **kernel b16 r2** | **36.61 dB** | 0.9514 | **0.00595** | **86%** |

Capacity matters again: b8 → b16 is worth +1.23 dB, where under the residual
parameterisation an 8.8× larger model was worth 0.03 dB.

Two metrics were added to see any of this. PSNR and SSIM cannot: a box blur
that removes an eighth of the frame's remaining detail costs 0.06 dB and
0.002 SSIM, and *improves* the same error measured in display space. Detail
retention is displayed gradient energy as a fraction of the canonical frame's,
and relMSE keeps a bright region from deciding the score alone.

### The scenes had to change too

Blade's fallback environment is a white 1×1 texture, so an open scene is a
uniform furnace in which nothing can be in shadow: none of the old validation
pixels fall below a displayed luminance of 0.10. Every material was a constant
colour, so albedo demodulation — one of the larger wins in a production
denoiser — measured as 0.01 dB. `--canopy`, `--textures`, `--gloss` and
`--ground-patches N` fix that, all off by default. Textures go through the same
BC1 asset path a glTF material's base colour does.

Retrained on those scenes, both architectures for 8,000 steps, scored on a
separate 128-scene set:

| external validation | PSNR | SSIM | relMSE | worst crop | detail |
|---|---:|---:|---:|---:|---:|
| HR guide 5×5 | 28.36 dB | 0.8736 | 0.187 | 5.80 | 47% |
| residual b8 | 29.96 dB | 0.8704 | 17.204 | **171.31** | 61% |
| kernel b16 r2 | **30.26 dB** | 0.8563 | 0.044 | 0.47 | 67% |
| **kernel b16 r2, `--demodulate`** | 30.21 dB | **0.8840** | **0.032** | **0.29** | **83%** |

This corrects something. The learned residual added 0.02 dB on the old scenes
and 1.60 dB here, so most of that null result was the data rather than the
parameterisation — a residual over a filter has nothing to predict when the
truth inside every object is smooth. But it is unbounded, and its relative
error is 92× worse than the base it corrects, with a worst crop of 171. PSNR
reports the same model as a 1.60 dB win, because the failure is in the dark
third of the frame where PSNR has almost no weight.

The kernel model's worst crop is 0.47 — better than the deterministic base's
own worst case. That is the formulation rather than the training: the output is
a convex combination of radiance the renderer measured, so there is no
arithmetic by which it invents any.

Demodulating the albedo — dividing it out before the gather and multiplying the
exact output-resolution one back after — is level on PSNR and better on
everything else, and is the only arm that beats the deterministic base on SSIM
too. The offset that bounds how far a pixel may be rescaled matters more than it
sounds: at 0.05 it allows a factor of twenty, which lands pixels where the
compressed gather has no precision left and costs 1.5 dB. At 0.25 it allows
four.

Three sweeps settled the rest. Nine taps cannot denoise four samples per pixel;
twenty-five is the knee and forty-nine buys 0.32 dB for twice the head. A 1×1
head looked like free money and costs 0.73 dB — and saves less than it appears,
since 22% of the arithmetic is 8.5% of the frame. They have one thing in common:
every knob is really "how much smoothing", PSNR always prefers more of it, and
detail retention crossing 100% is what marks a variant that failed to denoise.
Every underperforming arm in this work sat above it.

Reprojected history extends the formulation naturally — the accumulated estimate
is one more tap, and how much to trust it is one more weight — but a single-frame
objective never asks for it. The learned bias for that tap did not move off its
initialisation in 8,000 steps: twenty-five current-frame taps already denoise a
4-spp frame, so history buys nothing a per-frame squared error can see, and the
sequence flickered while every individual frame improved.

Stability had to go into the objective. The temporal metric rearranges into a
squared error against a host-assembled target, so it costs the graph six
operations rather than a per-pixel gather it cannot express, with the previous
frame's answer coming from a detached copy of the network. Temporal error then
moves monotonically with its weight, and at weight 1 the reconstruction gives up
0.31 dB of PSNR for 0.95 dB of stability while also improving relative error,
worst case, and SSIM — 1.49 dB better per frame than the deterministic
accumulation, five times better on relative error, sixteen times better in its
worst crop, and within 0.17 dB on stability.

Weighting that term toward moving pixels seemed obvious and is wrong: the target
is least trustworthy exactly where motion is largest, so it amplifies target
error rather than a deficiency, monotonically. Moving pixels remain the
unsolved case. See the
[`single-operation result`](docs/results/monolithic-kernel-2026-08-15.md).

SSIM should not be read on this content. Split by crop brightness, the darkest
third scores 0.9810 and the brightest 0.7905: C2 is absolute, and where the
local variance falls below it the structure term reports agreement whatever the
images did. `metrics::ssim` is unchanged so published figures still compare, and
the diagnostic now reports the split.

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
  stronger low-resolution guided base, a 74k-parameter b8 model stays within
  0.03 dB of b24 and runs in 7.76 ms end to end at 960×540 → 1920×1080. The
  sharper v0.3 HR-guided path adds roughly one millisecond in unpack.
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
    --input-frames 4 --canonical-frames 1024 --hr-gbuffer

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
albedo, specular reflectance, and roughness. With `--hr-gbuffer`, it also stores
output-resolution depth, normal, and albedo. That is the structural advantage a
renderer has over photographic super-resolution—it can provide exact
silhouettes rather than ask the upscaler to infer them. Input-resolution planes
come from sparse shading; output-resolution planes may require a separate
primary-surface pass in a pure path tracer.
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
