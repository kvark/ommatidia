# ommatidia

Neural frame reconstruction from sparse samples — a portable DLSS replacement.

The network runs through [meganeura](https://github.com/kvark/meganeura) on
Vulkan and Metal, so there is no CUDA, no vendor SDK, and no Python anywhere in
the pipeline. Training data comes from
[blade](https://github.com/kvark/blade): the real-time ReSTIR estimator
provides the input and the canonical path tracer provides the ground truth, so
the network learns to remove the estimator's bias rather than merely to sharpen
its output.

> **Status:** early. The first milestone is single-frame spatial upscaling with
> no temporal context. See [`docs/design.md`](docs/design.md) for the
> formulation and the roadmap.

On 2400 procedural scenes at 128x128 to 256x256, with 360 held out, the
network beats nearest upsampling by **5.12 dB**. Two findings from that run
are worth knowing before reading further:

- **The direct objective beats the diffusion one**, by 5.12 dB against 1.54,
  with the diffusion arm given half again as much training. Worse for
  diffusion, its sampler makes its own output worse the more it is used —
  +3.31 dB at one step, +1.55 at twenty. Conditioning on the renderer's
  G-buffer leaves little for a sampler to explore, so iterating only
  accumulates the model's own error.
- **The G-buffer is worth half a decibel** over colour alone, steadily, for
  channels the renderer produced anyway.
- **It is not real-time yet, but the gap is closing.** A 1080p frame took 656
  ms; two kernel fixes in meganeura — parallelising GroupNorm over the image
  and making the Winograd transforms read contiguously — took it to 122 ms,
  with the weights untouched. The budget is two. See the latency section of the
  design doc for where the rest is.

## Layout

| crate | what it does |
|---|---|
| `ommatidia` | the model, the dataset format, the diffusion schedule, and the host-facing `Upscaler` |
| `ommatidia-data` | renders training pairs with blade |
| `ommatidia-train` | trains a checkpoint and evaluates it |

Both siblings are path dependencies, so a checkout expects `../blade` and
`../meganeura` beside it. The workspace `[patch]` section forces meganeura's
`blade-graphics` and blade's own to resolve to the same crate — without that,
the `Context` a host renderer owns is not the type meganeura's session accepts.

## Try it

```sh
# Render a training set. Pick the GPU with OMMATIDIA_DEVICE_ID if there are several.
cargo run --release -p ommatidia-data -- \
    --out data/train.omd --samples 2400 --lr 128x128 --scale 2

# Train, then reconstruct a crop and write input/nearest/predicted/reference PNGs.
cargo run --release -p ommatidia-train -- \
    --data data/train.omd --steps 8000 --objective direct --batch 8 \
    --lr 3e-4 --lr-final 1e-5 --eval-every 1000 --checkpoint-every 1000 \
    --out runs/first --eval-out runs/first-eval
```

`--objective direct` regresses the frame in one forward pass and is the main
line; `diffusion` shares the same backbone and is kept for comparison, not
because it wins — see the status note above. A checkpoint of either loads into
the same runtime, and `--eval-only` re-scores a finished one without retraining
it, which is how the sampler-step sweep above was measured.

Each sample stores the colour alongside the renderer's own depth, normals,
albedo, specular reflectance, and roughness. That is the structural advantage a
renderer has over photographic super-resolution — it knows where the
silhouettes are rather than having to infer them — and at input resolution it
costs nothing, since the renderer filled those targets on its way to shading.
The trainer takes the plane set from the file header, so `--color-only` gives
the other arm of that ablation without regenerating anything.

## Using it from Blade

Hand over the context you already have. The network executes on your device and
queue — no second context, no external memory import, no cross-device copy.

```rust
let mut upscaler = ommatidia::Upscaler::from_checkpoint(
    context.clone(), "runs/first", /* sampler steps = */ 20, /* timesteps = */ 1000,
)?;

upscaler.upscale(
    &mut encoder,
    &ommatidia::FrameInputs::color_only(color_view, color_view),
    output_view, // Rgba16Float, at upscaler.output_extent()
);
```

Connecting at raw Vulkan instead, so C++ engines can call in, is on the
roadmap; the internals do not change, only the surface.

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
unpack shaders reproduce the CPU batching value for value. That is the one
contract in the system that fails silently — the network trains against the CPU
path, so if the shaders drift, training keeps looking perfect and the renderer
produces garbage.
