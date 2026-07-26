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
    --out data/train.omd --samples 192 --lr 128x128 --scale 2

# Train, then reconstruct a crop and write input/nearest/predicted/reference PNGs.
cargo run --release -p ommatidia-train -- \
    --data data/train.omd --steps 2000 --objective direct \
    --out runs/first --eval-out runs/first-eval
```

`--objective direct` regresses the frame in one forward pass; `diffusion` is
the quality ceiling and needs far more training. Both share a backbone, so a
checkpoint of either loads into the same runtime.

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
