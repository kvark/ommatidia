# ommatidia

[![check](https://github.com/kvark/ommatidia/actions/workflows/check.yml/badge.svg)](https://github.com/kvark/ommatidia/actions/workflows/check.yml)

Portable neural reconstruction from sparse path samples. Rust,
[Meganeura](https://github.com/kvark/meganeura), Vulkan and Metal.

**Research prototype—not a demonstrated DLSS replacement.** The active model
reconstructs diffuse illumination and specular radiance separately, selects
among multiscale and reprojected candidates, and preserves known surface albedo
and emission. Training includes confidence supervision and short radiance BPTT.
The published v0.3.1 checkpoint remains supported by the legacy runtime.

## Build

Place `ommatidia`, `blade` and `meganeura` in sibling directories. Use the pins
in `.github/workflows/check.yml`: Blade `b208f3b` and Meganeura `230cab0`.
Blade 0.9 is intentionally not mixed with Meganeura's 0.8 graphics ABI.

```sh
cargo +1.92.0 test --workspace --locked
cargo +1.92.0 run -p ommatidia-train --bin transport -- --help
bash benchmarks/transport-lavapipe.sh
```

## Quality and integration

See [the architecture and input contract](docs/design.md),
[the quality harness](benchmarks/README.md), and
[the shared-transport direction](docs/shared-transport.md).
`ommatidia::transport::native::Native` exposes the new Rust GPU path;
`process` adds synchronous upload/readback for tests. Existing `Upscaler`/C ABI
clients retain legacy checkpoint behavior.

[Published weights](https://huggingface.co/mad-bot/ommatidia) ·
[Historical datasets](https://huggingface.co/datasets/mad-bot/ommatidia) ·
[Historical results](docs/results-overview.md)

Old captures may have mismatched path depth or camera-motion history. The new
trainer requires matched transport provenance and disjoint scene/family splits;
do not relabel old data or edit old checkpoint sidecars into the new model.
Historical training commands require their recorded Git revisions.
