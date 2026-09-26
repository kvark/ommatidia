# ommatidia

[![check](https://github.com/kvark/ommatidia/actions/workflows/check.yml/badge.svg)](https://github.com/kvark/ommatidia/actions/workflows/check.yml)

Neural reconstruction of sparse path-traced frames, in Rust on
[Meganeura](https://github.com/kvark/meganeura) and
[Blade](https://github.com/kvark/blade).

One model: a **recurrent, lobe-separated radiance-residual U-Net**. It reconstructs
diffuse illumination and specular radiance, then applies observed material albedo
and emission. At 2x scale it has 188,160 parameters. Inputs include 1-spp
low-resolution radiance, motion/jitter and output-resolution primary surfaces.

## Measured results

Fresh-scene audit: **256 frames**, 1-spp 128×128 input → 256×256 output,
16-frame causal sequences, 4,096-spp references. The same model was fine-tuned
for 4,000 updates at this resolution. Development selected the final checkpoint
before the audit. Both checkpoints receive identical frames and run their own history.

| Audit | PSNR, previous → now ↑ | SSIM, previous → now ↑ | Temporal error ↓ |
|---|---|---|---|
| Procedural scenes | 28.18 → **28.65 dB** | 0.823 → 0.837 | −7.7% |
| Held-out object scenes | 28.91 → **29.60 dB** | 0.797 → 0.816 | −7.8% |

The comparison is against the previous **trained checkpoint**, not just the fixed
guide. PSNR improves on 246/256 frames; the worst regression is 0.26 dB.
The unchanged, previously published audit also improves: **28.24 → 28.67 dB**
(procedural) and **29.04 → 29.79 dB** (objects).

Mean energy bias on the fresh audit is now −0.40% / approximately 0%, down from
+1.34% / +2.58%. Specular noise, blotches and missing detail remain.
[Full metrics, protocol, hashes and limitations](docs/results/README.md).

Sequence 0, frame 7 in each set, chosen before evaluation. Native 256×256
outputs; identical display transform, no retouching.

| Previous checkpoint | Fine-tuned checkpoint | Reference |
|---|---|---|
| <img src="docs/results/procedural-previous.png" alt="Procedural scene: previous trained checkpoint" width="256" height="256"> | <img src="docs/results/procedural-model.png" alt="Procedural scene: fine-tuned residual U-Net" width="256" height="256"> | <img src="docs/results/procedural-reference.png" alt="Procedural scene: 4096-spp reference" width="256" height="256"> |
| <img src="docs/results/abo-previous.png" alt="Object scene: previous trained checkpoint" width="256" height="256"> | <img src="docs/results/abo-model.png" alt="Object scene: fine-tuned residual U-Net" width="256" height="256"> | <img src="docs/results/abo-reference.png" alt="Object scene: 4096-spp reference" width="256" height="256"> |

Object assets: Amazon.com, [ABO / CC BY 4.0](docs/catalog.md). Authored objects
replace the extra primitives; measured visible coverage is 4.2–40.5% per audit
frame. Scene seeds and asset families are disjoint from training and development;
these are fresh scenes using the same four audit families as the previous report.

## Build and train

Rust 1.92+. Place `ommatidia` and `blade` in sibling directories.
Cargo pins Blade to `fbb4f28` and Meganeura to `ee3aea4`; only Blade has a local
sibling override. Keep Blade's shader directory at the same revision.
Naga also needs the tracked [workgroup-layout correction](patches/README.md);
the preparation command below installs it into the ignored build directory.

```sh
python3 scripts/prepare-naga.py
cargo +1.92.0 test --workspace --locked
cargo build --release --workspace
cargo run --release -p ommatidia-train --bin transport -- \
  --data data/train.omd --eval-data data/dev.omd --out runs/model \
  --steps 4000 --channels 16 --unroll 2 --lr 0.0003 --eval-every 500
```

[Architecture and runtime](docs/design.md) ·
[Data, metrics and reproduction](docs/evaluation.md) ·
[External asset capture](docs/catalog.md)

## Scope

This is a reconstruction research prototype, not demonstrated DLSS parity.
The training corpus is still small and synthetic; game traces, broad interiors,
and matched external-denoiser comparisons remain missing.

The old diffusion, kernel-selector, field/relighting, C ABI, trainers and result
galleries were removed. They remain in Git at `b838674`. There are no hidden
architecture switches or compatibility interpretations of their weights.
The retained runtime is `ommatidia::transport::native::Native`, config version 3.

Numerical CPU/GPU, gradient, recurrence and reload checks pass on the tested
adapters. With the pinned compiler correction, debug GPU checks pass on RADV and
LavaPipe with zero validation errors; debug capture/train/reload also pass.
This fixes the reproduced Workgroup-array layout failure, not all possible
compiler bugs. [Quality-sprint evidence and remaining gates](docs/quality-week.md).
