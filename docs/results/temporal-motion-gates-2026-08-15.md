# Temporal motion and fusion gates, 2026-08-15

## Question

What should change after the first learned four-frame model added only 0.02 dB:
network width, more history, more statistics, or the training objective?

## Harder validation

The generator now supports `--random-camera-motion F`. Each sequence gets a
deterministic non-axis-aligned path with vertical drift and curvature; frame
zero remains the exact base pose. This retains reproducibility while covering
two-axis motion and more varied disocclusions than the original world-X-only
set.

The independent validation set contains 32 four-frame sequences from seed
23000, one sparse path per low-resolution pixel, 256 canonical frames per
target, and 0.05 world-unit nominal camera translation per frame. The merged
b8 checkpoint scores:

| reconstruction | MSE | PSNR | SSIM |
|---|---:|---:|---:|
| four-frame HR guide | 0.000510 | 32.92 dB | 0.9366 |
| merged b8 low-colour model | **0.000507** | **32.95 dB** | **0.9367** |

The +0.03 dB spatial gain transfers, but both have motion-compensated delta
MSE 0.000265. The old learned correction provides **+0.00 dB temporal gain**.

Held-out evaluation now reports that delta metric directly. It warps the
previous prediction and reference with current-to-previous motion, excludes
surface-rejected and out-of-crop history, and compares predicted change with
reference change in compressed radiance space. This does not reward a stable
but biased result the way a prediction-only flicker metric would.

## Curved-motion training result

The controlled training set contains 256 independently seeded curved-motion
sequences. Both models use the same b8 U-Net, seed 31, 2,000 steps, and
four-frame history. `variance` adds the standard deviation of accepted
compressed luminance as one ordinary input channel. It adds 72 parameters,
about 0.1 GFLOP at 1080p, and no graph operation or dispatch.

On the independent seed-23000 set:

| model | parameters | MSE | PSNR | SSIM | delta MSE | spatial gain | temporal gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| four-frame HR guide | 0 | 0.000510 | 32.92 dB | 0.9366 | 0.000265 | — | — |
| curved b8, basic history | 73,664 | 0.000389 | 34.10 dB | 0.9443 | 0.000236 | +1.17 dB | +0.50 dB |
| curved b8, history deviation | 73,736 | **0.000381** | **34.19 dB** | **0.9447** | **0.000232** | **+1.27 dB** | **+0.57 dB** |

The whole-sequence internal holdout agrees: basic reaches 34.15 dB and delta
MSE 0.000236 (+0.93/+0.54 dB over its guide), while variance reaches 34.20 dB
and 0.000232 (+0.97/+0.62 dB). The selected feature is therefore small but
repeatable on both splits.

During this experiment the trainer exposed a measurement bug: its final
in-process evaluator was a separate inference graph and did not receive the
final parameters. The checkpoint was saved correctly, but the immediately
following score measured initialization or the last periodic sync. Reloading
the saved checkpoints produced the results above. The trainer now synchronizes
parameters before its final score, preventing future architecture decisions
from using stale weights.

## Rejected blend target

The blend experiment constrained every output between observed current and
history radiance, with zero output preserving the unbiased accumulated mean.
After reloading the saved checkpoint, it still scored 31.52 dB / 0.8832 and
delta MSE 0.000383: 1.41 dB spatially and 1.61 dB temporally below the guide.
A per-frame canonical-derived blend coefficient is a noisy, ill-conditioned
label; fitting it teaches frame-specific Monte Carlo outcomes rather than a
reusable gate. That target and all implementation code were removed.

## History length

An independent eight-frame curved set was evaluated twice with identical
records and rejection, changing only the history cap:

| history cap | MSE | PSNR | SSIM |
|---|---:|---:|---:|
| four samples | **0.000430** | **33.67 dB** | 0.9391 |
| eight samples | 0.000430 | 33.66 dB | **0.9400** |

Eight samples gain 0.0009 SSIM but no radiometric accuracy. Under motion, four
valid samples already reach the bias floor of the deterministic guide; simply
extending recurrence does not unlock the 37.48 dB canonical-low ceiling. The
oracle accepts an optional history cap as its fourth numeric argument so this
can be repeated on future object-motion data:

```sh
cargo run --release -p ommatidia-train --bin temporal-oracle -- \
  DATA.omd 0.01 0.9 0.04 4
```

## Decision

Select the b8 low-colour model with history deviation for the next temporal
checkpoint. Do not widen it, switch to a transformer, use the blend target, or
extend the history cap: the retained one-channel change beats all of those
directions at essentially unchanged cost.

The next larger architecture change remains **sequence-level training**:
paired consecutive outputs and a motion-compensated temporal loss, with
radiometric loss retained so blur cannot win by being stable. That requires a
differentiable warp (or an equivalent paired formulation) in the training
graph. It should be designed as one reusable Meganeura primitive rather than
accumulating denoiser-specific shader groups. These weights should not replace
the published spatial checkpoint until the native runtime has its temporal
history path.

Before that experiment, data still needs independently moving objects and
animated specular response. Camera-only motion is now varied enough to expose
flicker, but it cannot validate object-motion disocclusion or reactive masks.
