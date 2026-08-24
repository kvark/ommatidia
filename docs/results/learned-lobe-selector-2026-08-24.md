# Pooled lobe-scale selector — 2026-08-24

## Question

Can a tiny block-local classifier recover the measured per-lobe filter-scale
oracle without adding a spatial model or native runtime path?

The 32-sequence, 16-frame lobe-scale audit was split by whole scene: sequences
0–19 trained the selector, 20–23 selected only its convex-mixture temperature,
and 24–31 were untouched evaluation. Each model saw 76,800 retained 8×8-output
blocks per lobe. References remained independent 16,384-spp captures.

The deployable features contained candidate luminance and gradients, adjacent
filter-scale differences, depth/normal/albedo discontinuities, roughness,
history confidence, and visible history deviation. Two follow-ups added
per-scale maxima, per-lobe projection-phase variance, soft scale targets, and a
3×3 block neighborhood. No feature read a reference or oracle label at
evaluation time.

## Held-out result

| reconstruction | PSNR | SSIM | relMSE | detail | energy | LF PSNR |
|---|---:|---:|---:|---:|---:|---:|
| fixed split-lobe filter | 30.10 dB | 0.9221 | 0.19868 | 73.7% | 0.995 | 34.15 dB |
| local hard-target mixture | 30.57 dB | 0.9261 | **0.08780** | 74.8% | 0.994 | 35.03 dB |
| soft target + phase/max features | 30.55 dB | 0.9255 | 0.09397 | 74.6% | 0.994 | 35.02 dB |
| spatial-context hard target | **30.61 dB** | **0.9263** | 0.19125 | **74.8%** | 0.994 | **35.08 dB** |
| 8×8 per-lobe oracle | 31.84 dB | 0.9310 | 0.18803 | 77.4% | 0.995 | 37.66 dB |

The best model recovers 29% of the oracle PSNR gap, 47% of its SSIM gap, and
26% of its low-frequency gap. The gate required at least half. Convex mixing
is valuable—the first model more than halves dark-region relative error—but
the pooled representation cannot identify enough of the broad noise.

## Decision

Reject the pooled MLP and remove its implementation. It did not earn a GPU
contract, shader, checkpoint field, or permanent experiment framework.

The next probe preserved spatial arrangement: a small convolutional head
predicted per-output-pixel diffuse and specular mixtures over the same six
candidates and trained through the composed reconstructed image. It required
no new Meganeura operation or shader group, but it also failed to beat the
fixed estimator. That follow-up and its removal are recorded in the
[`spatial-mixture result`](spatial-lobe-mixture-2026-08-24.md).
