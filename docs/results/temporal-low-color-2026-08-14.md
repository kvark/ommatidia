# Sequence-aware low-radiance reconstruction, 2026-08-14

## Question

Can a learned temporal model improve on surface-rejected sparse-path history
without adding a new Meganeura operation or giving up the deterministic safe
fallback?

## Controlled setup

- Input: four-frame sequences, one independent three-bounce path per 128×128
  pixel and a 0.05 world-X camera translation per frame.
- Target: 256 accumulated canonical frames (1,024 paths/pixel) at 256×256,
  eight-bounce maximum depth, plus output-resolution primary surfaces.
- Training: 512 sequences generated from seed 14000; the tail 77 sequences is
  held out by whole sequence. Reset frames never enter temporal batches.
- External validation: 32 separately generated sequences from seed 12000; 372
  non-overlapping 64×64 crops cover frames 2–4 of 31 sequences.
- Reprojection: current-to-previous motion, a four-sample cap, and the selected
  encoded-depth/normal/albedo rejection thresholds (0.01, 0.9, 0.04).
- Metrics: MSE/PSNR in compressed linear-radiance space and SSIM.

## Target selection

The first temporal attempt retained the spatial checkpoint's 12-channel
sub-pixel residual. It stayed at **+0.00 dB** versus rejected history through
1,000 steps. That target is mostly unobserved high-resolution Monte Carlo
detail after the deterministic guide has already pooled compatible samples.

The selected experiment instead predicts three low-resolution RGB corrections
over the exact geometry-guided base. A zero output therefore reproduces the
safe temporal HR guide exactly. Output-resolution depth, normal, and albedo
then drive the existing 5×5 gather. Current sparse RGB, accumulated RGB,
normalized history confidence, and guided RGB reach the unchanged U-Net as
ordinary channels.

An oracle replaces the predicted low-resolution colour with a 2×2 average of
the canonical target:

| reconstruction over frames 2–4 | MSE | PSNR | SSIM |
|---|---:|---:|---:|
| single-frame temporal control | 0.000763 | 31.17 dB | 0.9119 |
| surface-rejected history + spatial guide | 0.000584 | 32.33 dB | 0.9301 |
| rejected history, direct HR gather | 0.001067 | 29.72 dB | 0.8233 |
| canonical low colour + direct HR gather | 0.000245 | **36.11 dB** | **0.9602** |

The poor direct-history row is why the learned path corrects the guided base
rather than replacing it. The canonical-low row is an information ceiling,
not a model result.

## Learned result

On the independent seed-12000 set:

| model | parameters | estimated 1080p arithmetic | MSE | PSNR | SSIM | gain over temporal HR guide |
|---|---:|---:|---:|---:|---:|---:|
| temporal HR guide, no network correction | 0 | — | 0.000595 | 32.26 dB | 0.9292 | — |
| b8 low-colour residual, 2,000 steps | 73,664 | 12.7 GFLOP | 0.000592 | 32.28 dB | 0.9292 | +0.02 dB |
| b16 low-colour residual, 1,500 steps | 289,920 | 47.2 GFLOP | 0.000589 | **32.30 dB** | **0.9294** | +0.04 dB |

The internal whole-sequence holdout agrees: b8 gains +0.03 dB and b16 gains
+0.05 dB. The external result is smaller but has the same ordering.

## Decision

Keep b8 as the temporal implementation target. B16 spends 3.7× the arithmetic
and 3.9× the parameters for another 0.02 dB externally. Most of the measured
quality improvement remains the 1.16 dB supplied by valid history, not network
capacity.

Do not publish these weights as the default yet. The current data has camera
translation but no object motion, animated specular response, exposure change,
reactive mask, or camera cut. The CPU sequence contract and checkpoint metadata
are now explicit, but the native runtime deliberately rejects temporal
checkpoints until its GPU history pack/unpack path exists. The next quality
gate is harder motion plus motion-compensated temporal stability, followed by a
gated or local-attention fusion ablation only if the b8 convolutional path
fails on that evidence.

No Meganeura source, operator, autodiff rule, compiler path, or shader group
changed in this experiment.
