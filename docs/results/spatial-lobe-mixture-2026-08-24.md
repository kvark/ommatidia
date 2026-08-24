# Spatial lobe-filter mixture — 2026-08-24

## Question

Can a compact spatial model turn the measured lobe-scale oracle into a useful
estimator without adding Meganeura operations, shaders, or native runtime
surface area?

The experiment used the same 32-scene, 16-frame phase-lobe corpus as the
oracle audit, split by whole scene into 24 training and 8 untouched validation
sequences. An 81,560-parameter B8 U-Net predicted positive, normalized
mixtures over six existing à-trous scales independently for diffuse and
specular radiance at each 2x output subpixel. Exact albedo and emission then
composed those lobes into the output image. Training used 32x32 low-resolution
tiles in batches of eight and optimized the composed image directly.

This design reused existing Meganeura convolution, activation, normalization,
and loss operations. It introduced no new operation, shader group, checkpoint
contract, or native path.

## Evaluation correction

The trainer's capped evaluation previously consumed a contiguous prefix of the
validation iterator. With `--eval-crops 64`, that meant only the first held-out
scene and its earliest frames could determine a reported score. Capped,
non-recurrent evaluation now distributes samples deterministically across the
whole held-out scene/frame/crop product. Recurrent-output evaluation retains
strict sequential order. The table below uses the balanced 64-crop audit.

The experiment also exposed a generic Meganeura cooperative-convolution issue:
batched convolutions whose output channel count did not fill a complete
cooperative matrix tile stored later batches with a padded stride, while
consumers used the logical stride. A GPU regression and the narrow eligibility
guard are in [Meganeura PR #152](https://github.com/kvark/meganeura/pull/152).
All controlled results below were measured after that correction.

## Held-out result

| reconstruction | PSNR | SSIM | relMSE | detail | LF PSNR |
|---|---:|---:|---:|---:|---:|
| fixed split-lobe filter | **31.10 dB** | **0.9309** | **0.04074** | 74% | 34.57 dB |
| spatial mixture, step 1,000 | 31.01 dB | 0.9258 | 0.04227 | 76% | **34.81 dB** |
| decisions averaged over 8x8 output blocks | 31.01 dB | 0.9259 | 0.04231 | 76% | 34.80 dB |
| 500-step low-frequency-loss continuation | 30.94 dB | 0.9240 | 0.04219 | **77%** | 34.75 dB |

The plain spatial model gained only 0.24 dB of low-frequency PSNR while losing
0.09 dB overall and materially reducing SSIM. Averaging decisions over the
oracle's block size made no meaningful difference. Continuing with a 4x
low-frequency image-loss term improved neither low-frequency nor aggregate
quality. None approached the predeclared half-oracle-gap gate.

## Decision

Reject the spatial mixture and remove its implementation. Together with the
failed pooled selector, this shows that the current oracle gap is measurable
but not predictably accessible from the present feature set and corpus merely
by choosing among fixed filters. It does not justify native integration or more
shader complexity.

The next quality experiment should first broaden the independent clean corpus
and establish a data-scale curve, then predict clean diffuse and specular
radiance directly while retaining the fixed multiscale result as an input and
baseline. The scale oracle remains useful as a diagnostic ceiling, not as the
next product architecture.
