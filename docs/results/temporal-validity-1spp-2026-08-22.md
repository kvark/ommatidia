# Valid temporal history at 1 spp — 2026-08-22

## Question

The curated spatial model preserves edges, but its uncertainty appears as broad
illumination mottling. Can recurrent history remove that low-frequency noise
without blurring the frame, and does the result survive the actual 1-spp input
distribution?

## A correctness bug before an architecture experiment

Previous-output reprojection stored a rejected or out-of-frame pixel as zero.
The learned post-gather mix could still select that value because the graph and
CPU reference assembler did not receive per-sub-pixel validity. Black was
therefore indistinguishable from valid black radiance. It created darkening at
disocclusions and made the network learn a rejection boundary from indirect
features it could never reproduce exactly.

`warp_previous_output` now returns compressed color and explicit validity.
Validity hard-closes the existing history gate in both the training graph and
CPU assembler. Reset frames supply zero validity, so the detached teacher is
exactly the spatial reconstruction until real history exists. This adds one
multiply to the existing experimental graph and no Meganeura operation, shader,
or shader group. A regression test pins the invariant that rejected history
cannot change current radiance.

The fix alone raises the old checkpoint from 29.29 to 29.83 dB on the 384-crop
audit subset and turns moving-pixel stability from -0.32 to +0.32 dB versus the
accumulated guide. Retraining is still necessary: the old network learned
around the invalid contract.

## Measuring the visible failure

Ordinary temporal delta MSE mixes broad brightness breathing with zero-mean
grain. The evaluator now also averages the motion-compensated delta residual in
16x16 output blocks before squaring it. This low-frequency temporal metric
largely cancels grain and edge placement while retaining coherent changes on
smooth surfaces. Per-frame low-frequency PSNR uses the same block size, and a
sequence-age breakdown checks whether recurrence improves or drifts.

These are diagnostics alongside PSNR, SSIM, relative MSE, worst-crop relative
error, and detail retention; no single scalar is the acceptance test.

## Matched 1-spp data

Applying a 4-spp-trained checkpoint to the old 1-spp set cost 2.98 dB versus
the accumulated guide. That was distribution shift, not a temporal result. A
new set uses one independent sparse path per frame, four-frame sequences,
curved camera motion, independent object motion, shadows, textures, gloss, and
an eight-bounce 1,024-spp reference (256 accumulated canonical frames at 4 spp
each).

```sh
cargo run --release -p ommatidia-data -- --device-id 0x744c \
  --out data/rich-temporal-1spp-512.omd --samples 512 --lr 128x128 \
  --scale 2 --input-frames 1 --canonical-frames 256 \
  --canonical-bounces 8 --sequence-frames 4 \
  --random-camera-motion 0.05 --object-motion 0.10 \
  --hr-gbuffer --canopy --textures --gloss --seed 40000

cargo run --release -p ommatidia-data -- --device-id 0x744c \
  --out data/rich-temporal-1spp-validation-64.omd --samples 64 --lr 128x128 \
  --scale 2 --input-frames 1 --canonical-frames 256 \
  --canonical-bounces 8 --sequence-frames 4 \
  --random-camera-motion 0.05 --object-motion 0.10 \
  --hr-gbuffer --canopy --textures --gloss --seed 41000
```

The validation score holds out 63 complete unseen sequences: four
non-overlapping crops from each of frames 2-4, or 756 crops and 746 temporal
crop pairs with at least one valid reprojection.

## Radius gate

Two 2,000-step b16 probes use the same data, seed, temporal weight, and
previous-output mix. Radius three is modestly better at the expected cost.

| kernel | PSNR | SSIM | relMSE | detail | low-frequency PSNR | temporal gain | low-frequency temporal gain | arithmetic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r2, 25 taps | 25.41 dB | 0.7588 | **0.0862** | **65%** | 29.09 dB | +3.30 dB | **+7.18 dB** | 62.0 GFLOP |
| **r3, 49 taps** | **25.67 dB** | **0.7651** | 0.0888 | 63% | **29.34 dB** | **+3.38 dB** | +7.02 dB | 76.4 GFLOP |

The final quality checkpoint therefore uses r3. It has 318,200 parameters and
200 output channels. The model was trained for 8,000 steps:

```sh
cargo run --release -p ommatidia-train -- --device-id 0x744c \
  --data data/rich-temporal-1spp-512.omd --out runs/lf-1spp-r3-b16 \
  --steps 8000 --batch 8 --tile 64 --lr 3e-4 --lr-final 1e-5 \
  --base-channels 16 --prediction kernel --kernel-radius 3 \
  --reconstruction-base sample --demodulate --history-frames 4 \
  --temporal-weight 1 --temporal-features variance --previous-output \
  --seed 0 --log-every 250 --eval-every 2000 --checkpoint-every 2000 \
  --eval-crops 128 --eval-out runs/eval-lf-1spp-r3-b16

cargo run --release -p ommatidia-train -- --device-id 0x744c \
  --data data/rich-temporal-1spp-validation-64.omd \
  --out runs/lf-1spp-r3-b16 --eval-only --val-fraction 0.999 \
  --eval-crops 10000
```

## Independent result

All figures below are compressed-linear metrics over 756 crops. Temporal gain
is against the deterministic high-resolution surface-guided accumulation.

| reconstruction | PSNR | SSIM | relMSE | worst crop | detail | low-frequency PSNR |
|---|---:|---:|---:|---:|---:|---:|
| nearest 2x | 19.20 dB | 0.5425 | 4.4936 | 108.67 | 278% | 26.39 dB |
| bilinear 2x | 20.42 dB | 0.5813 | 2.7667 | 73.78 | 177% | 28.72 dB |
| low-resolution guide | 25.62 dB | 0.7760 | 1.1253 | 35.80 | 34% | 30.02 dB |
| accumulated HR guide | 26.04 dB | 0.7839 | 0.4330 | 29.95 | 34% | 30.40 dB |
| **recurrent Ommatidium** | **27.62 dB** | **0.8106** | **0.0707** | **2.16** | **52%** | **31.89 dB** |

The model is +1.59 dB per frame over the accumulated guide while retaining
substantially more reference detail. More importantly for the original visual
failure:

| motion-compensated delta | accumulated HR guide | recurrent Ommatidium | gain |
|---|---:|---:|---:|
| all valid pixels | 0.001641 | **0.000633** | **+4.14 dB** |
| 16x16 block average | 0.000578 | **0.000064** | **+9.54 dB** |
| nonzero-motion pixels (85.4% of valid) | 0.001786 | **0.000684** | **+4.16 dB** |

The coherent component is about nine times smaller. Recurrent age does not
show drift: PSNR rises from 27.19 dB on frame 2 to 27.69 on frame 3 and 28.02
on frame 4; low-frequency PSNR rises from 31.20 to 32.03 and 32.55 dB.

## What the screenshot says

The frame below is the first non-reset crop from the independent set. The
accumulated guide has smooth horizontal illumination bands and loses the small
foreground edge. The recurrent output suppresses the bands and restores more
structure. It is still softer than the 1,024-spp reference around silhouettes,
which is the next quality problem rather than hidden temporal instability.

| accumulated HR guide | recurrent Ommatidium | 1,024-spp reference |
|---|---|---|
| ![Accumulated high-resolution guide with broad low-frequency bands](../temporal-low-frequency/hr-guided.png) | ![Recurrent Ommatidium prediction with the broad bands suppressed](../temporal-low-frequency/predicted.png) | ![Independent high-sample reference](../temporal-low-frequency/reference.png) |

## Four-spp control and rejected experiment

Retraining the same r2/b16 shape on the existing 4-spp sequences leaves
per-frame PSNR essentially tied with the corrected old checkpoint (29.72 versus
29.75 dB), but raises low-frequency temporal gain from +2.56 to +6.22 dB and
moving-pixel gain from +0.94 to +2.76 dB. The validity contract is therefore
not specific to a noisier input.

Initializing the history gate at a 50/50 mix instead of its near-spatial floor
was decisively worse: 27.75 dB, -1.83 dB temporal, and -2.55 dB on moving
pixels after 2,000 steps. A recurrent model must first reconstruct the current
frame and earn its use of history; forcing early feedback only recirculates
noise.

## Deployment boundary

This result is implemented and exercised in the Rust training/evaluation path.
The native inference API still rejects temporal checkpoints until it owns the
two output-history textures and implements motion reprojection, surface
validation, reset, and validity upload on GPU. The checkpoint is therefore an
experimental quality result, not the published runtime default. That boundary
is intentional: silently treating a temporal checkpoint as spatial would make
the measured contract false.
