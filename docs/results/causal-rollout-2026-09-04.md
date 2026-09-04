# Causal previous-output rollout, 2026-09-04

## Question

The deployed recurrent model starts with invalid history and feeds each
reconstructed frame into the next one. Training previously sampled one frame
at a time and reconstructed its predecessor from a spatial reset. Does training
on the actual causal state distribution fix the over-confident gather and the
remaining low-frequency temporal noise?

## Implementation

`ommatidia-train --rollout-training` selects one sequence and crop for every
batch slot, visits all four frames in order, and carries a detached teacher's
reconstructed output forward. Frame zero always has invalid history. Meganeura
already supports gradient accumulation, so the four frame losses are averaged
into one Adam update instead of requiring a larger graph, operation, or shader.
The teacher is synchronized by one staged in-memory parameter readback; the old
temporary-checkpoint/environment path is gone.

The option requires direct kernel prediction, a sequence dataset, and
`--previous-output`. Ordinary randomly sampled training is unchanged. When the
temporal loss is disabled, batch assembly also skips its unused canonical-delta
target.

The long arm is reproduced with:

```sh
cargo run --release -p ommatidia-train -- \
  --data data/dlss-native-motion-train-exact-256x4.omd \
  --eval-data data/dlss-native-motion-holdout-exact-32x4.omd \
  --steps 2000 --batch 8 --tile 64 --base-channels 16 \
  --lr 3e-4 --lr-final 1e-5 --prediction kernel \
  --reconstruction-base sample --kernel-radius 3 --demodulate \
  --history-frames 4 --temporal-features variance \
  --previous-output --guide-mix --rollout-training \
  --out runs/dlss-linear-guide-rollout-r3-b16
```

## Equal-frame-exposure control

The exact-motion procedural training set has four-frame sequences. Thus a
500-update rollout and the old 2,000-update run both see 2,000 frame batches;
a 2,000-update rollout and the selected 8,000-update run both see 8,000. Batch
size, 64-pixel crops, seed, b16/r3 model, 3e-4 to 1e-5 cosine schedule, and
validation crop order are otherwise fixed. The first table compares the early
equal-exposure checkpoints.

| 2,000 frame batches | PSNR | SSIM | LF PSNR | energy | fine temporal vs guide | LF temporal vs guide | history requested |
|---|---:|---:|---:|---:|---:|---:|---:|
| old one-pair training | **26.29 dB** | **0.7672** | **30.01 dB** | **0.943** | **-0.59 dB** | **-0.17 dB** | 6.5% |
| causal rollout | 26.18 dB | 0.7631 | 29.91 dB | 0.937 | -0.81 dB | -0.50 dB | 3.6% |

The longer comparison reaches the same conclusion on both the procedural
selector and the disjoint eight-family ABO mesh audit:

| 8,000 frame batches | PSNR | SSIM | LF PSNR | energy | fine temporal vs guide | LF temporal vs guide | history requested |
|---|---:|---:|---:|---:|---:|---:|---:|
| procedural, old | **26.32 dB** | **0.7655** | **30.03 dB** | **0.941** | **-0.23 dB** | **+0.47 dB** | **16.5%** |
| procedural, rollout | 26.20 dB | 0.7646 | 29.90 dB | 0.936 | -0.59 dB | -0.15 dB | 9.0% |
| ABO, old | **33.14 dB** | **0.9404** | **39.37 dB** | 0.994 | **-0.32 dB** | **-0.04 dB** | **17.6%** |
| ABO, rollout | 32.84 dB | 0.9386 | 38.85 dB | **0.994** | -0.70 dB | -0.72 dB | 11.8% |

Plain per-frame MSE responds to accumulated self-error by trusting history less,
not by learning better confidence: rollout roughly halves the learned history
share at equal early exposure and remains far behind after convergence. Every
reported quality/stability metric regresses, including the real-mesh audit.
The rollout checkpoint is rejected; the selected checkpoint does not change.

A directional 1,000-update probe reintroduced the smallest useful temporal
weight (`0.1`). Against the no-temporal-loss rollout at the same update and
frame count, procedural PSNR/SSIM changed from 26.22/0.7656 to 26.18/0.7636,
and its fine/broad temporal comparisons changed from -0.65/-0.27 dB to
-0.82/-0.44 dB. ABO changed from 32.79/0.9381 to 32.71/0.9359, while temporal
comparisons fell from -0.92/-1.09 dB to -1.37/-1.69 dB. Requested history
also fell from 6.6% to 5.0% procedurally and from 8.8% to 3.9% on ABO. The two
runs used different cosine horizons, so this is not the primary architecture
control above, but a result that loses every gate does not justify a longer
weight sweep. The temporal-loss checkpoint is rejected too.

## Real geometry from initialization

The previous ABO experiment was a short continuation of a procedural model.
As a separate control, the ordinary one-pair model was trained from
initialization for 8,000 steps, alternating complete batches from the exact
procedural set and 24 disjoint ABO families. This does learn a more conservative
gate, but requested history falls from about 17% to 7%. Multiplying history odds
by 2.5 restores a comparable 14–15% valid share. On uniformly sized 256x256
full frames:

| full-frame result | PSNR | SSIM | LF PSNR | energy | fine temporal vs guide | LF temporal vs guide |
|---|---:|---:|---:|---:|---:|---:|
| procedural, selected checkpoint | **26.26 dB** | 0.7616 | **29.93 dB** | **0.935** | -0.24 dB | +0.38 dB |
| procedural, mixed + 2.5x history | 26.18 dB | **0.7676** | 29.61 dB | 0.930 | **-0.04 dB** | **+0.47 dB** |
| ABO, selected checkpoint | 33.22 dB | 0.9334 | **39.88 dB** | **0.995** | -0.32 dB | **-0.01 dB** |
| ABO, mixed + 2.5x history | **33.26 dB** | **0.9357** | 39.79 dB | 0.993 | **-0.26 dB** | -0.05 dB |

This is a genuine Pareto trade rather than the failed adaptation: structure and
fine temporal error improve on both domains. It is still not a quality advance.
The procedural image gives back PSNR, low-frequency fidelity, detail, and
energy, while full-frame differences remain subtle and the visible broad
mottling remains. The 2.5x setting was also selected on already-inspected data.
The mixed checkpoint is therefore not promoted; fresh mesh families remain
necessary after a confidence objective produces a clear win.

This is still useful architecture evidence. Causal exposure is necessary for
an honest state experiment, but it is not supervision. The next bounded probe
must directly supervise reactive/confidence state that distinguishes reliable
estimators, disocclusions, and changing appearance; another scalar temporal
weight is not enough. It must beat this control on both domains before any
attention block or native-state expansion is justified.
