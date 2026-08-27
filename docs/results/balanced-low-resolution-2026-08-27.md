# Balanced low-resolution correction — 2026-08-27

## Question

Can a compact model remove broad split-lobe estimator error without redrawing
the output-resolution detail that the deterministic reconstruction already
gets right?

The experiment predicts one RGB correction per input pixel, bilinearly adds it
in compressed radiance to the fixed output-resolution split-lobe image, and
leaves the latter's exact albedo and geometry detail in place. It adds no
Meganeura operation or shader group.

## Protocol

- Training alternates complete optimizer batches equally between 32 procedural
  seed-62000 sequences and 24 ABO-object sequences. Record count cannot make
  either corpus dominate.
- Model selection uses eight mesh-disjoint ABO hold-out sequences: 480 crops,
  covering four 128x128-output regions at every history-bearing frame.
- The model is direct B8, three levels, one block, 78,272 parameters and an
  estimated 17.5 GFLOP per 1920x1080 network pass. It sees 16-frame
  projection-jittered phase/lobe history.
- Targets are independent 16,384-spp, eight-bounce 256x256 references. A
  separate measurement puts one-reference noise at 51.67 dB PSNR and
  low-frequency noise at 73.14 dB, far below the error here.
- After checkpoint and correction share were frozen, a new seed-90000 family
  was generated and opened once. Its score covers all 120 history-bearing
  frames as full 256x256 images.

## Training curve

The fixed split guide scores 32.91 dB / 0.9507 SSIM / 37.08 dB low-frequency
PSNR on the ABO selector set.

| step | PSNR | SSIM | LF PSNR | relMSE | worst crop |
|---:|---:|---:|---:|---:|---:|
| 1000 | 33.30 | 0.9371 | 37.61 | 0.02075 | 0.17 |
| 1500 | **33.34** | **0.9386** | **37.63** | **0.02047** | 0.18 |
| 2000 | 33.35 | 0.9356 | 37.63 | 0.02059 | 0.19 |
| 2500 | 33.37 | 0.9377 | 37.67 | 0.02294 | 0.58 |
| 3000 | 33.36 | 0.9379 | 37.66 | 0.02281 | 0.91 |

The low-resolution constraint roughly halves the old unconstrained residual's
SSIM loss, but a full correction remains overconfident and late training again
damages dark crops. Step 1500 was selected before any new audit was rendered.

Adding the existing training-only 8x8 block-average loss at weight four was
strictly worse at step 1000: 33.12 dB / 0.9363 SSIM / 37.42 dB LF versus
33.30 / 0.9371 / 37.61 for ordinary MSE. Support for applying that loss to this
head was removed again; the rejected arm adds no graph or checkpoint surface.

## Correction calibration

The network output is a normalized physical correction, so evaluation can
scale it without retraining. The share is selected on the ABO set, never on
the final audit.

| correction share | PSNR | SSIM | LF PSNR | relMSE | worst crop | detail |
|---:|---:|---:|---:|---:|---:|---:|
| 0% (fixed guide) | 32.91 | 0.9507 | 37.08 | 0.02550 | 0.25 | 79% |
| 12.5% | 33.05 | **0.9513** | 37.26 | 0.02420 | 0.24 | 79% |
| **20%** | **33.12** | **0.9513** | **37.35** | **0.02352** | **0.23** | **80%** |
| 25% | 33.16 | 0.9511 | 37.41 | 0.02311 | 0.23 | 80% |
| 50% | 33.32 | 0.9488 | 37.62 | 0.02152 | 0.21 | 82% |
| 100% | 33.34 | 0.9386 | 37.63 | 0.02047 | 0.18 | 86% |

Twenty percent dominates 12.5% except for a tied rounded SSIM and keeps more
structural margin than 25%. It also improves reprojected temporal error by
0.18 dB and 16x16-block temporal error by 0.33 dB on the selector set.

## Untouched full-frame audit

| reconstruction | PSNR | SSIM | LF PSNR | relMSE | worst frame | detail |
|---|---:|---:|---:|---:|---:|---:|
| fixed split-lobe guide | 30.85 | 0.9309 | 35.14 | 0.04420 | 0.21 | 78% |
| B8 correction, 20% | **31.16** | **0.9328** | **35.60** | **0.03905** | **0.17** | **79%** |

| temporal diagnostic | fixed split | calibrated model | gain |
|---|---:|---:|---:|
| reprojected delta MSE | 0.000337 | **0.000325** | +0.16 dB |
| 16x16-block delta MSE | 0.000104 | **0.000096** | +0.32 dB |
| moving-pixel delta MSE | 0.000384 | **0.000370** | +0.16 dB |

Every registered metric moves in the right direction on unseen procedural
scenes, so this is a valid generalization result rather than another
PSNR-for-structure trade. Full-frame evaluation also makes the comparison PNGs
an identical 256x256; the crop-size override changes no weights.

## Decision

Keep the constrained head and the 20% candidate as the next offline baseline,
but do not call the visual goal achieved or publish it as the native default.
The full-frame result remains very close to the fixed estimator and still
misses canonical glossy structure. A useful model only supplies one fifth of
the correction it learned, which says the training objective or corpus still
does not calibrate confidence.

The next quality work is conservative correction training on more disjoint
scene and lighting families, followed by joint history reconstruction that
introduces genuinely new samples. Native split-radiance unpack should wait for
a visibly stronger result. A transformer remains behind that data/history
gate: this compact convolutional model has not saturated the available signal.
