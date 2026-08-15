# Object-motion temporal gate, 2026-08-15

## Scope

Camera-only sequences cannot test independently moving silhouettes, object
disocclusion, or changing specular response. The data generator now accepts
`--object-motion F`. For each scene it splits one randomized sphere and one
randomized box into separate Blade objects, gives each a deterministic curved
XZ trajectory, and supplies both `transform` and `prev_transform`. Blade's
existing G-buffer therefore writes the correct current-to-previous vectors;
this adds no Blade or Meganeura code.

Static generation still bakes one model. The extra object models exist only
when object motion is requested. Object and curved camera motion can be used
together, and frame zero is unchanged for an unambiguous history reset.

The object-only validation set is 32 unseen four-frame sequences from seed
26000, with 0.10 world-unit nominal object translation, one sparse input path
per low-resolution pixel, and 256 canonical frames per target. Nonzero motion
covers 4.8% of valid temporal pixels, so the object result is also reported on
that region rather than being hidden by the static background.

## Existing checkpoint generalization

The retained camera-trained variance b8 checkpoint already generalizes:

| reconstruction | MSE | PSNR | SSIM | delta MSE |
|---|---:|---:|---:|---:|
| four-frame HR guide | 0.000445 | 33.52 dB | 0.9334 | 0.000146 |
| camera-trained variance b8 | 0.000358 | 34.46 dB | 0.9413 | **0.000127** |

On moving pixels alone, delta MSE falls from 0.000451 to 0.000391, a 0.63 dB
gain. Surface rejection accepts 99.2% of all pixels, and the deterministic
history oracle reaches 33.57 dB versus 31.66 dB for one frame. Object motion
therefore preserves useful history without hiding ghosting behind a global
score.

## Mixed-motion training

A controlled training set contains 256 four-frame sequences from seed 27000.
Every sequence combines 0.05 curved camera motion and 0.10 independent object
motion. The architecture remains the selected variance b8 model: 73,736
parameters, 12.8 GFLOP at 1080p, and 209 training dispatches.

Two proposed changes were rejected:

- A 2,000-step mixed-motion retrain traded roughly 0.03 dB between the camera
  and object gates rather than clearly improving both.
- Supplying bounded XY velocity as two more ordinary channels reduced the
  internal mixed holdout from 34.73 to 34.67 dB, SSIM from 0.9470 to 0.9461,
  and moving-pixel temporal gain from 0.72 to 0.62 dB. The feature and packing
  code were removed.

The 4,000-step variance-only run with Adam cosine decay is the useful result:

| held-out distribution | HR-guide PSNR / SSIM | old b8 PSNR / SSIM | mixed 4k b8 PSNR / SSIM | old / new delta MSE |
|---|---:|---:|---:|---:|
| object only, seed 26000 | 33.52 / 0.9334 | 34.46 / 0.9413 | **34.52 / 0.9418** | 0.000127 / **0.000127** |
| curved camera, seed 23000 | 32.92 / 0.9366 | 34.19 / 0.9447 | **34.21 / 0.9453** | **0.000232** / 0.000233 |
| internal mixed holdout | 33.67 / 0.9394 | — | **34.76 / 0.9472** | — / **0.000209** |

On the object-only moving region, the new model retains the old 0.000391 delta
MSE and +0.63 dB gain. On the internal mixed moving region it reaches 0.000230
and +0.74 dB. The longer model is a small but consistent spatial improvement
with effectively unchanged external temporal stability, at identical runtime
cost.

## Optimizer correction

This run exposed a trainer bug in `--lr-final`. Training initially configured
Adam, but each scheduled update called Meganeura's `set_learning_rate`, whose
documented meaning is to select SGD. A preliminary run stayed near 1.48 loss
instead of following the Adam control and was stopped. The trainer now updates
Adam with `set_adam` at the scheduled rate, preserving its moment buffers; the
cosine endpoints have a unit test. The corrected 4,000-step run fell from
1.440 to 1.107.

## Decision

Keep variance-only conditioning and select the mixed-motion 4,000-step b8
checkpoint for the next native temporal runtime gate. Do not retain a velocity
feature, widen the network, add a transformer, or add a Meganeura primitive.
The next architecture experiment remains paired sequence training with a
differentiable temporal loss, now backed by data that exercises both camera
and object motion.
