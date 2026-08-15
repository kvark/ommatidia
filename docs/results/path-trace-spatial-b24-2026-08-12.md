# Sparse-path spatial baseline — 2026-08-12

This is a controlled negative result. It establishes what the current
single-frame architecture can recover from genuinely sparse path tracing and
why temporal history is now the next quality milestone.

> Historical optimizer note (2026-08-15): the reported images and metrics are
> valid outputs of the saved checkpoint, but `--lr-final` unintentionally
> switched the trainer from Adam to SGD on its first scheduled update. This
> run therefore does not establish an Adam convergence plateau. The later
> guided and temporal controls supersede that interpretation.

## Setup

- 2,400 procedural scenes at 128×128 input and 256×256 reference; 2,040 train
  and 360 held out.
- Input: one or four independent three-bounce paths per low-resolution pixel.
- Target: 4,096 spp, eight-bounce canonical path trace with Russian roulette
  after bounce four.
- Model: direct objective, base 24, three levels, one block, 649,200
  parameters, 104.1 convolution GFLOP per 1080p frame.
- Training: 20,000 steps, batch 8, 64×64 crops, cosine learning rate
  `3e-4` to `1e-5`.
- Final scores below use a separate 128-scene, seed-10000 dataset. Its last 19
  scenes supply 76 non-overlapping crops.

| input | nearest MSE | nearest PSNR | nearest SSIM | network MSE | network PSNR | network SSIM |
|---|---:|---:|---:|---:|---:|---:|
| 1 spp | 0.011973 | 19.22 dB | 0.3365 | 0.011389 | 19.44 dB | 0.3363 |
| 4 spp | 0.004385 | 23.58 dB | 0.4593 | 0.004303 | 23.66 dB | 0.4595 |

The 1-spp model gains only **0.22 dB** and slightly lowers SSIM. The favorable
static 4-spp proxy gains only **0.08 dB**. Training loss plateaus near the
identity solution in both runs (`1.138480 → 1.078881` and
`1.123818 → 1.095525` respectively), so extra steps on the same formulation
are not the missing ingredient.

## Sample-count curve

The exact same reference records were reused while only the independent path
count changed. G-buffer equality is checked record by record before a reused
reference is accepted.

| sparse paths/pixel | nearest MSE | PSNR | SSIM |
|---:|---:|---:|---:|
| 1 | 0.011973 | 19.22 dB | 0.3365 |
| 2 | 0.006977 | 21.56 dB | 0.3913 |
| 4 | 0.004385 | 23.58 dB | 0.4593 |
| 8 | 0.002791 | 25.54 dB | 0.5303 |
| 16 | 0.001693 | 27.71 dB | 0.6099 |

Extending the one-sample input from three to eight path bounces changes PSNR
from 19.22 to 19.30 dB. Spending the ray budget on more independent evidence
is far more useful than extending these small scenes' rare long paths.

## ReSTIR+SVGF comparison

Blade's ReSTIR+SVGF capture against the byte-identical seed-10000 references
scores MSE 0.004351, PSNR 23.61 dB, and SSIM **0.8776**. It is almost tied with
the 4-spp raw path input by PSNR, but is dramatically ahead by SSIM. PSNR sees
similar total radiometric error; SSIM sees that one error field is structured
and the other is high-frequency Monte Carlo noise. This is direct evidence
that PSNR alone is not an adequate release gate.

The comparison is diagnostic, not a cost-equivalence claim: ReSTIR settles its
reservoirs and SVGF runs three passes, while the path arm traces four static
samples. The product contract remains sparse path tracing followed by
Ommatidium, with no ReSTIR dependency.

## Decision

Do not publish either spatial path checkpoint as the next default model. The
current network is useful after a temporally/spatially structured estimator,
but a single noisy path realization does not contain enough locally predictable
signal for this small spatial regressor. Implement the sequence contract in
[`../temporal.md`](../temporal.md), train on motion/disocclusion sequences, and
repeat this exact held-out comparison. Static 4-spp accumulation is only a
favorable evidence proxy: real reprojection must also reject invalid history.
