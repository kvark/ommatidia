# Teacher-owned occlusion, 2026-08-19

## Question

The temporal term works, but weighting it toward moving pixels made every
column worse. The target is least trustworthy where motion is largest:
reprojection was ordinary bilinear and inherited the sample-history
validity mask. Does giving the teacher its own occlusion test — on the
converged high-resolution G-buffer, dropping bilinear taps that fail the
surface test — fix the moving-pixel column?

## What the inherited mask was doing

History accumulation rejects on the noisy low-resolution G-buffer and
writes a confidence. The teacher and the metric both treated
`confidence * frames > 1` as “this pixel may be compared.” That is a
fact about the sparse samples, not about whether the canonical frames
see the same surface.

On eight sequences of the external validation set the two masks
disagree:

| | all pixels | moving pixels |
|---|---:|---:|
| teacher keeps | 92.9% | 92.3% |
| history mask keeps | 92.7% | 85.1% |
| only the teacher | 13,468 | — |
| only the history mask | 12,097 | — |

The history mask is more conservative where things move, so those
pixels got no temporal gradient. It is also more permissive at
silhouettes: 12,097 output pixels were scored and trained against a
bilinear mix of two surfaces. Both are the same bug. The ignored test
`teacher_and_history_masks_disagree_on_real_sequences` pins that they
are not the same set.

## The change

`temporal::sample_reprojected` bilinear-samples an interleaved linear
image and keeps only the taps whose previous high-resolution surface
matches the current one, using the same `RejectionConfig` history
already uses. A pixel with no surviving tap is missing, not clamped.
`metrics::temporal_error` and `batch::temporal_target` both call it, so
the loss is still the metric rearranged — the existing numerical
agreement test now rejects on a real depth discontinuity rather than
on a hand-me-down low-resolution bit.

The sample-history path is untouched. It still rejects on the noisy
low-resolution G-buffer; that is the evidence the network is handed.
The teacher no longer consults it.

A sequence dataset without high-resolution depth, normal, and albedo
is now rejected up front. Every temporal set this project has already
carries them.

## The measurement change is the result

Same weight-1 checkpoint (`runs/tl-w1`), same external validation file,
1,524 crops. The published table used the inherited mask; this table
uses the teacher. Spatial columns move only because the crop set is
larger and harder (the HR guide is 28.10 dB here against 28.54 dB
there). The temporal columns are a different quantity.

| | PSNR | SSIM | relMSE | worst crop | detail | temporal | moving |
|---|---:|---:|---:|---:|---:|---:|---:|
| four-frame HR guide | 28.10 dB | 0.8664 | 0.203 | 8.17 | 44% | — | — |
| kernel, weight 1, inherited mask (published) | 30.04 dB | 0.8802 | 0.036 | 0.22 | 76% | −0.17 | **−2.24** |
| kernel, weight 1, teacher metric | 29.58 dB | 0.8745 | 0.075 | 13.14 | 79% | **+0.01** | **+0.11** |

Moving pixels go from 2.24 dB worse than the deterministic base to
0.11 dB better, with no weight change. The “moving pixels never beat
the base” finding was the inherited mask scoring invalid
reprojections as flicker.

## Training against the new target does not move it further

Matched 8,000-step b16 kernel, demodulated, temporal weight 1, same
data and seed, scored on the same 1,524 crops.

| | PSNR | SSIM | relMSE | worst crop | detail | temporal | moving |
|---|---:|---:|---:|---:|---:|---:|---:|
| four-frame HR guide | 28.10 dB | 0.8664 | 0.203 | 8.17 | 44% | — | — |
| kernel, weight 1, old target, new metric | 29.58 dB | 0.8745 | 0.0750 | **13.14** | **79%** | +0.01 | +0.11 |
| **kernel, weight 1, new target** | 29.56 dB | 0.8745 | **0.0745** | 14.14 | 78% | **+0.03** | **+0.14** |

Three hundredths of a decibel on the column this was built to move.
The old target, scored honestly, was already the answer. Fitting the
network to the honest target does not buy a second one.

## What this does not settle

The second thread from the previous stop is recorded in
[`unrejected-history-tap-2026-08-19.md`](unrejected-history-tap-2026-08-19.md).
Exposing the ungated reprojection as another tap does not move the
score. Leave it off.

The residual model still posts a larger moving-pixel gain under the
old metric, with relative error 73 and a worst crop of 2,801. It has
not been re-scored here. Under the new metric the kernel already
beats the base on that region without inventing radiance, so that
comparison is no longer the reason to keep the residual path.
