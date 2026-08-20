# Un-rejected history tap, 2026-08-19

## Question

The first history tap is the accumulated estimate after the hand-tuned
surface gate. A sample the gate throws away cannot be recovered, no
matter what weight the network would have given it. Does exposing the
ungated reprojection as a second tap let rejection be learned?

## The change

`temporal::Config::unrejected_tap` adds one gather tap. It is the
previous estimate, bilinear-sampled at the motion landing, with no
depth/normal/albedo test. Out of the frame it is this frame's colour,
so the tap is a no-op rather than a black ghost.

The rejected accumulation stays tap 25. The new one is tap 26. Both
start at the bilinear-bias floor, so an untrained network still
reconstructs the current frame. The flag defaults off and is missing
from older sidecars, so `runs/tl-w1` and `runs/tl-teacher` still load
as a single history tap.

`--unrejected-tap` turns it on. 108 output channels instead of 104,
304,860 parameters instead of 304,280, 62.6 GFLOP instead of 62.0.

## Result

Matched 8,000-step b16 kernel, demodulated, temporal weight 1, teacher
occlusion, same data and seed. Scored on the same 1,524 external
crops as the previous stop.

| | PSNR | SSIM | relMSE | worst crop | detail | temporal | moving |
|---|---:|---:|---:|---:|---:|---:|---:|
| four-frame HR guide | 28.10 dB | 0.8664 | 0.203 | 8.17 | 44% | — | — |
| kernel, one history tap (`tl-teacher`) | **29.56 dB** | **0.8745** | 0.0745 | **14.14** | 78% | **+0.03** | +0.14 |
| **kernel, un-rejected second tap** | 29.51 dB | 0.8737 | **0.0718** | 15.96 | **79%** | +0.02 | **+0.15** |

A hundredth of a decibel on moving pixels, five hundredths lost on
PSNR, a slightly better relative error and a worse worst crop. The
gate was not hiding a recoverable signal the network wanted. Leave
the flag off.

## What this does not settle

The runtime still refuses temporal kernel checkpoints: pack/unpack
have no history path. That is independent of how many history taps
training writes.

The next quality step is not another sample-history variant. It is
recorded in
[`previous-output-2026-08-19.md`](previous-output-2026-08-19.md).
