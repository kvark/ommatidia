# Previous-output history at b8, 2026-08-19

## Question

Kernel b16 with previous-output taps is +0.53 dB temporal on the
external set, and 15.0 ms / 63.8 GFLOP at 1080p. DLSS-class is 1–2 ms.
If history is now doing real work, is b8 (21 GFLOP, 9.1 ms) enough?

Same data, seed, steps, demodulation, temporal weight, and
`--previous-output` as `tl-prevout`. Only `base_channels` changes,
8 instead of 16: 81,772 parameters against 306,020.

## Result

External validation, 1,524 crops, 1,475 temporal pairs.

| | PSNR | SSIM | relMSE | worst crop | detail | temporal | moving | ms | GFLOP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| four-frame HR guide | 28.10 dB | 0.8664 | 0.203 | 8.17 | 44% | — | — | — | — |
| **b16 previous output** | **29.36 dB** | **0.8682** | **0.0694** | 11.11 | 84% | **+0.53** | **+0.22** | 15.0 | 63.8 |
| b8 previous output | 26.32 dB | 0.8307 | 0.0699 | **6.02** | 95% | −3.53 | −2.17 | **9.1** | **21.0** |

b8 is 1.65× faster and 3× less arithmetic. It is also 3.0 dB worse
spatially than b16, 1.8 dB worse than the deterministic guide, and
4.1 dB worse temporally. Detail at 95% is the same under-smoothing
the original kernel b8 showed on single frames; recurrence then feeds
that grain back as history, which is the −3.53 dB.

Width still matters under kernel prediction, and previous-output
history makes a too-small network worse rather than cheaper. Do not
drop to b8.

Network-only times, 960×540 → 1920×1080, RX 7900 XT, load counter
unavailable:

| | ms | GFLOP | output channels |
|---|---:|---:|---:|
| residual b8 | 7.25 | 12.9 | 12 |
| kernel b8 r2 | 8.58 | 19.4 | 100 |
| kernel b8 prevout | 9.08 | 21.0 | 116 |
| kernel b16 r2 | 14.59 | 60.7 | 100 |
| kernel b16 prevout | 15.02 | 63.8 | 116 |
| kernel b16 r2 head1 | 13.50 | 47.4 | 100 |

The previous-output taps add half a millisecond. A 1×1 head on b16
saves 1.1 ms and was already measured as −0.73 dB spatially when
history was unused; it is not a path around the width result.

## What this does not settle

The remaining 15 ms → 2 ms is not a model-width knob on this
architecture. It is pack/unpack, Meganeura dispatch, and whether a
different reconstruction (fewer taps, a 1×1-dominated head after a
real history blend, or a smaller gather) can stand on previous-output
history without having to re-denoise the current frame from scratch
at b16 capacity.
