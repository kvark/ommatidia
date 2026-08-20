# Previous-output mix after the gather, 2026-08-20

## Question

Kernel b16 with previous-output history as four extra gather taps is
+0.53 dB temporal on the external set and 15.0 ms / 63.8 GFLOP at
1080p. Those four taps inflate the head to 116 channels. Can the
warped previous frame be mixed *after* the spatial gather — one gate
per sub-pixel — so the head stays a spatial kernel, without the b8
collapse?

## The change

`--previous-output` no longer adds gather taps. The network still
emits the 25-tap spatial kernel (100 channels at scale 2) plus four
softplus mix gates. Gather is spatial only. Then

```
h = m / (m + 1)
out = (1 - h) * gather(current) + h * warp(prev)
```

in compressed sub-pixel space, matching the training graph and
`assemble_kernel`. An untrained mix sits at the same floor as the old
history taps, so step zero still reconstructs the current frame.

Old sidecars without `previous_output` still have one accumulated-sample
gather tap and zero mix channels.

104 output channels instead of 116, 304,280 parameters, 62.0 GFLOP.

## Result

Same 8,000-step b16 kernel, demodulated, temporal weight 1, same data
and seed. External validation, 1,524 crops, 1,475 temporal pairs.
Network-only times, 960×540 → 1920×1080, RX 7900 XT, load counter
unavailable.

| | PSNR | SSIM | relMSE | worst crop | detail | temporal | moving | ms | GFLOP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| four-frame HR guide | 28.10 dB | 0.8664 | 0.203 | 8.17 | 44% | — | — | — | — |
| b16, 4 extra taps (`tl-prevout`) | **29.36 dB** | **0.8682** | 0.0694 | 11.11 | 84% | +0.53 | **+0.22** | 15.0 | 63.8 |
| **b16, mix after gather** | 29.25 dB | 0.8651 | 0.0704 | **8.47** | **88%** | **+0.54** | +0.20 | **14.66** | **62.0** |

The mix is 0.34 ms faster than the extra-tap previous-output head
(14.66 vs 15.02 measured in the same `kernel_reconstruction_frame_cost`
harness: 14.66 vs 15.0 recorded). Temporal is even. PSNR pays 0.11 dB.
Detail and worst-crop improve. It stays 1.15 dB above the HR guide
spatially and +0.54 dB temporally — not the b8 collapse.

## What this does not settle

15 ms → 2 ms is still pack/dispatch and a reconstruction that does not
re-denoise the current frame at b16 capacity. This only takes the
history parameterization off the wide head. 1-spp under previous-output
is still unmeasured.
