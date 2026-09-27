# Phase 1: v3 training profile

Measured 2026-09-27, RX 7900 XT / isolated RADV 26.2.3, Rust 1.92.
Run: `runs/v4-phase1/profile-200/` (manifest, source diff, full command and input
hashes). Original 40-scene corpus, 2,560 frames; width 16, unroll 2, seed 31,
200 updates from scratch. This is a timing probe, not a candidate checkpoint.
Development was loaded for split validation but not evaluated.

| Stage | Mean ms/update | Share |
|---|---:|---:|
| Parameter synchronization | 0.320 | 0.07% |
| Frame decode | 60.861 | 13.89% |
| Warmup advance | 69.848 | 15.94% |
| Per-slot advance | 9.156 | 2.09% |
| Readback and CPU history maps | 276.143 | 63.00% |
| Feed uploads | 5.770 | 1.32% |
| Training step + wait | 16.172 | 3.69% |
| Loss readback | 0.0015 | <0.01% |
| Other | 0.029 | 0.01% |
| **Total** | **438.300** | **100%** |

The 200 timed updates took 87.660 seconds (2.282 updates/s). Excluding the first
10 gives 430.693 ms/update (2.322 updates/s), with 61.99% in readback/history maps
and 3.76% in step/wait. Times are non-overlapping host wall times; GPU waits are
included. Corpus loading, session setup, evaluation, checkpoints and profile CSV
writes are outside this measurement. `read_prepared` includes previous-state and
prepared-input readback together with CPU history-map construction; it does not
separate those costs.

**R1 is not triggered:** step/wait is well below 50%. The measurement supports
the planned loop improvements; it does not establish their eventual speedup.
The profile records zero Vulkan validation errors. Both debug GPU test suites
(RADV and LavaPipe) pass separately.

The inference graph counts **1,044,381,696 MACs / 2.088763392 GFLOP** at LR 128²,
width 16 (188,160 parameters). This counts dense forward convolution taps,
including padding, and excludes activations, warp, preparation and backward.
The training graph additionally contains a low-frequency-loss convolution, so
its MAC count divided by unroll is not the inference budget.
