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

## Phase 3: cursor loop

`runs/v4-phase3/profile-200/`, same RX 7900 XT / isolated RADV, 40 scenes,
200 updates from scratch, default v4 width 16. Eight persistent 64² LR crops,
four-frame BPTT, one mean-gradient Adam update, four-pixel HR loss margin.

| Measurement | Phase 1 | Phase 3 |
|---|---:|---:|
| Updates/s | 2.282 | 16.074 |
| Nominal pixel-gradients/update | 131,072 | 524,288 |
| Valid interior pixel-gradients/update | 131,072 | 460,800 |
| Valid pixel-gradients/s | 299,046 | 7,406,965 |
| Relative valid throughput | 1× | **24.769×** |

Warm-up/inference pixels never count as gradients. The stricter interior-only
numerator passes both the ≥10× gate and the ideal 20× target. All 200 updates
take 12.442 s (62.212 ms/update); after the first ten, valid throughput is
7.444 M/s. No true-batch fallback is needed to meet this gate.

Mean host times: input upload/zeroing 3.870 ms; GPU preparation submission
0.230 ms; backward/accumulation/Adam submission and waits 57.725 ms; detached
state carry/final fence 0.124 ms; scalar loss read 0.013 ms; crop-prefetch wait
0.231 ms. Worker crop decode averages 36.538 ms **overlapped**, not additive.
Startup mapping/hashing/validation, graph setup, checkpoints and evaluation
are excluded, as in Phase 1. Update timing includes crop waits and loss CSV
writes. The full stage values and per-update rows are retained with the run.

Actual cold fraction is 16.31%. Captures are read-only memory maps, and the
worker expands crop rows only; I/O wait is 0.37% of update time, so no tiling
preprocessing is justified. Only B scalar losses return to the CPU per update;
recurrent state and prepared features stay on GPU. Evaluation shares parameters.
Checkpoint-only state/Adam readback is outside ordinary updates.

This is a throughput probe, not quality evidence. The 5,000-step full-corpus
run and development evaluation remain the outstanding Phase 3 gate.
