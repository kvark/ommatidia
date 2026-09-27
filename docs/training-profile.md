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

The probe is not a quality candidate. The separate, from-scratch 5,000-step run
(`runs/v4-phase3/train-5000-retry/`) completes in 311.448 timed training seconds:
16.054 updates/s, 7.398 M valid pixel-gradients/s, 24.738× Phase 1. It consumes
40,000 windows, 6,991 cold (17.48%). Its 640-frame development evaluation and
independent reload are byte-identical, including per-lobe diagnostics. All 84
parameter/Adam tensors and all cursor state values are finite; optimizer step,
RNG/cursors, settings and data hashes pass the recorded bundle audit.

### Quality remains an open problem

This smoke checkpoint is **not promoted**. Causal development, paired against
the frozen v3 control; percentile 95% intervals use 1,000 whole-sequence
resamples of all ten sequences. Raw reference hashes and protocol match.

| Metric | v3 control | v4, 5,000 steps | v4 − v3 [95% CI] |
|---|---:|---:|---:|
| Overall PSNR, dB ↑ | 30.271 | 24.638 | −5.632 [−6.523, −4.935] |
| Cold PSNR, dB ↑ | 26.267 | 21.160 | −5.107 [−7.723, −2.876] |
| Warm PSNR, dB ↑ | 30.498 | 24.840 | −5.658 [−6.636, −4.797] |
| LDR FLIP ↓ | 0.107752 | 0.232705 | +0.124952 [+0.106281, +0.144641] |

Overall energy ratio is 0.983824 (cold 1.430018); temporal MSE is 0.000393361.
Temporal difference is +0.000034834 [−0.000033390, +0.000082118], inconclusive.
First/last 100-update mean loss is 0.045119 → 0.007313, **but fourteen finite
spikes exceed 1**, reaching 547,544.7 at update 303. The full-run mean is 155.615,
median 0.008878; no spikes are discarded. Their cause is not established, and
lower typical training loss is not a quality pass. Phase 5's sanity/debugging
rung must address this evidence before drawing budget/capacity conclusions.

Checkpoint SHA-256: `47611ca9937b649fd456233dc8c527591c517ec09a693421821926e721d5ed46`.
Commands, failures, raw outputs, FLIP scores and full intervals are recorded in
`reload-5000/`, `flip-causal/`, `causal-comparison/` and `verify-run/` under
`runs/v4-phase3/`. Confirmation and the published gallery remain untouched.
