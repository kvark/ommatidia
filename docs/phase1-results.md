# Phase 1 measurements

Completed 2026-09-27. These establish the corrected **v3 control**, not a new
model or a quality improvement over the previously trained checkpoint. README
images and published weights remain unchanged. Confirmation was not evaluated.

## Training cost

The [200-update profile](training-profile.md) on the original 40 scenes gives
438.30 ms/update (2.282 updates/s): **63.00% readback/history-map preparation,
3.69% training step/wait**. R1 is not triggered. The planned loop work is justified;
its eventual speedup is not yet measured. Inference is 1,044,381,696 convolution
MACs / 2.088763392 GFLOP per LR 128² frame at width 16.

## Development control

Checkpoint `89e81df0…`, archived executable/source `cf37c0a`, RX 7900 XT with
isolated RADV 26.2.3. Both protocols cover ten sequences × 64 frames, LR 128² →
HR 256². Full-precision outputs and every PNG are retained. The final evaluations
and scoring runs have the `-complete` suffix in [the ledger](experiments.md).

| Protocol | Reset age | Frames | PSNR (dB) | LDR-FLIP | Temporal MSE | Energy ratio |
|---|---|---:|---:|---:|---:|---:|
| Causal | All | 640 | 30.270709 | 0.107752 | 0.000358527 | 1.004229 |
| Causal | Cold, 0 | 10 | 26.267388 | 0.168705 | — | 0.991669 |
| Causal | Early, 1–7 | 70 | 29.174099 | 0.126336 | 0.000577790 | 0.997986 |
| Causal | Settling, 8–15 | 80 | 30.366274 | 0.107924 | 0.000351954 | 0.999514 |
| Causal | Warm, ≥16 | 480 | 30.498107 | 0.103744 | 0.000327646 | 1.006187 |
| Reset-16 | All | 640 | 29.607780 | 0.119287 | 0.000447977 | 1.000087 |
| Reset-16 | Cold, 0 | 40 | 26.535138 | 0.164773 | — | 0.993808 |
| Reset-16 | Early, 1–7 | 280 | 29.231235 | 0.125323 | 0.000566706 | 0.998378 |
| Reset-16 | Settling, 8–15 | 320 | 30.321338 | 0.108319 | 0.000344089 | 1.002368 |
| Reset-16 | Warm, ≥16 | 0 | — | — | — | — |

Temporal pairs crossing cuts are excluded: 630 causal pairs and 600 reset-16
pairs. Dashes denote undefined metrics, not zero error. Energy's target is 1.
FLIP uses official `flip-evaluator==1.7`, 67 PPD, `x/(1+x)` then sRGB, without
quantization. Its reference-image mean passes the required 1e-4 tolerance; the
additional magma-image discrepancy is disclosed in [the protocol](evaluation.md).

Differences below are **v3 minus its zero-head guide**, not v4 minus v3 and not
relative to the old published runtime. Intervals are paired, 1,000-resample,
whole-sequence percentile 95% CIs (seed 31), retaining frame weighting.

| Metric | Causal difference [95% CI] | Reset-16 difference [95% CI] |
|---|---|---|
| PSNR | +2.1952 [+1.8758, +2.5867] dB | +3.2647 [+2.9263, +3.6505] dB |
| FLIP | −0.02385 [−0.03108, −0.01717] | −0.04454 [−0.05596, −0.03371] |
| Temporal MSE | −0.0003353 [−0.0003940, −0.0002759] | −0.0007798 [−0.0009035, −0.0006400] |
| Energy ratio | −0.00428 [−0.01217, +0.00442] | +0.00049 [−0.00640, +0.01103] |

No energy improvement is established by those intervals. Full-precision metrics
and CIs for **every reset-age stratum** are in [the verified result](phase1-results.json).

## Crops

The 24 historical selections are intact. With owner approval, reference-only
inspection added frame 3 to the sixteen unchanged rectangles, recorded before
the complete evaluations. [Definitions and reference hashes](dev-crops.json).
Coverage is five of ten sequences; this is not exhaustive spatial coverage.

MSE ratios below are v3 / zero-head guide, in compressed RGB (lower is better).

| Protocol / crop age | Smooth | Edge | Texture |
|---|---:|---:|---:|
| Both / cold, frame 0 | 0.1923 | — | — |
| Both / early, frame 3 | 0.2041 | 0.4874 | 0.0735 |
| Causal / warm, frame 31 | 0.5260 | 0.7732 | 0.1394 |
| Reset-16 / settling, frame 31 (age 15) | 0.4217 | 0.7084 | 0.1277 |

Cold edge/texture selections and reset-16 warm frames do not exist. Gradient
ratios and exact crop totals are retained in the JSON; no missing stratum is
treated as a pass. These are descriptive control measurements, not the sprint's
≤0.50 gate against the original published learned checkpoint.

## Reproduction and checks

`runs/archive/v3-runtime/` contains the executable, unchanged checkpoint/config,
source tarball, recorded clean release build and hash inventory. The final raw
outputs are `runs/v4-phase1/v3-control-{causal,reset16}-complete/outputs/`.
Corresponding `v3-flip-*`, `v3-ci-*`, and `v3-crops-*` runs retain all commands,
input hashes, source revisions and results. The tools are documented in
[evaluation.md](evaluation.md); the owner creates PRs from the feature branch.

The recorded `verify-control` audit passed: both protocols have complete frame
coverage; references match byte for byte; every raw output and PNG reproduces
the first runs exactly; the first sixteen frames match across protocols; reset
ages, nulls, aggregates, crop coverage and all 1,000-resample intervals agree.
The original audit/confirmation benchmark files and published freeze are unchanged.

92 Rust tests, workspace Clippy, formatting, Python tests, the official FLIP mean
test and the end-to-end LavaPipe evaluator smoke pass. The unchanged inference/
training kernels passed all eight debug GPU checks on each of RADV and LavaPipe,
with zero validation errors. Saved-control self-comparisons and serialized reload
are exact; mismatched controls fail. No v4 training or model-selection decision
was performed. The next implementation phase is the single v4 model.
