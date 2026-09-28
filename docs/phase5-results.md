# Phase 5: training ladder

In progress. The thresholds in [PLAN.md §9](../PLAN.md#9-phase-5-training-ladder-and-decision-point-a-3-days)
are frozen. Every run, including failed diagnostics, is in the [ledger](experiments.md).
No confirmation evaluation or gallery promotion belongs to this phase.

## Fixed protocol

- Single-scene sanity: at most 10,000 updates from scratch, original
  `fit-train-long.omd`, tested on the independent-noise `fit-long.omd`. Pass at
  ≥31.6 dB resetting every frame and ≥32.6 dB causal. Neither stream establishes
  generalization; `fit-dev-long.omd` remains scene-disjoint.
- Each ladder rung also gets the complete ten-scene development set under causal
  and reset-every-16 protocols, sequence-bootstrap intervals against the frozen
  v3 control, fixed crop errors, FLIP, energy and temporal metrics, and convolution
  FLOPs per output pixel. Only development selects candidates.
- Alpha diagnostics use the same three development frames in every run, fixed
  before the first Phase 5 development evaluation:

| Sequence | Frame | Case | Purpose |
|---|---|---|---|
| 0 | 0 | static | Cold reset: history gates must be zero |
| 2 | 31 | camera | Moving camera after accumulation |
| 6 | 63 | lights | Final lighting response |

Request `--alpha-frame 0:0 --alpha-frame 2:31 --alpha-frame 6:63`. Both lobe
gates are saved as exact row-major f32 maps and direct linear grayscale PNGs,
with 20-bin histograms in `diagnostics.json`. They are diagnostic outputs only.
No radiance compression or sRGB transform is applied to alpha maps.
Lighting response is also reported at frame 63 of both lighting sequences (6
and 7), individually and with a pooled whole-sequence interval. Use
`bootstrap-evaluation.py --final-sequence 6 --final-sequence 7`; a good sequence
must not hide a regression in the other. Only two lighting sequences contribute
to that interval, not 128 independent frames.

The ladder is sanity → 40-scene/60,000-update budget curve → four configurations
on 200 scenes ({16,32} channels × {3,4} levels), then second seeds for the best
two → decision A. A failed decision permits one prescribed replacement fallback,
not an additional architecture option. Stop and report at decision A.

## Sanity checkpoint (2026-09-28)

The initial seed-1 fit completed 10,000 updates in 621.90 timed training seconds
(16.08 updates/s, 17.72% cold windows); parameters reload bit-exactly and remain
finite. On independent input noise it scores **31.3621 dB reset-every-frame**
and **32.4875 dB causal**, missing the fixed gates by 0.2379 and 0.1125 dB.
These are single-scene diagnostics, not generalization estimates or independent
frame confidence intervals. The 60,000-step ladder is held for model/loop
diagnosis; no threshold/data changes or warm starts. Cost remains 24,864
convolution FLOPs/output pixel. Full development results are reported below.

Localization: the outer four-pixel border contributes 28.15% of compressed-RGB
error (Phase 2 fit: 6.85%). The initial crop loss masked every crop border,
including real image edges, so those pixels receive no direct supervision.
Trimming four pixels raises diagnostic PSNR to 33.65 dB, but **does not change
the full-frame gate**. Correct the training masks to retain supervision at real
image boundaries while excluding artificial crop boundaries.

The seed-7 40-scene prefix replay reproduces spikes at updates 303/396/397.
At 303, one cold cursor has loss 4,386,234; its final-frame predicted lobes peak
at 30,571 versus targets peaking at 10.18 across the window (inputs: 104.43).
This establishes a prediction explosion, not extreme target radiance; it does
not yet establish the optimizer/root cause. Data, losses and thresholds remain
unchanged for this read-only replay.

The two spike windows also contain unusually large *observed first-frame motion*
(135 and 1,305 HR pixels; following frames below 1 pixel). Motion enters the
network unscaled, including on cold resets when no prior image exists. This is
a candidate source of the cold prediction explosions, not yet an ablation result.
The border-only repeat keeps these inputs unchanged to isolate the loss fix.

The initial sanity fit's full development report is deliberately retained:

| Protocol | PSNR | Delta vs v3 (95% CI) | Cold smooth MSE ratio (95% CI) | FLIP |
|---|---:|---|---|---:|
| Causal | 19.8347 | −10.436 [−12.251, −9.030] dB | 3.080 [1.574, 4.245] | 0.33823 |
| Reset 16 | 19.8179 | −9.790 [−11.453, −8.434] dB | 3.080 [1.574, 4.245] | 0.33803 |

All reset-age/frame/crop intervals, energy and temporal metrics are in
`runs/v4-phase5/sanity-score-{causal,reset16}/`; raw outputs and the six alpha
maps per protocol are in `sanity-dev-{causal,reset16}/outputs/`. Cold alpha is
exactly zero; camera-frame means are diffuse/specular 0.797/0.668, and final
lighting-frame means 0.784/0.618 (causal). Cost is 24,864 FLOPs/output pixel.
This one-scene fit does not generalize and is not a candidate promotion.

The image-edge correction passes 98 regular tests, all-target Clippy, and all
twelve GPU checks on each of debug RADV/LavaPipe, with zero validation errors.
Checks include every-parameter f64 gradients with asymmetric loss masks, all
16 masks' spatial/coarse normalization, actual GPU mask copies and schema-2
resume. Old training bundles are rejected for resume; this correction leaves
inference unchanged.

Cold-motion counterfactuals subsequently fail on both CPU packing and native
GPU output: changing only motion to the observed 1,305/−306-pixel magnitude
changes cold output despite absent reconstruction history. The fix zeros only
these reset-frame motion features; warm vectors and geometric warp math remain
unchanged. Checkpoint schema 3 rejects resume across this input-contract change.
The controlled 500-update training pair keeps data, settings and all 4,000
sampled windows identical, changing only the reset-motion feature contract.
With corrected borders but old motion features, update 303 still spikes to
557,953.06 loss. With reset-motion fixed it is 0.02509; maximum loss is 0.09408
and there are no spikes above 1. This establishes the effect on the reproduced
failure, not a stability guarantee for the full training budget. Both complete
with finite weights and exact reload. All 13 GPU gates now pass on each of
RADV/LavaPipe; the full CLI evaluation/resume smoke also passes.

The border-only seed-1 repeat clears the sanity gates on the unchanged independent
noise stream: **31.7441 dB reset-every-frame** and **33.1141 dB causal** (+0.3820
and +0.6266 dB over the initial fit). It uses the archived matching runtime,
unchanged seed/data/10,000-step schedule and full-frame scoring, with no warm
start or inference change. It is a controlled diagnostic, not a promotion;
the final corrected input contract still needs its own from-scratch sanity fit.

The corrected seed-7 replay now completes all 5,000 updates with identical
sampling and schedule to the original Phase 3 run. There are **zero loss spikes
above 1** (original: 14); maximum loss is 0.43470. All 174,576 parameters and
349,152 Adam moment values are finite, with bit-exact parameter reload. Full
causal development is **24.8592 dB** versus 24.6385 originally; cold PSNR improves
from 21.1600 to 23.9078 dB. This diagnostic remains far below v3 and does not
replace the from-scratch seed-1 budget curve.

The final corrected seed-1 sanity fit passes both fixed independent-noise gates:
**31.8499 dB reset-every-frame** and **33.0874 dB causal**. It completes 10,000
updates from scratch, with no loss spikes above 1, finite weights and bit-exact
reload; 17.71625% of sampled windows are cold. Background LavaPipe correctness
checks mean its timing is not a speed benchmark. Its full development report is:

| Protocol | PSNR | Delta vs v3 (95% CI) | Cold smooth MSE ratio (95% CI) | FLIP |
|---|---:|---|---|---:|
| Causal | 20.2038 | −10.067 [−12.060, −8.491] dB | 3.866 [1.958, 6.016] | 0.32090 |
| Reset 16 | 20.0869 | −9.521 [−11.440, −7.958] dB | 3.866 [1.958, 6.016] | 0.32328 |

Both protocols cover all 640 frames; the complete bucket/crop, energy, temporal
and lighting-final comparisons are in `corrected-score-{causal,reset16}/` under
`runs/v4-phase5/`. Fixed alpha maps and histograms are in the corresponding
`corrected-dev-*/outputs/` directories; cold gates are exactly zero. Cost is
unchanged at 24,864 convolution FLOPs/output pixel. The single-scene fit remains
far below v3 on development and is not a candidate promotion.

## Budget and capacity ladder

The from-scratch **40-scene, width-16/three-level, seed-1, 60,000-update** budget
curve is running at `runs/v4-phase5/budget-40-w16-l3-seed1-60000/`. Causal metrics
are recorded every 10,000 updates. It overlaps report generation and correctness
checks, so its timing is not an isolated throughput benchmark.

Before capacity training, all three additional planned sizes (16×4, 32×3, 32×4)
pass recurrent HDR/reset parity, every-parameter f64 gradients (fused/unfused),
and production-extent gradient-direction checks on debug RADV and LavaPipe,
with zero validation errors. These are correctness checks, not capacity results.
The 200-scene four-configuration sweep and second seeds remain to be trained.
