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
convolution FLOPs/output pixel. Full development protocols follow.

Localization: the outer four-pixel border contributes 28.15% of compressed-RGB
error (Phase 2 fit: 6.85%). The crop loss currently masks every crop border,
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
