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
curve is complete at `runs/v4-phase5/budget-40-w16-l3-seed1-60000/`. Causal metrics
were recorded every 10,000 updates. It overlapped report generation and correctness
checks, so its timing is not an isolated throughput benchmark.

Scheduled causal development checkpoints (all 640 frames):

| Updates | Last-1,000 mean training loss | PSNR | Change from previous (95% CI) |
|---|---:|---:|---|
| 10,000 | 0.0059652 | 25.4850 dB | — |
| 20,000 | 0.0046064 | 26.0469 dB | +0.562 [−0.016, +1.201] dB |
| 30,000 | 0.0045400 | 26.3870 dB | +0.340 [−0.160, +0.888] dB |
| 40,000 | 0.0035796 | 26.3681 dB | −0.019 [−0.414, +0.356] dB |
| 50,000 | 0.0033819 | 26.3941 dB | +0.026 [−0.335, +0.277] dB |
| 60,000 | 0.0030917 | 26.0374 dB | −0.357 [−0.586, −0.136] dB |

At 60,000, the gap against v3 remains −4.233 [−5.874, −3.174] dB; warm PSNR
is −4.425 [−6.178, −3.186] dB behind. Temporal MSE ratio is 0.82413
[0.72857, 0.89348], and energy is 1.01270 [0.96802, 1.08362]. Energy's point
estimate is in range, but its interval is not contained within the target.
Development plateaus after 30,000 and regresses at the final checkpoint while
mean training loss falls about 32% from 30,000. This meets the plan's overfitting
diagnosis: proceed with the approved 200-scene corpus, not more budget on the
same 40 scenes. No checkpoint is promoted or substituted for the scheduled final
checkpoint. Reports at
`runs/v4-phase5/budget-curve-{10000,20000,30000,40000,50000,60000}/` use
whole-sequence intervals and freeze each observed loss prefix. Reference hashes
are transferred from a completed full evaluation only after verifying identical
recorded executable and ordered development capture/provenance hashes. These
metrics-only checkpoints do not claim FLIP/crop results. The final checkpoint's
full causal/reset-16 reports with those metrics and fixed alpha frames are now
complete:

| Protocol | PSNR | Delta vs v3 (95% CI) | Cold smooth MSE ratio (95% CI) | FLIP (v3) |
|---|---:|---|---|---|
| Causal | 26.0374 | −4.233 [−5.874, −3.174] dB | 0.940 [0.827, 1.307] | 0.19921 (0.10775) |
| Reset 16 | 25.9030 | −3.705 [−5.097, −2.776] dB | 0.940 [0.827, 1.307] | 0.20092 (0.11929) |

Warm smooth-crop MSE is 4.202 [2.523, 6.029] times v3; early smooth error is
2.458 [1.407, 3.650] times v3 under either protocol. Reset-16 has no warm
coverage, left null. Its temporal MSE ratio is 0.67009 [0.60322, 0.73564] and
energy 1.01289 [0.96939, 1.08186]. Temporal gains do not offset spatial misses.
The fixed static/camera frame-31 images show softened structure and appearance
errors, consistent with the quantitative results; this is not a promotion.

Lighting-final results are not hidden by pooling: causal sequence 6 loses
1.088 dB and increases FLIP by 0.06476, while sequence 7 gains 1.592 dB but has
1.078× linear MSE and 1.186× temporal MSE. Reset-16 final PSNR loses 1.504/1.956
dB on sequences 6/7, with higher FLIP on both. The no-worse lighting condition
is not met by this rung.

Every protocol retains all 640 frames, six fixed alpha maps and histograms, and
640 byte-verified shared reference files. Cold alpha is zero; causal camera-frame
diffuse/specular means are 0.814/0.769 and lighting-final means 0.815/0.744.
Full reports are `budget-score-causal-retry/` and `budget-score-reset16/` under
`runs/v4-phase5/`; the interrupted first causal scoring attempt remains intact.

The final audit verifies all 60,000 updates, 480,000 sampled windows (17.4983%
cold), all 174,576 parameters and both equally sized Adam moments finite,
2,228,224 finite carried-state values, matching checkpoint/state hashes and
bit-exact parameter reload. Final checkpoint SHA-256 is
`9c5750da8e90a6916dcb6cd56e9e0ea6efe6f81c65baa1ef8f5db368dad70a7d`.
Cost remains 24,864 convolution FLOPs/output pixel.

At update 15,991, the first long-budget loss above 1 is recorded (batch 1.0605;
one cold HDR cursor 8.4269). Its predicted peak is 1,225.6 versus a target peak
of 120.7, with finite state; observed motion stays below 0.521 HR pixels, unlike
the earlier absent-frame motion spikes. Subsequent batch losses fall below
0.06. This is retained as a finite prediction overshoot, not treated as proof of
its root cause or a reason to alter the run. The read-only observation report is
`runs/v4-phase5/budget-outlier-15991/`.

Before capacity training, all three additional planned sizes (16×4, 32×3, 32×4)
pass recurrent HDR/reset parity, every-parameter f64 gradients (fused/unfused),
and production-extent gradient-direction checks on debug RADV and LavaPipe,
with zero validation errors. These are correctness checks, not capacity results.
The 200-scene first-seed 16×3 run and both full reports are complete; 16×4 is
now training, with 32×3 and 32×4 queued. Each trains from scratch for 60,000
updates and is followed by both complete development protocols. Every child run is recorded independently;
any failed command stops the queue. The best two configurations' second seeds
and Decision A remain pending.

Pairwise capacity reports will compare raw per-crop error sums on the same
sequences, not divide independently reported ratios or confidence intervals.
The report adapter passes exact comparison against direct pixel scoring,
self-pair and mismatch checks, and the fixed ≥10% gain plus paired-interval
condition. It does not rank or promote checkpoints automatically.

The first 200-scene run's recorded updates 11–2,500 take 186.01 ms/update versus
70.62 ms on the same 40-scene prefix. GPU step/wait remains about 64 ms in both;
input-batch wait rises from 0.13 to 116.48 ms (62.6% of current wall time).
The frozen-prefix report is `runs/v4-phase5/capacity-prefix-timing/`. This
identifies input delivery as the current measured bottleneck, not its exact
underlying cause or an isolated speed comparison. The active schedule, model,
loss and data are unchanged.

Scheduled **200-scene 16×3, seed-1** checkpoints, all 640 causal development
frames; paired intervals resample whole sequences:

| Updates | Last-1,000 mean loss | PSNR | Delta vs previous (95% CI), dB | Delta vs v3 (95% CI), dB |
|---|---:|---:|---|---|
| 10,000 | 0.0079259 | 25.5216 | — | −4.749 [−5.397, −4.108] |
| 20,000 | 0.0082268 | 26.8564 | +1.335 [0.596, 1.924] | −3.414 [−4.628, −2.487] |
| 30,000 | 0.0046552 | 27.0808 | +0.224 [−0.173, 0.644] | −3.190 [−4.080, −2.404] |
| 40,000 | 0.0043241 | 27.7273 | +0.647 [0.414, 0.875] | −2.543 [−3.317, −1.821] |
| 50,000 | 0.0040542 | 27.7346 | +0.007 [−0.174, 0.195] | −2.536 [−3.357, −1.865] |
| 60,000 | 0.0039458 | 27.7914 | +0.057 [−0.141, 0.252] | −2.479 [−3.209, −1.856] |

At 60,000 updates, cold/warm PSNR are 25.6818/27.9190 dB; the warm difference
against v3 is −2.579 [−3.371, −1.822] dB. Energy is 1.00011
[0.97649, 1.03602], and temporal MSE ratio is 0.85410 [0.75496, 0.92508].
The positive 30k→40k PSNR gain is followed by nearly flat 40k→50k→60k results;
warm quality remains well below v3. The energy point estimate is in range,
not its entire interval. The temporal interval is below parity, without
establishing the separate lighting-final condition.
The completed run contains only one finite loss above 1
(1.474924 at update 19,829). Reports are `capacity-w16-l3-seed1-curve-{step}/`
under `runs/v4-phase5/`. These metrics-only reports do not establish a capacity
ranking or claim FLIP/crop results; the complete two-protocol report follows.

The completed-training audit verifies 480,000 sampled windows (17.4983% cold),
174,576 finite parameters and each Adam moment tensor set, 2,228,224 finite
carried-state values, checkpoint/state hashes and bit-exact parameter reload.
Final checkpoint SHA-256:
`3ca52b124d8efd8fb5a5d89283ce10a583be4ccd62338fcc71188c8cec4ff1d4`.
Cost remains 24,864 convolution FLOPs/output pixel.

## First capacity result: 200 scenes, 16×3, seed 1

Both complete development protocols use all 640 frames, byte-verified shared
references and the final 60,000-update checkpoint. Ratios compare against the
learned v3 control; brackets are whole-sequence bootstrap 95% intervals.

| Metric | Causal | Reset every 16 |
|---|---|---|
| PSNR, dB | 27.7914 | 27.4277 |
| PSNR delta vs v3, dB | −2.479 [−3.209, −1.856] | −2.180 [−2.837, −1.662] |
| Cold smooth-crop MSE ratio | 1.155 [0.687, 1.612] | 1.155 [0.687, 1.612] |
| Early smooth-crop MSE ratio | 2.855 [1.477, 4.390] | 2.855 [1.477, 4.390] |
| Warm smooth-crop MSE ratio | 3.625 [2.156, 5.139] | Unavailable |
| FLIP | 0.17455 | 0.17896 |
| FLIP delta vs v3 | +0.06679 [0.05378, 0.07841] | +0.05967 [0.04800, 0.06983] |
| Energy ratio to reference | 1.00011 [0.97649, 1.03602] | 1.00185 [0.97676, 1.04022] |
| Temporal MSE ratio | 0.85410 [0.75496, 0.92508] | 0.69502 [0.62306, 0.75750] |

Cold crop intervals have 999 valid resamples because some draws contain no
selected cold crop. Reset-16 has no warm frames; null is not a passing score.
Warm PSNR is 27.9190 dB, −2.579 [−3.371, −1.822] dB versus v3. Temporal
improvement does not compensate for the spatial, warm-quality and FLIP misses.
The individual lighting-final results also prevent a no-worse claim:

| Protocol / sequence (frame 63) | PSNR delta, dB | FLIP delta | Linear MSE ratio | Temporal MSE ratio |
|---|---:|---:|---:|---:|
| Causal / 6 | −0.1823 | +0.04689 | 0.98756 | 0.99206 |
| Causal / 7 | +2.6008 | −0.03733 | 1.08978 | 1.21770 |
| Reset 16 / 6 | −0.6502 | +0.05864 | 0.91998 | 0.97942 |
| Reset 16 / 7 | −0.9060 | +0.01133 | 1.25587 | 1.13783 |

Both protocols retain the three fixed alpha frames with raw maps, grayscale
PNGs and histograms. Diffuse/specular means at 0:0, 2:31 and 6:63 are
0/0, 0.84160/0.74225, 0.83771/0.70977 (causal), and
0/0, 0.84161/0.74227, 0.83764/0.70983 (reset-16). Inspection of the fixed
0:0 and 2:31 RGB images shows persistent broad color/texture errors despite
reduced speckling; no images or checkpoint are promoted.

The recorded `capacity-40-vs-200-w16-l3-seed1/` comparison holds model, loss,
seed and 60k schedule fixed. For this seed/budget, 200 scenes improve PSNR by
1.754 [1.175, 2.671] dB causal and 1.525 [1.045, 2.305] dB reset-16; FLIP
falls by 0.02467 [0.01977, 0.03061] and 0.02196 [0.01785, 0.02622].
Cold smooth MSE is 1.230 [0.726, 1.628] times the 40-scene run: no established
improvement or regression. Thus data expansion helps overall quality here,
but does not solve the cold-crop failure. The remaining capacity configurations
and best-two second seeds are still required before Decision A.
