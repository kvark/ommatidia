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
The 200-scene first-seed 16×3 run and both full reports are complete; the 16×4
from-scratch retry is training after the execution interruption below, with 32×3 and
32×4 queued. Each trains for 60,000 updates and is followed by both complete
development protocols. Every child run is recorded independently;
any failed command stops the queue. The best two configurations' second seeds
and Decision A remain pending.

Execution recovery: the original 16×4 attempt stopped after 792 finite updates,
before its first optimizer checkpoint; trainer, recorder and queue processes
were absent. No exit status or cause was captured. Preserve its stale manifest
and partial artifacts; `capacity-200-w16-l4-seed1-60000-retry1/` changes only the
output directory and starts the same numerical command from scratch. The
remaining queue and final curve audits now run under the transient user service
`ommatidia-phase5-capacity-20260928.service`, independently of the tool session,
with no automatic restart on failure. This is not a numerical-quality failure.

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

## 16×4 retry: training observations

The immutable first-2,500-update audit records three finite batch-loss spikes
at 1,589–1,591 (4.5442, 43.9215, 1.6969), all from cursor 5 on warm windows
29/33/37 of sequence 50 (`train-objects-00.omd`, scene 5307974659).
At the largest spike, input/target/prediction peaks are 1,031.8/73.1/7,238.0;
the recorded latent state remains finite. Subsequent batch loss through update
2,500 is at most 0.029109. This localizes a transient prediction overshoot,
not its root cause. No batch is skipped and no loss/model/schedule is changed.
The reproducible report is `capacity-w16-l4-warm-outliers-2500-recorded/`;
its source snapshot corrects the first audit's missing stdin-script provenance.

The scheduled causal checkpoints cover all 640 development frames. Recorded
`capacity-w16-l4-seed1-curve-{20000,30000,40000,50000,60000}/` analyses freeze
the corresponding loss prefixes and use the same reference-identity proof and
whole-sequence intervals as the other curves:

| Updates | Last-1,000 mean loss | PSNR | Delta vs previous (95% CI), dB | Delta vs v3 (95% CI), dB |
|---|---:|---:|---|---|
| 10,000 | 0.0063584 | 25.8484 | — | −4.422 [−4.943, −3.892] |
| 20,000 | 0.0055403 | 26.8684 | +1.020 [−0.174, 1.902] | −3.402 [−4.741, −2.478] |
| 30,000 | 0.0045525 | 27.0070 | +0.139 [−0.274, 0.558] | −3.264 [−4.323, −2.429] |
| 40,000 | 0.0042245 | 27.9262 | +0.919 [0.581, 1.276] | −2.344 [−3.097, −1.672] |
| 50,000 | 0.0037667 | 27.7825 | −0.144 [−0.311, 0.006] | −2.488 [−3.259, −1.828] |
| 60,000 | 0.0036426 | 28.2648 | +0.482 [0.104, 1.017] | −2.006 [−2.276, −1.643] |

The first two consecutive-change intervals cross zero, but the 30k→40k gain
excludes zero. The following 40k→50k point estimate declines, with an interval
that includes zero, followed by a positive 50k→60k gain. This is not a
capacity ranking or a reason to change one configuration's fixed budget.
The complete from-scratch run has 480,000 sampled windows (17.4983% cold),
654,256 finite parameters and each Adam moment set, and 2,228,224 finite
carried-state values. Checkpoint/state hashes match and parameters reload
bit-exactly. Four finite losses exceed 1 (maximum 43.92154). Cost is 31,200
convolution FLOPs/output pixel. Final checkpoint SHA-256:
`ff05999a9073809f07203b5756a9551d5a237c24541cf6beb758048ee6f94268`.

## Second capacity result: 200 scenes, 16×4, seed 1

Both full protocols use the final 60,000-update retry checkpoint and all 640
development frames. Ratios compare against learned v3, with whole-sequence
bootstrap 95% intervals:

| Metric | Causal | Reset every 16 |
|---|---|---|
| PSNR, dB | 28.2648 | 27.9967 |
| PSNR delta vs v3, dB | −2.006 [−2.276, −1.643] | −1.611 [−1.761, −1.441] |
| Cold smooth-crop MSE ratio | 1.034 [0.696, 1.452] | 1.034 [0.696, 1.452] |
| Early smooth-crop MSE ratio | 2.186 [1.258, 3.243] | 2.186 [1.258, 3.243] |
| Warm smooth-crop MSE ratio | 2.830 [2.016, 3.902] | Unavailable |
| FLIP | 0.16943 | 0.17246 |
| FLIP delta vs v3 | +0.06167 [0.05031, 0.07361] | +0.05318 [0.04292, 0.06375] |
| Energy ratio to reference | 0.98283 [0.96310, 1.00506] | 0.98314 [0.96321, 1.00516] |
| Temporal MSE ratio | 0.81871 [0.72568, 0.88317] | 0.66458 [0.60137, 0.72096] |

Cold crop intervals have 999 valid resamples; reset-16 warm coverage remains
null. Causal warm PSNR is 28.3815 dB, −2.117 [−2.490, −1.556] dB behind v3.
Energy point estimates are in range, but their intervals are not contained
within it. Cold/warm crop and FLIP targets still miss. The lighting-final
condition also remains unmet despite improved final-frame PSNR:

| Protocol / sequence (frame 63) | PSNR delta, dB | FLIP delta | Linear MSE ratio | Temporal MSE ratio |
|---|---:|---:|---:|---:|
| Causal / 6 | +0.9437 | +0.03434 | 0.71551 | 1.02376 |
| Causal / 7 | +3.5597 | −0.03633 | 1.03454 | 1.20202 |
| Reset 16 / 6 | +0.5265 | +0.04560 | 0.66113 | 1.00994 |
| Reset 16 / 7 | +0.0654 | +0.01110 | 1.19875 | 1.11746 |

The `capacity-w16-l4-report-audit/` check verifies exact reproduction of all
640 causal metric rows, all 1,280 verified shared references, and the 12 raw
alpha maps, histograms and PNG headers. Cold alpha is zero. Diffuse/specular
means at camera 2:31 and lighting 6:63 are 0.83295/0.77677 and
0.81845/0.75490 (causal), and 0.83296/0.77678 and 0.81840/0.75487 (reset-16).
Fixed cold/camera image inspection shows reduced speckling but persistent
material-color and detail errors. Full reports are
`capacity-w16-l4-seed1-score-{causal,reset16}/`; no images or weights are promoted.

The recorded `capacity-w16-l3-vs-l4-seed1/` comparison verifies identical
executable, ordered data/provenance hashes, loss, seed and 60k schedule, with
only depth changed. Four levels improve causal/reset-16 PSNR by
0.473 [0.113, 0.980]/0.569 [0.183, 1.142] dB and reduce FLIP by
0.00512 [0.00124, 0.00931]/0.00649 [0.00174, 0.01200]. However, cold smooth
MSE is 0.89484 [0.71271, 1.01260] times the three-level result: its paired
difference interval [−0.00031211, +0.00001200] includes zero. At 25.5% extra
FLOPs, it does not yet meet the plan's exception to preferring the cheaper
configuration. Width-32 runs and the best-two second seeds remain required
before selection and Decision A.

## 32×3: training observations

Scheduled causal checkpoints cover all 640 development frames. Recorded
`capacity-w32-l3-seed1-curve-{10000,20000,30000,40000,50000,60000}/` analyses
freeze the corresponding loss prefixes and use the same reference-identity
proof and whole-sequence bootstrap as the other curves:

| Updates | Last-1,000 mean loss | PSNR | Delta vs previous (95% CI), dB | Delta vs v3 (95% CI), dB |
|---|---:|---:|---|---|
| 10,000 | 0.0061488 | 26.5075 | — | −3.763 [−4.154, −3.347] |
| 20,000 | 0.0057168 | 27.2636 | +0.756 [0.211, 1.234] | −3.007 [−3.803, −2.292] |
| 30,000 | 0.0041916 | 27.2123 | −0.051 [−0.491, 0.378] | −3.058 [−4.207, −2.100] |
| 40,000 | 0.0038097 | 27.8720 | +0.660 [0.310, 1.026] | −2.399 [−3.282, −1.610] |
| 50,000 | 0.0033609 | 28.2899 | +0.418 [0.163, 0.642] | −1.981 [−2.885, −1.208] |
| 60,000 | 0.0032854 | 28.3656 | +0.076 [−0.117, 0.303] | −1.905 [−2.609, −1.272] |

The 10k→20k, 30k→40k and 40k→50k improvement intervals exclude zero;
20k→30k and 50k→60k include zero despite falling training loss. The completed
from-scratch run has 480,000 sampled windows (17.4983% cold), 657,840 finite
parameters and each Adam moment set, and 2,228,224 finite carried-state values.
Checkpoint/state hashes match and parameters reload bit-exactly. Cost is
79,680 convolution FLOPs/output pixel. The fixed budget is unchanged and no
intermediate checkpoint is selected. Final checkpoint SHA-256:
`18a3565d7de67d301dc0fa4c68ea2858926457582e421d023ab61575060ac597`.

Two finite batch losses exceed 1 at updates 10,737/10,738 (3.6016/3.3204),
both dominated by warm cursor 4 on sequence 50, windows 27/31. Recorded
predicted-lobe peaks are 3,827.1/2,159.9 against target peaks 65.5/65.0;
all recorded input/target/prediction/latent summaries have zero nonfinite
values. No further losses above 1 occur through 60,000. The outlier telemetry
is snapshotted identically in the 20k–50k analyses. This records a
transient prediction overshoot, not its root cause; no batch, loss, model or
schedule is changed.

## Third capacity result: 200 scenes, 32×3, seed 1

Both full protocols use the final 60,000-update checkpoint and all 640
development frames. Ratios compare against learned v3, with whole-sequence
bootstrap 95% intervals:

| Metric | Causal | Reset every 16 |
|---|---|---|
| PSNR, dB | 28.3656 | 28.0262 |
| PSNR delta vs v3, dB | −1.905 [−2.609, −1.272] | −1.582 [−2.214, −1.069] |
| Cold smooth-crop MSE ratio | 1.153 [0.695, 1.648] | 1.153 [0.695, 1.648] |
| Early smooth-crop MSE ratio | 2.242 [1.235, 3.491] | 2.242 [1.235, 3.491] |
| Warm smooth-crop MSE ratio | 2.925 [2.102, 3.910] | Unavailable |
| FLIP | 0.16453 | 0.16893 |
| FLIP delta vs v3 | +0.05678 [0.04354, 0.06928] | +0.04965 [0.03745, 0.06187] |
| Energy ratio to reference | 0.99388 [0.97095, 1.02624] | 0.99081 [0.96615, 1.02535] |
| Temporal MSE ratio | 0.79689 [0.70604, 0.85753] | 0.68765 [0.61587, 0.74322] |

Cold crop intervals have 999 valid resamples; reset-16 warm coverage remains
null. Causal warm PSNR is 28.4976 dB, −2.001 [−2.785, −1.227] dB behind v3.
Energy point estimates are in range, but their intervals are not contained
within it. Cold/warm crop and FLIP targets still miss. Lower aggregate temporal
error does not satisfy the lighting-final condition:

| Protocol / sequence (frame 63) | PSNR delta, dB | FLIP delta | Linear MSE ratio | Temporal MSE ratio |
|---|---:|---:|---:|---:|
| Causal / 6 | +0.1518 | +0.04655 | 0.69788 | 1.11086 |
| Causal / 7 | +3.2130 | −0.04014 | 1.01988 | 1.01178 |
| Reset 16 / 6 | −0.1922 | +0.05634 | 0.63458 | 1.09657 |
| Reset 16 / 7 | +0.1173 | +0.00408 | 1.17520 | 0.94428 |

The `capacity-w32-l3-report-audit/` check verifies exact reproduction of all
640 causal metric rows, all 1,280 verified shared references, and the 12 raw
alpha maps, histograms and PNG headers. Cold alpha is zero. Diffuse/specular
means at camera 2:31 and lighting 6:63 are 0.89197/0.85868 and
0.86740/0.83791 (causal), and 0.89202/0.85895 and 0.86700/0.83747 (reset-16).
Fixed cold/camera image inspection shows less speckling than v3 but persistent
material-color and detail errors. Full reports are
`capacity-w32-l3-seed1-score-{causal,reset16}/`; no images or weights are promoted.
The 32×4 run and best-two second seeds remain required before selection and
Decision A; these results alone do not justify a cost exception.

The recorded `capacity-w16-l3-vs-w32-l3-seed1/` and
`capacity-w16-l4-vs-w32-l3-seed1/` comparisons verify identical executable,
ordered corpus/provenance hashes, loss, seed and 60k schedule, allowing only
the registered width/depth differences. The generalized comparison wrapper
first reproduces the existing width-16 depth comparison exactly, including
both paired protocol reports and cost. Cold ratios below compare 32×3 against
the named baseline, using paired raw crop sums rather than ratios of ratios:

| Baseline | FLOPs multiplier | Causal PSNR gain (95% CI), dB | Reset-16 PSNR gain (95% CI), dB | Cold MSE ratio (95% CI), both protocols |
|---|---:|---|---|---|
| 16×3 | 3.205× | +0.574 [0.418, 0.747] | +0.599 [0.440, 0.754] | 0.99789 [0.97097, 1.02240] |
| 16×4 | 2.554× | +0.101 [−0.431, 0.488] | +0.030 [−0.514, 0.421] | 1.11516 [0.99860, 1.36968] |

Neither cold comparison meets the ≥10% improvement requirement, and both
paired difference intervals include zero. Thus 32×3 does not meet the larger
configuration's cost exception against either completed smaller configuration.
Against 16×3, FLIP improves by 0.01002 [0.00664, 0.01442] causal and
0.01002 [0.00619, 0.01446] reset-16. Against 16×4, neither PSNR nor FLIP
improvement is established; reset-16 temporal MSE is higher, ratio 1.03471
[1.01403, 1.05640]. This is first-seed evidence, not final selection.

## 32×4: early training observations

The managed queue started the final first-seed configuration from scratch,
with the same 200-scene corpus, seed and 60,000-update schedule. The recorded
`capacity-w32-l4-outliers-2000/` audit freezes the first 2,000 updates and
cursor diagnostics. Six finite batch losses exceed 1 in two episodes:

- Updates 1,022/1,023: batch losses 13.2411/2.3460, dominated by cursor 7
  on sequence 173 (lighting), gain 3.6784. The first window resets and the
  second carries history. Last-frame prediction peaks are 789.4/356.8 versus
  target-window peaks 98.2/98.1.
- Updates 1,497–1,500: batch losses 8,868.602/837.659/82.815/4.368, dominated
  by cursor 6 on sequence 65 (camera), gain 3.8918. The first window resets;
  the following three carry history. The first event's last-frame predicted
  lobe peak is 20,873.0 versus target-window peak 29.4 and input peak 1,008.0.

All recorded input, target, prediction and latent summaries are finite. After
update 1,500, batch loss through 2,000 is at most 0.065291. The source snapshots
retain sequence identity, crop origin and frame windows. This identifies
transient prediction overshoots, not their root cause; no batch is skipped,
no clipping is added, and the model/loss/schedule remain unchanged. The larger
early spikes are retained as evidence, not treated as a successful quality gate.

Scheduled causal checkpoints cover all 640 development frames. Recorded
`capacity-w32-l4-seed1-curve-{10000,20000,30000,40000,50000,60000}/` analyses freeze
the corresponding loss prefixes and outlier telemetry. They use the same
reference-identity proof and whole-sequence bootstrap as the other curves:

| Updates | Last-1,000 mean loss | PSNR | Delta vs previous (95% CI), dB | Delta vs v3 (95% CI), dB |
|---|---:|---:|---|---|
| 10,000 | 0.0084783 | 26.0701 | — | −4.201 [−4.948, −3.615] |
| 20,000 | 0.0074709 | 26.9652 | +0.895 [0.632, 1.216] | −3.306 [−4.053, −2.601] |
| 30,000 | 0.0042334 | 27.3616 | +0.396 [0.101, 0.733] | −2.909 [−3.681, −2.167] |
| 40,000 | 0.0037826 | 28.1300 | +0.768 [0.464, 1.019] | −2.141 [−2.784, −1.507] |
| 50,000 | 0.0033130 | 28.6405 | +0.510 [0.298, 0.685] | −1.630 [−2.189, −1.047] |
| 60,000 | 0.0031770 | 28.6585 | +0.018 [−0.335, 0.541] | −1.612 [−1.935, −1.229] |

The first four consecutive improvement intervals exclude zero; the final
50k→60k interval includes zero despite falling training loss. The completed
from-scratch run has 480,000 sampled windows (17.4983% cold), 2,575,664 finite
parameters and each Adam moment set, and 2,228,224 finite carried-state values.
Checkpoint/state hashes match and parameters reload bit-exactly. Cost is
105,024 convolution FLOPs/output pixel. The fixed budget is unchanged and
no intermediate checkpoint is selected. Final checkpoint SHA-256:
`a9930ab4cac9954009b9ffd6c7e50befe9cbfdc436bc5002c910e5b0593f7ef8`.

The frozen 10k prefix contains eight batch losses above 1. In addition to the six
early events, update 3,260 has batch loss 3.2651 (warm cursor 4, sequence 14),
and update 9,653 has batch loss 1.5453 (warm cursor 5, sequence 144). Recorded
last-frame predicted-lobe peaks are 3,178.7/948.5 versus target-window peaks
54.1/396.1; prediction and latent summaries for those cursors are finite.
Their telemetry is retained in the 10k analysis. No root cause or training
change is inferred from these observations.

Through 20k, the count reaches eleven, with additional batch losses
1.0737/2.0109/2.3051 at updates 10,736/12,023/16,332. These are dominated by
one warm and two resetting windows, respectively; the recorded prediction
and latent summaries of those cursors are finite. No additional losses above
1 occur through 60k; the 20k–50k telemetry snapshots are identical and
retain all eleven events. Training remains unchanged.

## Fourth capacity result: 200 scenes, 32×4, seed 1

Both full protocols use the final 60,000-update checkpoint and all 640
development frames. Ratios compare against learned v3, with whole-sequence
bootstrap 95% intervals:

| Metric | Causal | Reset every 16 |
|---|---|---|
| PSNR, dB | 28.6585 | 28.3445 |
| PSNR delta vs v3, dB | −1.612 [−1.935, −1.229] | −1.263 [−1.496, −1.018] |
| Cold smooth-crop MSE ratio | 1.382 [0.860, 2.074] | 1.382 [0.860, 2.074] |
| Early smooth-crop MSE ratio | 2.080 [1.254, 3.144] | 2.080 [1.254, 3.144] |
| Warm smooth-crop MSE ratio | 2.500 [2.204, 2.738] | Unavailable |
| FLIP | 0.16279 | 0.16619 |
| FLIP delta vs v3 | +0.05504 [0.04223, 0.06922] | +0.04690 [0.03431, 0.06073] |
| Energy ratio to reference | 0.96532 [0.94990, 0.98030] | 0.96689 [0.94941, 0.98299] |
| Temporal MSE ratio | 0.77834 [0.69620, 0.83439] | 0.66091 [0.59756, 0.71279] |

Cold crop intervals have 999 valid resamples; reset-16 warm coverage remains
null. Causal warm PSNR is 28.7864 dB, −1.712 [−2.179, −1.125] dB behind v3.
Energy point estimates are below the target range. Cold/warm crop and FLIP
targets still miss. Lower aggregate temporal error does not satisfy the
lighting-final condition:

| Protocol / sequence (frame 63) | PSNR delta, dB | FLIP delta | Linear MSE ratio | Temporal MSE ratio |
|---|---:|---:|---:|---:|
| Causal / 6 | +0.8895 | +0.03809 | 0.63187 | 0.94332 |
| Causal / 7 | +4.5396 | −0.04077 | 0.97070 | 1.02551 |
| Reset 16 / 6 | +0.5012 | +0.04835 | 0.58141 | 0.92780 |
| Reset 16 / 7 | +1.0016 | +0.00965 | 1.12720 | 0.95188 |

The `capacity-w32-l4-report-audit/` check verifies exact reproduction of all
640 causal metric rows, all 1,280 verified shared references, and the 12 raw
alpha maps, histograms and PNG headers. Cold alpha is zero. Diffuse/specular
means at camera 2:31 and lighting 6:63 are 0.87267/0.82083 and
0.87514/0.80406 (causal), and 0.87266/0.82080 and 0.87504/0.80417 (reset-16).
Fixed cold/camera image inspection shows less speckling than v3 but persistent
material-color and detail errors, notably the gold checker object reconstructed
as brown/pink. Full reports are `capacity-w32-l4-seed1-score-{causal,reset16}/`;
no images or weights are promoted.

The three `capacity-{w16-l3,w16-l4,w32-l3}-vs-w32-l4-seed1/` comparisons
verify identical executable, ordered corpus/provenance hashes, loss, seed
and schedule, allowing only the registered width/depth differences. Cold
ratios compare 32×4 against each baseline using paired raw crop sums:

| Baseline | FLOPs multiplier | Causal PSNR gain (95% CI), dB | Reset-16 PSNR gain (95% CI), dB | Cold MSE ratio (95% CI), both protocols |
|---|---:|---|---|---|
| 16×3 | 4.224× | +0.867 [0.364, 1.613] | +0.917 [0.391, 1.709] | 1.19569 [1.09444, 1.29085] |
| 16×4 | 3.366× | +0.394 [0.173, 0.664] | +0.348 [0.145, 0.598] | 1.33621 [1.21099, 1.53626] |
| 32×3 | 1.318× | +0.293 [−0.205, 1.049] | +0.318 [−0.159, 1.075] | 1.19823 [1.12162, 1.26296] |

All three cold paired-difference intervals are strictly positive: 32×4 has
worse cold smooth-crop error than every smaller configuration. None earns
the larger-cost exception. Its FLIP gains against both width-16 configurations
are established, but neither PSNR nor FLIP gain against 32×3 is established.
Those improvements do not replace the plan's cold-error cost criterion.

## Second-seed allocation

The recorded `second-seed-selection/` audit applies the frozen cost rule to
all six completed first-seed pairs. No larger configuration satisfies both
≥10% cold-MSE improvement and a paired-difference interval below zero.
Therefore the two cheapest configurations, **16×3 and 16×4**, receive seed 2.
The latter has the best first-seed cold point estimate; its 10.5% gain over
16×3 remains uncertain (ratio 0.89484 [0.71271, 1.01260]), making the repeat
particularly relevant. The width-32 PSNR gains do not justify their cost
under this rule.

`second-seed-recipe-check/` verifies both training commands differ from
their completed seed-1 counterparts only in seed and fresh output path.
Data, loss, 60,000 updates, batch 8, unroll 4, crop 64 and evaluation remain
unchanged; neither command resumes weights or optimizer state. The durable
`ommatidia-phase5-second-seeds-20260928.service` queue starts 16×3 then 16×4,
with both full protocols and final curve audits after each. It stops on
failure and never restarts a job automatically. Both repeats, final capacity
selection and Decision A remain required. Confirmation remains untouched.

## 16×3 second-seed curve

Scheduled checkpoints cover all 640 causal development frames.
`capacity-w16-l3-seed2-curve-{10000,20000,30000,40000,50000}/` freezes each loss prefix and
the empty outlier log, using the same reference-identity proof and
whole-sequence bootstrap as seed 1:

| Updates | Last-1,000 mean loss | PSNR | Delta vs previous (95% CI), dB | Delta vs v3 (95% CI), dB |
|---|---:|---:|---|---|
| 10,000 | 0.0061091 | 25.2895 | — | −4.981 [−6.506, −3.902] |
| 20,000 | 0.0050200 | 26.6589 | +1.369 [0.522, 2.523] | −3.612 [−4.237, −3.011] |
| 30,000 | 0.0049041 | 26.5194 | −0.139 [−0.783, 0.494] | −3.751 [−4.618, −2.870] |
| 40,000 | 0.0042379 | 27.4749 | +0.955 [0.461, 1.483] | −2.796 [−3.463, −2.163] |
| 50,000 | 0.0039092 | 27.2580 | −0.217 [−0.471, −0.021] | −3.013 [−3.923, −2.222] |

The 10k→20k and 30k→40k improvement intervals exclude zero; the 20k→30k
interval includes zero. The 40k→50k PSNR decline is established despite
falling training loss. Warm PSNR at 50k is 27.3511 dB,
−3.147 [−4.178, −2.157] dB behind v3. Energy is 0.98405 [0.95894, 1.02612]:
the point estimate is in range, not its entire interval. Temporal MSE ratio
is 0.84661 [0.75077, 0.91080]. No batch losses above 1 occur through 50k.
These metrics-only checkpoints do not claim crop/FLIP results or select
a model. The fresh seed-2 run continues to the
unchanged 60,000-update budget.
