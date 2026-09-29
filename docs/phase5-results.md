# Phase 5: training ladder

The six-run capacity matrix is complete; **Decision A fails**. Select 16×4
under the frozen cost rule and stop for the [owner handoff](#decision-a-owner-handoff)
before the one fallback. The owner approved **F1 on 2026-09-29**; the replacement
is implemented and passes both-GPU correctness, with fresh training next.
Phase 5 is not complete. The thresholds in
[PLAN.md §9](../PLAN.md#9-phase-5-training-ladder-and-decision-point-a-3-days)
are unchanged. Every run, including failed diagnostics, is in the [ledger](experiments.md).
The sections below preserve the chronological evidence; earlier pending-work
notes are superseded by the final decision. No confirmation evaluation or
gallery promotion belongs to this phase.

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

Final capacity comparison will include both registered seeds equally, using
the final 60k checkpoints. Pool raw crop error sums, not ratios; bootstrap
the original ten sequence identities jointly across seeds so a repeated
scene is not treated as independent. Keep both per-seed comparisons visible.
The combined interval measures scene uncertainty conditional on these two
seeds, not the population variance over possible training seeds. The frozen
≥10% cold improvement plus paired-difference interval rule is unchanged;
the pooled result concerns capacity, not an ensemble prediction or a quality
pass. This aggregation is specified before either second-seed full report.
The helper also requires identical reference hashes, reset ages and v3 crop
control evidence across seeds, in addition to the existing per-seed checks.

## 16×3 second-seed curve

Scheduled checkpoints cover all 640 causal development frames.
`capacity-w16-l3-seed2-curve-{10000,20000,30000,40000,50000,60000}/` freezes each loss prefix and
the empty outlier log, using the same reference-identity proof and
whole-sequence bootstrap as seed 1:

| Updates | Last-1,000 mean loss | PSNR | Delta vs previous (95% CI), dB | Delta vs v3 (95% CI), dB |
|---|---:|---:|---|---|
| 10,000 | 0.0061091 | 25.2895 | — | −4.981 [−6.506, −3.902] |
| 20,000 | 0.0050200 | 26.6589 | +1.369 [0.522, 2.523] | −3.612 [−4.237, −3.011] |
| 30,000 | 0.0049041 | 26.5194 | −0.139 [−0.783, 0.494] | −3.751 [−4.618, −2.870] |
| 40,000 | 0.0042379 | 27.4749 | +0.955 [0.461, 1.483] | −2.796 [−3.463, −2.163] |
| 50,000 | 0.0039092 | 27.2580 | −0.217 [−0.471, −0.021] | −3.013 [−3.923, −2.222] |
| 60,000 | 0.0040278 | 27.8122 | +0.554 [0.438, 0.666] | −2.459 [−3.285, −1.749] |

The 10k→20k and 30k→40k improvement intervals exclude zero; the 20k→30k
interval includes zero. The 40k→50k PSNR decline is established despite
falling training loss; the 50k→60k recovery also excludes zero, while the
last-1,000 mean loss rises slightly. No batch losses above 1 occur in the
completed 60k run (maximum 0.256878). These metrics-only checkpoints do
not claim crop/FLIP results or select a model.

## 16×3 second-seed final report

`capacity-200-w16-l3-seed2-60000/` completed from scratch at 60,000 updates,
480,000 sampled windows and 17.5025% cold windows. Its final audit verifies
174,576 finite parameters and each Adam tensor set, 2,228,224 finite carry
values, matching checkpoint hashes and bit-exact parameter reload.
Checkpoint SHA-256:
`4cdae95d890f0072d273a975412db51269ad4117eb667bd932ddcbdcd5517c37`.
The immutable loss-prefix SHA-256 is
`a13e4a3f50d0d3801e30a5a730d7663c7742eb3c2eb05da570c4206d00736a30`.
Cost remains 24,864 convolution FLOPs/output pixel (1.629487 GFLOP/frame).

All four `capacity-w16-l3-seed2-{dev,score}-{causal,reset16}` runs completed
with zero validation errors. Against the same learned v3 control, using
whole-sequence paired 95% intervals:

| Metric | Causal | Reset every 16 |
|---|---:|---:|
| PSNR | 27.8122 dB | 27.5007 dB |
| PSNR delta vs v3 | −2.459 [−3.285, −1.749] dB | −2.107 [−2.792, −1.524] dB |
| Cold smooth-crop MSE ratio | 1.004 [0.721, 1.414] | 1.004 [0.721, 1.414] |
| Early smooth-crop MSE ratio | 2.452 [1.117, 3.767] | 2.452 [1.117, 3.767] |
| Warm smooth-crop MSE ratio | 3.182 [2.286, 3.625] | null: no warm coverage |
| FLIP | 0.173892 | 0.178305 |
| FLIP delta vs v3 | +0.066139 [0.049928, 0.081457] | +0.059019 [0.044478, 0.072144] |
| Energy ratio | 0.99880 [0.97501, 1.03850] | 1.00123 [0.97708, 1.04095] |
| Temporal MSE ratio vs v3 | 0.84234 [0.73999, 0.91140] | 0.68347 [0.61273, 0.74622] |

Causal warm PSNR is 27.9026 dB: −2.596 [−3.512, −1.729] dB versus v3,
well outside the −0.1 dB allowance. The cold point misses ≤0.80; the warm
crop and FLIP misses remain. Energy point estimates are in range, but not
their entire intervals. Cold crops have four sequences/eight rectangles
and 999 valid resamples; identical early/cold entries reflect the frozen
crop selections, not extra reset-frame coverage.

Lighting-final frame 63 still fails the individual-sequence condition:

| Protocol / sequence | PSNR delta | FLIP delta | Linear MSE ratio | Temporal MSE ratio |
|---|---:|---:|---:|---:|
| Causal / 6 | −0.3723 dB | +0.047928 | 1.18898 | 0.98563 |
| Causal / 7 | +2.3424 dB | −0.028678 | 1.03360 | 1.19392 |
| Reset-16 / 6 | −0.8018 dB | +0.059043 | 1.10822 | 0.96921 |
| Reset-16 / 7 | −1.1711 dB | +0.019624 | 1.19244 | 1.11612 |

`capacity-w16-l3-seed2-report-audit/` verifies 640 exact causal metric
rows, 1,280 shared references and twelve finite alpha maps, exact f32
histograms and 256×256 PNGs. Alpha diffuse/specular means at frames
0:0, 2:31 and 6:63 are respectively 0/0, 0.85441/0.74539 and
0.84835/0.73940 causal; reset-16 gives 0/0, 0.85444/0.74544 and
0.84823/0.73936. Visual inspection of the fixed causal images still shows
mottled surfaces and the gold patterned object reconstructed purple/brown;
the cleaner temporal aggregate does not imply restored color or detail.

The managed queue has started the fresh 16×4 seed-2 run at the same 60k
budget. Both-seed capacity comparison and final Decision A remain open;
no confirmation evaluation or README/gallery promotion occurred.

## 16×4 second-seed telemetry

The first-4,800-update snapshot (`capacity-w16-l4-seed2-outliers-4800/`)
contains one finite batch loss above 1: 1.030057 at update 4,703. Cursor 0
has loss 8.20894 in a reset-start four-frame window, sequence 50/frame 24,
gain 2.70308, origin [28,19], object scene seed 5307974659. Its last-frame
prediction peak is 482.35, input peak 729.83 and target peak 106.35;
all recorded input/target/prediction/latent summaries are finite. Following
batch losses through 4,800 are ≤0.175524. This does not establish the cause
or locate the peak at the reset frame. The fixed recipe continues unchanged,
without skips, clamps or restart; the immutable loss-prefix SHA-256 is
`53b3c63e23f4f47d0c77883dffb1802ab0ae953f0ea8dd6159dfcaf1e1d5b3c2`.

## 16×4 second-seed curve

`capacity-w16-l4-seed2-curve-{10000,20000,30000,40000,50000,60000}/` covers all 640 causal development
frames with the same reference-identity proof and whole-sequence bootstrap.
Each report freezes its loss prefix and the one-event outlier telemetry.

| Updates | Last-1,000 mean loss | PSNR | Delta vs previous (95% CI), dB | Delta vs v3 (95% CI), dB |
|---|---:|---:|---|---|
| 10,000 | 0.0061000 | 25.6862 | — | −4.585 [−5.547, −3.674] |
| 20,000 | 0.0051748 | 26.8478 | +1.162 [0.478, 1.996] | −3.423 [−3.751, −3.079] |
| 30,000 | 0.0048583 | 27.0901 | +0.242 [−0.209, 0.630] | −3.181 [−3.765, −2.564] |
| 40,000 | 0.0048389 | 27.6519 | +0.562 [0.322, 0.830] | −2.619 [−3.317, −1.975] |
| 50,000 | 0.0037824 | 27.6353 | −0.017 [−0.305, 0.354] | −2.635 [−3.085, −2.107] |
| 60,000 | 0.0038524 | 27.9233 | +0.288 [0.084, 0.453] | −2.347 [−2.913, −1.759] |

Warm PSNR is 25.7272 dB, −4.771 [−5.843, −3.677] dB behind v3.
Energy is 0.98470 [0.94197, 1.04113]: the point is in range, not its entire
interval. Temporal MSE ratio is 0.86056 [0.75896, 0.92494]. The one finite
outlier above is the only loss above 1 through 10k. The loss-prefix SHA-256
is `8988ac6f5013fd2429ebb65f375262444b5024f35b666bb960ae7c5489766264`.
No crop/FLIP result, capacity selection or quality pass is inferred from
this interim checkpoint; the unchanged fresh 60k run continues.

At 20k, the PSNR improvement over 10k excludes zero, but warm PSNR remains
26.8966 dB, −3.601 [−4.028, −3.124] dB behind v3. Energy is
0.98980 [0.97642, 1.00385], with the point but not the full interval in
range. Temporal MSE ratio is 0.89182 [0.76935, 0.97858]. The outlier count
remains one through 20k. Its immutable loss-prefix SHA-256 is
`53323d3640a3e9c7556d39d0e02895667f48addc84d203c84b3388ff4a3367ba`.

At 30k, the gain over 20k does not exclude zero despite lower training loss.
Warm PSNR is 27.1773 dB, −3.321 [−4.038, −2.516] dB behind v3. Energy is
0.98582 [0.95263, 1.01851], with the point but not the full interval in
range; temporal MSE ratio is 0.89064 [0.79488, 0.95658]. The outlier count
remains one. The immutable 30k loss-prefix SHA-256 is
`ead35e66c4c5bb239df2c2eb416e98311d7418438bd1cf0455a1827e369a6c34`.
The fixed 60k budget continues; no early checkpoint is selected.

At 40k, the improvement over 30k excludes zero. Warm PSNR is 27.7575 dB,
−2.741 [−3.552, −1.910] dB behind v3. Energy is
0.99579 [0.97194, 1.02996], with the point but not the full interval in
range; temporal MSE ratio is 0.86567 [0.77566, 0.92881]. The outlier count
remains one. The immutable 40k loss-prefix SHA-256 is
`c471d25d1d2656aff166d93413806b5fdbc02c44bf72cd852590c717dbe2d719`.
This is still a metrics-only checkpoint, not a crop/FLIP result or selection.

At 50k, the PSNR change from 40k includes zero while training loss falls.
Warm PSNR is 27.7760 dB, −2.722 [−3.297, −2.005] dB behind v3. Energy is
0.96962 [0.94912, 0.99634], with its point below the [0.98, 1.02] target;
temporal MSE ratio is 0.82328 [0.73891, 0.88342]. The outlier count remains
one. The immutable 50k loss-prefix SHA-256 is
`7aa9da9b1dba15415b3d66a4c03ecc87938d205950776b7c911f81b4fd4d9962`.
The same 60k schedule continues; neither the lower training loss nor the
temporal result establishes a spatial-quality pass.

## 16×4 second-seed final report

`capacity-200-w16-l4-seed2-60000/` completed from scratch at 60,000 updates,
480,000 windows and 17.5025% cold windows. The final audit verifies 654,256
finite parameters and each Adam tensor set, 2,228,224 finite carry values,
matching hashes and bit-exact parameter reload. The one finite loss outlier
at update 4,703 remains the only loss above 1. Cost is 31,200 convolution
FLOPs/output pixel (2.044723 GFLOP per 256×256 frame).
Checkpoint SHA-256:
`aace1d510f4f31e1f13e95bbaf45d594e8a1b0eb54f9780a177e5fffde31cde1`.
The final immutable loss-prefix SHA-256 is
`d12e71bc0ef1fa79dc1ed200c975469ce0d567f69be1fd168803bf42e00b7529`.

All four `capacity-w16-l4-seed2-{dev,score}-{causal,reset16}` runs completed
with zero validation errors. Against learned v3, with paired whole-sequence
95% confidence intervals:

| Metric | Causal | Reset every 16 |
|---|---:|---:|
| PSNR | 27.9233 dB | 27.6257 dB |
| PSNR delta vs v3 | −2.347 [−2.913, −1.759] dB | −1.982 [−2.460, −1.550] dB |
| Cold smooth-crop MSE ratio | 0.883 [0.562, 1.518] | 0.883 [0.562, 1.518] |
| Early smooth-crop MSE ratio | 2.085 [1.217, 3.134] | 2.085 [1.217, 3.134] |
| Warm smooth-crop MSE ratio | 3.253 [2.442, 3.709] | null: no warm coverage |
| FLIP | 0.171092 | 0.174878 |
| FLIP delta vs v3 | +0.063340 [0.049150, 0.077084] | +0.055591 [0.042912, 0.067821] |
| Energy ratio | 0.99533 [0.97559, 1.02589] | 0.99709 [0.97761, 1.02785] |
| Temporal MSE ratio vs v3 | 0.81327 [0.72724, 0.87492] | 0.68377 [0.62230, 0.73810] |

Causal warm PSNR is 28.0277 dB, a delta of −2.470 [−3.146, −1.690] dB.
Energy point estimates recover into range at 60k; neither full interval is
inside [0.98, 1.02]. Cold crops still cover four sequences/eight rectangles,
with 999 valid resamples. Identical causal/reset cold and early crops come
from the fixed selections, not additional independent reset-frame coverage.

| Protocol / sequence (frame 63) | PSNR delta | FLIP delta | Linear MSE ratio | Temporal MSE ratio |
|---|---:|---:|---:|---:|
| Causal / 6 | −0.0308 dB | +0.043931 | 0.78044 | 1.05789 |
| Causal / 7 | +3.8485 dB | −0.038271 | 1.02579 | 1.19317 |
| Reset-16 / 6 | −0.4419 dB | +0.054906 | 0.72277 | 1.04265 |
| Reset-16 / 7 | +0.2874 dB | +0.010530 | 1.19175 | 1.11418 |

`capacity-w16-l4-seed2-report-audit/` verifies 640 exact causal metric rows,
1,280 shared references, and twelve finite alpha maps with exact f32
histograms and valid 256×256 PNGs. Diffuse/specular alpha means at 0:0,
2:31 and 6:63 are 0/0, 0.86443/0.81578 and 0.84929/0.77055 causal;
reset-16 gives 0/0, 0.86441/0.81584 and 0.84896/0.76923. Fixed-image
inspection still shows mottled surfaces, purple/brown reconstruction of the
gold patterned object, and blurred lighting/detail. No visual pass is claimed.

## Final capacity selection

The recorded `capacity-w16-l3-vs-l4-two-seeds/` comparison uses the method
specified before either second-seed full report. Its fourteen synthetic
checks pass; per-seed data/recipe identity holds and seed-1 results exactly
reproduce the previous comparison. Both protocols give the same cold result:

| Training seeds | Cold MSE ratio, 16×4 / 16×3 (95% CI) | Frozen cost exception |
|---|---:|---|
| 1 | 0.89484 [0.71271, 1.01260] | No: paired difference includes zero |
| 2 | 0.87875 [0.63434, 1.07345] | No: paired difference includes zero |
| Both, raw errors pooled | 0.88736 [0.76014, 0.99008] | Yes |

**Select 16×4.** Pooled cold error falls 11.26%; the paired MSE difference
is −0.00017930 [−0.00038048, −0.00001432], satisfying the ≥10% point gain
plus difference-CI-below-zero rule for 25.5% extra compute (31,200 versus
24,864 convolution FLOPs/output pixel). The interval does **not** establish
that the gain is at least 10%; that was not the frozen requirement.
Repeated scene IDs are resampled jointly across seeds. This is scene
uncertainty conditional on two seeds, not eight independent cold-crop scenes,
training-seed population uncertainty, or an ensemble prediction.

Seed 2's PSNR gain from four levels is +0.111 [−0.123, 0.401] dB causal
and +0.125 [−0.077, 0.400] dB reset-16; neither establishes a gain alone.
Its FLIP differences also include zero. The selection is based on the
specified pooled cold-error/cost rule, not a claim of universal improvement.

## Decision A owner handoff

**Fail; Phase 5 remains incomplete.** The recorded `decision-a-audit/`
cross-checks 61 completed run manifests and their ledger coverage, the sanity
thresholds, all six fixed-budget checkpoint curves/hashes and both full
development protocols. It retains metrics for every candidate; none passes.
The selected capacity's seed 2 has the best cold error and is the checkpoint
reported below. Seed 1 has better overall PSNR (28.2648 dB) and FLIP (0.169426),
but worse cold error (1.03389); it also fails. Metrics are never mixed across
checkpoints to manufacture a passing candidate.

| Requirement vs learned v3 | Seed-2 evidence | Outcome |
|---|---|---|
| Cold smooth-crop MSE ≤0.80 | 0.88254 [0.56232, 1.51816], both protocols | Fail |
| Warm PSNR CI lower bound ≥−0.1 dB | Delta −2.470 [−3.146, −1.690] dB | Fail |
| Warm smooth-crop MSE ≤1.0 | 3.25315 [2.44194, 3.70943] | Fail |
| Aggregate temporal MSE ≤1.0 | 0.81327 causal / 0.68377 reset-16, both intervals below 1 | Pass |
| Each lighting-final frame no worse | Temporal ratios 1.058/1.193 causal, 1.043/1.114 reset-16; other misses above | Fail |
| Energy within [0.98, 1.02] | 0.99533 / 0.99709; full intervals extend outside range | Points pass; interval-level containment unproven |
| FLIP ≤v3 | 0.171092 vs 0.107752 causal; 0.174878 vs 0.119287 reset-16; delta intervals above zero | Fail |

Warm evidence is causal only; reset-16 has no frames aged ≥16, so its null
warm strata are not passes. Lighting frame comparisons are individual points,
not independent-frame confidence claims. Their two-sequence pooled temporal
ratios also exceed 1: 1.07356 [1.05789, 1.19317] causal and
1.05135 [1.04265, 1.11418] reset-16. The overall failure is unambiguous
regardless of the energy-interval interpretation.

Recommend **F1**, the already specified 5×5 LR kernel-prediction path with
softmax weights and a learned mix with the direct estimate, as a replacement
within the single 16×4 v4 model. Cold, early and warm spatial errors and FLIP
justify testing that hypothesis; they do not prove it will fix lighting or
texture/color failures. Do not stack F2/F3, increase the budget, or weaken gates.
After owner direction: implement the one replacement, repeat correctness gates
on both GPUs, train from scratch at the same 60k budget and evaluate the unchanged
development protocols. A second failed Decision A ends this ladder per §9.

The managed training queue has exited successfully (MainPID 0, exit status 0).
No fallback is implemented or running. Stop here under PLAN §3 and request
owner direction. Confirmation, README/gallery, runtime defaults and model code
remain unchanged; this is not a Phase 6 checkpoint freeze.

## Approved F1 replacement (2026-09-29)

The owner authorized the one F1 fallback. The maintained model is now 16×4
with learned 5×5 LR kernels and learned direct/kernel mixing per lobe/HR pixel;
there is no architecture toggle or old direct-only implementation branch.
See [the equations and initialization](design.md). Convolution accounting is
657,792 parameters and 32,864 FLOPs/output pixel (2.153775 GFLOP at 256×256),
excluding kernel weighting/reduction. This is a cost report, not a quality gain.
The old evaluator is preserved byte-for-byte in `runs/v4-f1/archive-base-runtime/`.

Correctness before production training:

- 102 regular tests, Clippy, formatting, debug/release builds and unchanged
  published-gallery verification pass.
- Scalar fixtures check LR stencil/channel order, nearest-edge extension,
  linear HDR/exposure units, per-output/lobe softmax and mixture, extreme logits
  and finite-difference gradients.
- Thirteen GPU gates and the single-encoder Blade example pass on RADV and
  LavaPipe: recurrence/cuts/HDR, CPU/WGSL packing, f64 every-parameter gradients
  under fused/unfused lowering, 128² gradient directions, accumulation and resume.
- Debug CLI smoke passes on the explicitly selected RX 7900 XT and LavaPipe,
  including exact resume/reload, alpha maps/histograms, resets and control checks.

Failures remain in the ledger. The only GPU test correction permits GPU
flush-to-zero of CPU subnormals: the failed exact-packing comparison contained
132 differences ≤8.996×10⁻³⁹ among 9,600 values, with every normal value identical.
No arithmetic, quality tolerance or training recipe changed to fix that assertion.
Training bundle schema 4 rejects the old contract; exact parameter-layout checks
reject direct-only checkpoints. All production training starts from scratch.

The seed-1 launch copies the selected direct-only 16×4 recipe: identical ordered
200-scene corpus, 60k updates, B=8, four-frame unroll, crop 64, losses and learning
rate. Full causal/reset-16 scoring and fixed alpha frames follow completion;
the queue never restarts a failed job or promotes a checkpoint automatically.
The published gallery remains the archived result, not an F1 quality claim.
