# Quality sprint protocol

## Closed — 2026-09-27

Miss: first frozen audit smooth-crop ratio 0.552 exceeded the ≤ 0.50 target.
Its +1.74 dB PSNR and 35% lower temporal MSE did not pass that primary gate.
Cold starts remained noisy (cold crop ratio 0.879); no replacement was published.
The completed width-32 control was rejected: 29.385 dB versus its parent's
30.271 dB on development, with worse temporal error and lighting response.
The later balanced run stopped after 9,277 recorded updates; its recorder still
says `running`, but no training/scoring process remained at closure. Do not resume.
Its last complete evaluation (8,000) scored 30.061 dB; cold/middle smooth ratios
0.941/0.468 versus the original, worse than the parent's 0.852/0.303.
Preserve all artifacts and the stale manifest as evidence of an interrupted run.
Keep published checkpoint `5d0c7411…`; use `89e81df0…` only as the v3 control.
No follow-up learned candidate consumed confirmation data. No more sprint runs.
Next hypothesis: [PLAN.md](../../PLAN.md), one end-to-end recurrent v4 model.

The following is historical evidence, not an active experiment queue. It
implemented the now-removed `TASK.md`. Targets are not achieved results. The
starting model is the published width-16 residual U-Net, checkpoint SHA-256
`5d0c7411c4581a0a8ad99cd87069e4344222dd43020bc28cc0f16a40e47d1321`
(`runs/quality-2026-09-26/train/model.safetensors`).

## Fixed scope and split policy

- 1-spp 128×128 input, 256×256 reconstruction, matching eight-bounce input and
  reference paths, observed high-resolution primary surfaces, split radiance.
- Use the isolated Mesa 26.2.3 driver already verified for catalog visibility.
  Record actual dependency revisions, driver, executable, dataset and weight
  hashes with `scripts/record-run.py`.
- Existing published/audit images are diagnostic and regression data. Do not
  describe them as untouched holdouts in subsequent reports.
- Preserve existing training/development/audit asset-family splits. Fresh audit
  scenes may reuse audit families, but never training or development families.
- Never select checkpoints on the final audit. Select on development visual,
  spatial, and temporal measurements; freeze the weights before the final audit.

## Diagnostic gate

Before broad training, capture a static, untextured procedural canopy scene with
eight independently sampled input frames and 4,096-spp references. Train only
that scene and evaluate both the training inputs and a second independent input
stream of the same scene. This is a fit/noise-generalization diagnostic, not a
scene-generalization result. A separate scene supplies development frames.

Diagnostic seeds: training scene 310001; development scene 310101. Independent
input stream: `--input-sample-offset 64`. Keep camera, objects and lights static;
enable projection jitter. Retain the zero-head guide and starting-checkpoint
results. Record RGB and separate diffuse/specular errors, reference composition
error, and recurrent versus reset-every-frame behavior.

If the model cannot substantially reduce error on this small controlled problem,
resolve the cause before increasing the corpus or training duration. Passing
this gate does not establish held-out quality.

## Fresh final audit specification

Capture two scenes per case, each with 64 consecutive frames and 4,096-spp
references. Enable canopy and projection jitter throughout. All motion values
below are the capture tool's world-unit-per-frame controls.

| Case | Base seed | Additional capture settings |
|---|---|---|
| Static diffuse-dominant surfaces | 410001 | No textures, gloss, or motion |
| Glossy camera motion | 420001 | `--gloss --textures --random-camera-motion 0.01` |
| Object motion/disocclusions | 430001 | `--gloss --object-motion 0.02` |
| Lighting changes | 440001 | `--gloss --light-motion 0.02` |
| Held-out authored objects | 450001 | Audit-only catalog, one object, `--gloss --random-camera-motion 0.01` |

Audit captures, crop coordinates and reference hashes must be locked before
candidate evaluation. Check scene-seed disjointness against all ancestor training
captures, not only the latest fine-tune. Check object visibility over the complete
trajectory. Geometry/reference inspection is permitted to define valid smooth
regions; candidate outputs must not influence crop selection.

The completed audit and 29 reference-selected crops (78 crop/frame pairs) are
locked in [quality-benchmark.json](../quality-benchmark.json), SHA-256
`28b712791d598b1d0ea444a5e11e964166391da523bac49135f3e52be997a58b`.
All ten scene seeds and the two sampled catalog families are disjoint from
training/development ancestry and the current diagnostic scenes. Catalog objects
remain visible throughout, with minimum coverage 3.04%. Crop frames are 0, 31,
and 63; global and temporal scores still cover all 640 frames.

Publication views are also fixed before candidate audit: frame 31 of global
sequences 0 (static), 2 (glossy camera motion) and 8 (catalog). Show original
published model, selected model and reference with unmodified native-resolution
PNGs. Include complete 64-frame comparison videos for the first sequence of
each of the five cases, at 24 fps with identical display transforms and labels.
Video encoding is for presentation only, never a source of numerical metrics.
Review all ten clips, not only the five illustrated ones. These choices do not
alter the frozen crop/data JSON or permit checkpoint reselection on the audit.

## Measurement contract

- Primary crop error: mean squared RGB error after fixed `x/(1+x)` compression,
  before sRGB or quantization, against high-sample references. Target a ratio of
  at most 0.5 versus the starting checkpoint. Report each crop and case as well as
  the aggregate. Include an independent-reference noise estimate.
- Preserve true edges and texture: compare reference-relative gradient error
  and fixed edge/texture crops. Raw gradient magnitude or reduced image variance
  alone cannot pass this gate.
- Report frame-mean PSNR (secondary target +1 dB), SSIM, linear energy bias,
  motion-compensated reference-relative temporal MSE, and rejected-history error.
  Separate cold starts, accumulated history, and reset behavior. Retain all
  frame scores and identify worst regressions.
- Inspect matched videos for flicker, ghosting and disocclusion artifacts. Use
  the same fixed display transform and frame rate for old/new/reference.
- Compare OIDN spatial quality with all input/resampling/guide differences
  declared. It is not a temporally stable reference.
- Measure inference GPU time and memory separately from capture, CPU upload,
  readback, image saving and shader compilation; do not label synchronous
  evaluator wall-clock time as GPU frame time.

For crop scoring, evaluate with `--save-linear`, recording the benchmark JSON,
ordered datasets, executable, and checkpoint as `scripts/record-run.py` inputs.
The evaluator writes row-major little-endian scene-linear RGB f32 files alongside
the display PNGs. `scripts/score-regions.py --benchmark docs/quality-benchmark.json
--before-run BEFORE_RECORDER --after-run AFTER_RECORDER --out crops.json` checks
the locked hashes, dataset order and identical references, then reports each
crop/frame and pixel-weighted summaries. It does not infer a temporal or visual
pass from spatial metrics. `python3 scripts/test-score-regions.py` checks crop
bounds, compression, reference-relative gradients and provenance rejection.

## Targeted long-sequence training

Use the same five case types as the audit, but independent scenes and asset
families. Training has four 64-frame scenes per case; development has two per
case. In case order (static, camera, objects, lights, catalog), base seeds are
510001–550001 and 610001–650001 in increments of 10000. These seeds were checked
against the starting checkpoint's full training/development ancestry, diagnostic
captures and final audit before capture. Training uses 1,024-spp targets
(`--canonical-frames 256`); development keeps 4,096-spp targets. Both keep matched
eight-bounce, 1-spp inputs, projection jitter and observed HR surfaces.

Catalog pools remain `runs/quality-2026-09-26/data/opaque-train-catalog.json`
(22 training families) and `runs/focused-2026-09-25/data/dev-catalog.json`
(four development families), disjoint from each other and all audit families.
Each catalog scene uses one object at target extent 2; inspect full-trajectory
visibility before admitting the capture. Train from the published starting
weights with the one corrected implementation. Do not warm-start from the
overfit diagnostic weights or select updates on the final audit.

All 1,280 training and 640 development frames were captured successfully.
The actual scene seeds and sampled families are disjoint across training,
development and audit. Minimum catalog coverage over every frame is 1.92% in
training and 5.60% in development. The bounded run `diverse-training/` starts
from the published weights, with 4,000 updates, seed 31, learning rate 0.0003,
unroll 2 and lobe loss weight 0.5. Evaluate development at 1,000-update intervals;
these are candidate checkpoints, not a commitment to promote the last update.

## External spatial reference

Use the official [OIDN 2.5.1 Linux SDK](https://github.com/RenderKit/oidn/releases/tag/v2.5.1),
archive SHA-256 `743c3e2aff8c220d5d70fe6cb970fb3d36f2702d2693c61d1d148e404cf37cd6`.
The `oidn-reference` executable invokes its `oidnDenoise` tool with RT, HDR, high
quality, automatic input scaling, CPU device, four threads, affinity disabled,
and clean auxiliary guides. This is an offline quality reference, not a speed
comparison. Record both the SDK archive and executable as run inputs.

Feed unaltered 1-spp **native 128×128** beauty, primary albedo approximated by
`clamp(diffuse_albedo + specular_F0, 0, 1)`, and normalized world normals. It has
no target, high-resolution guides or history as inputs. Denoise before bilinear
upsampling to 256×256 and removing input projection jitter: the
[RT documentation](https://www.openimagedenoise.org/documentation.html#rt) warns
against noise correlation introduced by pre-denoising interpolation. Thus it has
the same traced input budget, but fewer guides and no dedicated super-resolution
or temporal reconstruction. Disclose these differences alongside comparisons.

Run `cargo run --release -p ommatidia-train --bin oidn-reference -- --help` for
options. Repeat the same ordered `--eval-data` inputs as the learned evaluator;
use `--eval-only --save-linear` with the common recorded crop-scoring protocol.
The filename `learned` is only the common scorer's prediction slot; the JSON
report identifies OIDN explicitly. PFM orientation/HDR, jitter resampling and
target-input isolation have unit coverage. `oidn-smoke-run/` completed on eight
independent-noise diagnostic frames; images were inspected for orientation and
alignment. No audit-quality or temporal pass follows from this smoke test.

The shared capture/output helper refactor was checked against the retained
executable on the 64-frame diagnostic: identical per-frame scores and identical
final-frame PNG and full-precision radiance files (`output-refactor-check/`).

The complete 640-frame OIDN comparison is recorded in `audit-oidn-run/` and
`audit-oidn-crops.json`. Frame-mean PSNR is 27.39 dB versus the original model's
29.63 dB. OIDN reduces the predefined smooth-crop MSE by 47.7%, but texture-crop
gradient error rises 3.64×. This illustrates why smoother pictures or a single
aggregate score cannot establish the full quality gate. These are external
reference results, not a new selected Ommatidia checkpoint.

## Compiler conformance repair

The isolated [Naga correction](../../patches/README.md) replaces decorated workgroup
composites with undecorated storage types while retaining host-buffer layouts.
All three debug GPU correctness tests now pass with zero validation errors on
both RADV and LavaPipe (`corrected-compiler-debug/`,
`corrected-compiler-lavapipe/`). CPU/native discrepancy remains 7.47e-7 and
two-frame BPTT loss remains 0.00174994 → 0.00065509. Full-model directional and
parameter gradients pass. Debug capture, training and serialized reload also
pass with zero validation errors (`patched-capture-{train,dev}/`,
`patched-{train,reload}-smoke/`); the reloaded frame scores match exactly.
This resolves the project's reproduced conformance failure, not all possible
Vulkan/compiler bugs. The 64-frame diagnostic scores and final linear output
are unchanged by the compiler fix (`compiler-evaluation-run/`).

The patch's targeted structural and `spirv-val` regressions pass, along with all
139 Naga library unit tests and all-feature/all-target Clippy. A full
`cargo xtask test` was attempted but could not start: `cargo-nextest` is not
installed. No full-wgpu-suite or CTS pass is claimed. Early regression fixture
failures involved a reserved WGSL name, unsupported workgroup pointer arguments,
unsupported checked out-of-bounds atomics, and a disallowed test map type; these
fixtures were corrected to supported inputs. Fresh-checkout patch application
and repeat preparation were independently verified.

## Host memory and performance measurement

The long corpus previously retained every observation and target expanded to
f32. The trainer now retains the original f16 capture records, expanding only
consumed frames, without requantization. The 16-step `loader-{expanded,compact}`
comparison has identical loss/frame CSVs and all 36 checkpoint tensors,
including optimizer state. Safetensors headers differ only in key ordering.
The long diagnostic also matches (`compact-evaluation-run/`). The already
running bounded experiment keeps its recorded executable; no mid-run code swap
or dataset change is performed.

`benchmark` measures hardware GPU pass spans separately for preparation,
ordinary grouped neural execution and resolve. It excludes CPU validation,
upload/readback, compilation, submission and inter-submission queue gaps, and
must not be presented as end-to-end throughput. It reports requested resident
buffer bytes and separately samples process-local driver memory usage, including
warmup; unsupported queries are null. Percentiles use nearest rank.
Native's offline upload/readback buffers are included in memory accounting.
The new timed/untimed eight-frame regression produces exactly identical output
with zero validation errors on RADV and LavaPipe (`timing-parity/`,
`timing-parity-lavapipe/`). `benchmark-smoke-run/` only
checks the measurement plumbing: training was concurrent, so its timings are
**not publication measurements**. Run the selected model with no other GPU work,
two warmup sequences and three measured sequences before publishing performance.

## Development evidence and reset supervision

The first 1,000-update development evaluation scores 29.55472 dB versus 28.95310
with the published weights through the corrected decoder (`dev-starting-run/`).
Temporal MSE falls from 0.00052124 to 0.00043569; reference-gradient MSE also
improves. Cold starts regress, however, by up to 2.69 dB. The worst reset images
have conspicuous broad blotches; this checkpoint is not ready to promote.
At 2,000 updates, mean PSNR reaches 29.65474 dB, temporal MSE 0.00035966 and
gradient MSE 0.00060897. Cold-start mean PSNR is 25.35790 versus 25.66830 for the
starting weights; the worst cold-start regression shrinks to 0.64 dB, but the
inspected chair scene still has broad chromatic blotches. Let the original
bounded 4,000-update run finish; do not select this interim checkpoint from its
aggregate improvement alone. The final learned audit is still unseen.

The matched `dev-reset-2000-run/` evaluates 256 development frames (static and
catalog cases) with history reset before every frame and saves both components.
The four true sequence-start images exactly match causal evaluation. Causal
versus reset-every-frame PSNR is 31.98485 versus 27.79600 dB for static scenes,
and 29.49573 versus 24.51924 dB for catalog scenes. History lowers material-weighted
diffuse MSE by 47.2% / 65.3% and specular MSE by 62.1% / 57.8%, respectively.
The component images show diffuse chromatic blotches, not only noisy reflections.
Separate compressed component errors are diagnostics and cannot be added to
recover composed RGB error. These development diagnostics support retaining
recurrence while improving reset behavior, not replacing it with a spatial model.

Uniform starts in a 64-frame sequence with two-frame unroll expose frame zero
in only 1/63 of updates. Replaying seed 31 yields 65 such updates out of 4,000;
one training scene has none. Test one targeted sampling change with the same
published warm start, corpus, seed, 4,000 updates, learning rate, unroll and loss:
every fourth update (zero-based phase zero) skips warmup at the uniformly sampled
start. The other updates keep the complete causal prefix. This gives varied
simulated cuts throughout every scene, rather than repeatedly fitting only 20
first frames. Sequence/start RNG draws are unchanged. Record actual starts and
warmup lengths in `loss.csv`, and the policy in `training.json`. Test correctness
and reload before starting the bounded comparison; select only on development.
The unit regression covers all 20 scenes and 63 possible starts. The eight-update
debug smoke exercises cuts at nonzero positions; train and reload complete with
zero validation errors, identical frame scores and all 24 PNGs identical
(`reset-policy-{smoke,reload}-run/`). All 80 non-ignored workspace tests and
Clippy pass on Rust 1.92, along with formatting and five crop-scoring tests.

The original `diverse-training/` completed all 4,000 updates. Update 3,000 has
the highest mean development PSNR (30.08857 dB); update 4,000 scores 29.98015 dB
with temporal MSE 0.00035807, gradient MSE 0.00060280 and reset PSNR 25.55948.
At 4,000, 630/640 development frames improve over the starting weights through
the corrected decoder; the worst regression is 0.46 dB on a cold start.
Neither the highest mean PSNR nor the final update is automatically selected.

The matched reset-balanced run is `reset-training/`, using the clean
`ff724e3` source and `reset-policy-runtime/transport`. Its first 1,000-update
evaluation scores 29.51474 dB versus 29.55472 for uniform warmup at the same
budget. Reset PSNR improves from 24.10975 to 25.59155 dB, and temporal MSE falls
from 0.00043569 to 0.00037933. Mean energy ratio is 0.98010, so brightness and
long-history behavior still need review at later checkpoints. This is evidence
for continuing the bounded comparison, not a selected result.

An additional development-only PNG crop diagnostic uses inverse sRGB on
quantized outputs. Reference-inspected **middle-frame** crops cover five cases;
they are tuning data, not the frozen audit or its full-precision metric.
`dev-display-middle-{3000,4000}.json` reports smooth-error ratios 0.711 / 0.629
versus the starting weights through the corrected decoder. Edge error and
reference-relative gradient error also improve. The initial fixed-rectangle
three-frame attempt is not used: camera motion changed some regions' content.
This correction follows reference inspection, and none of these development
crops replaces or modifies the predeclared final audit. A separate
`dev-original-run/` evaluates the original published runtime, to distinguish
implementation gains from retraining gains.
That evaluation completed at 28.27361 dB, versus 28.95310 with unchanged weights
through the corrected runtime. The uniform 4,000-update run therefore gains
1.70654 dB over the actually published implementation on development, not just
1.02706 dB over the corrected decoder. Its development middle-frame smooth-crop
PNG-error ratio versus the original implementation is 0.337; edge/texture ratios
are 0.692 / 0.425 (`dev-display-original-4000.json`). These remain tuning-set
diagnostics, not final-audit claims or a reason to skip reset/motion review.

### Illumination response: a remaining structural constraint

Reset-balanced training completed all 4,000 updates. At 3,000, development PSNR
is 30.09515 dB, reset PSNR 26.02241, temporal MSE 0.00036847 and energy ratio
1.00262. All ten cold-start frames improve over the original published runtime.
At 4,000, mean PSNR is 29.92771, reset PSNR 26.09906 and temporal MSE 0.00035775,
but energy ratio rises to 1.01212. The last checkpoint is not automatically best.

Worst-frame review exposes excessive brightness as the second lighting scene
darkens. At update 3,000, its final frame scores 25.95757 dB, versus 26.72249 for
the original model. Resetting history on that same frame scores 32.27004 dB
(`dev-lights-reset-3000-run/`). The images confirm persistent bright illumination,
not just grain. Its mean compressed RGB is 0.21439 versus reference 0.17814;
the original model is also too bright, at 0.20594.

This failure is not captured by a better temporal average alone: the
development-only 8×8-block change diagnostic improves by about 50% in that scene
(`dev-light-flicker-3000.json`). Both static development scenes also improve,
by 44–46% (`dev-static-flicker-4000.json`). These inverse-sRGB PNG diagnostics
use fixed primary geometry, not motion compensation or full-precision audit
data. Slow response bias and random flicker must be inspected separately.

The first corrected decoder clamps the corrected incoming estimate before
history blending. That imposes `output >= h * history`, so the network cannot
subtract stale illumination quickly. Test keeping the innovation signed until
after accumulation, while retaining its `(1-h)` weighting. This is the same
six-channel head and 188,160 parameters, not another model family. The
256-frame regression still prevents amplification of stationary corrections,
now covering negative, zero-clipped and positive values. A separate regression
shows that stale history can be removed while final radiance stays nonnegative.

All four debug GPU checks pass on RADV and LavaPipe with zero validation errors
(`signed-numerics/`, `signed-numerics-lavapipe/`), including gradients,
recurrence, timing parity and reload. All 80 regular workspace tests, Rust 1.92
Clippy and formatting pass. With unchanged update-3,000 weights, the 256-frame
static/lighting diagnostic changes mean PSNR by −0.042 / −0.023 dB; the failing
final lighting frame only rises to 26.00429 dB (`dev-signed-3000-run/`).
Do not claim that the code change alone fixes response quality.

The bounded follow-up starts from `reset-fit/step-3000.safetensors`, the
reset-balanced checkpoint with the best mean development PSNR and lower energy
bias than update 4,000. Use the same corpus, reset sampling and loss, fresh Adam,
seed 31, learning rate 0.0001, two-frame unroll and 1,000 updates, evaluating at
500 and 1,000. Check illumination response and static noise together before
selecting any final candidate. The learned final audit remains unseen.

### Development selection

The signed follow-up completed 1,000 updates. With the original 32-frame diffuse
limit it reaches 29.99799 dB, reset PSNR 26.10267 and temporal MSE 0.00036002.
Reducing the diffuse limit to 16 frames, with the specular limit unchanged at
eight, improves the response/noise trade-off on development. The full 640-frame
comparison uses unchanged weights and the signed decoder for both candidates:

| Development candidate | PSNR | Reset PSNR | Temporal MSE | Smooth-crop error ratio |
|---|---:|---:|---:|---:|
| Reset-balanced update 3,000, diffuse limit 16 | 30.20729 | 26.01235 | 0.00037738 | 0.348 |
| Plus 1,000 signed-decoder updates, diffuse limit 16 | 30.16899 | 26.10250 | 0.00036634 | 0.330 |

Both improve all 640 frames over the original published runtime (28.27361 dB).
The crop ratios now use full-precision radiance with the same reference-inspected
development rectangles, not PNGs (`dev-full-crops-{signed-,}history16.json`).
The refined candidate has edge/texture MSE ratios 0.705 / 0.491 and corresponding
reference-gradient ratios 0.767 / 0.529 versus the original implementation.
Its lowest frame gain is +0.343 dB. Mean energy ratio is 1.00170. The previously
failing final lighting frame reaches 29.63875 dB versus 26.72249 originally.
Brightness lag is reduced, not eliminated; do not call that frame reference-clean.

Select the refined weights with diffuse/specular limits 16/8: the 0.038 dB mean
PSNR trade-off buys lower smooth-surface error, better cold starts and lower
temporal/gradient/rejected-history errors. Texture MSE is higher than the other
candidate but remains less than half the original model's on the development
texture crop. This is a multi-metric development selection, not highest-PSNR
selection. Keep the single architecture and make 16/8 its default configuration.

Fixed-geometry development block-change diagnostics also improve on both static
scenes (ratios 0.607 / 0.576) and both lighting scenes (0.596 / 0.475), with all
64 frames included (`dev-flicker-signed-history16.json`). These are quantized
tuning diagnostics, not a substitute for the frozen temporal audit. The new
40-frame continuous CPU/native test verifies output and exact history-age
saturation at both diffuse limits, on RADV and LavaPipe with zero validation
errors (`history-cap-numerics{,-lavapipe}/`). Freeze checkpoint, config and source
provenance before evaluating the final audit; no audit result informed this choice.
After adopting the 16-frame default, all five debug GPU tests pass on both
backends with zero validation errors (`selected-numerics-{radv,lavapipe}/`).
All 80 regular tests, Clippy and formatting pass on Rust 1.92; the eight Python
crop/catalog tests also pass.

### First frozen audit: improvement, but not a quality-gate pass

Selection was frozen and pushed at `d7ffd8d`, before the learned audit began
at 2026-09-26 22:26:39 UTC. The immutable inference build is `c6eefca` and the
checkpoint hash is `0d75f36762aedf4d0e96ecc03d7e6d8d273cfcfc5ec9ea4905d7dc9aaff8d653`.
`audit-selected-run/` evaluates all 640 frames with each implementation's own
history. PSNR rises from 29.62718 to 31.36767 dB; SSIM from 0.81985 to 0.87846.
Temporal MSE falls 35.3%, rejected-history MSE 48.8%, and gradient MSE 34.3%.
638/640 frames improve; the worst regression is 0.142 dB on the first catalog
frame. Mean energy ratio improves from 0.99180 to 0.99898, but scene-linear MSE
rises 1.8%. Do not claim every diagnostic improves.

The strict crop report (`audit-selected-crops.json`) has smooth/edge/texture MSE
ratios **0.552 / 0.774 / 0.797**; reference-gradient ratios are
**0.731 / 0.829 / 0.861**. Smooth-crop error decreases 44.8%, missing the
predeclared ratio-at-most-0.5 gate. Lighting-case smooth crops regress in
aggregate (ratio 1.106), despite better whole-frame scores. This result is not
promoted as completion of the quality sprint. Do not round 44.8% into a pass,
change crop weights, omit cold starts, or select another checkpoint on this audit.
The fixed smooth-crop ratios at frames 0 / 31 / 63 are 0.879 / 0.332 / 0.315.
Cold starts are a major part of the missed gate, and remain in its denominator.
The inspected cold-frame progressions show conspicuous colored outliers and
surface blotches, even where the late sequence is substantially cleaner.

`score-sequences` reuses the existing temporal metric on saved full-precision
outputs and observed capture motion. It reproduces both evaluator averages to
floating-point precision and exposes all ten sequence scores. Every sequence's
temporal error decreases (ratios 0.585–0.820). An additional 8×8-block diagnostic
also decreases in every sequence (0.557–0.753); this is a supplementary diagnostic,
not a new predeclared pass criterion. `audit-videos/` contains all ten matched
64-frame clips and chronological review sheets. Visible blotches, specular
sparkle and illumination bias remain. No perceptual-video-study claim is made.

The uncontended `selected-performance-run/` measures 192 frames after 128 warmup
frames on RX 7900 XT: GPU pass-span mean 1.178 ms, p50 1.183 ms, p95 1.207 ms.
This excludes host upload/readback, compilation and submission gaps; it is not
end-to-end throughput. It is timing evidence for this candidate, not a quality pass.

### Bounded follow-up: train with the retained history configuration

The selected weights were trained with a 32-frame diffuse limit and evaluated
with 16. Test that remaining training/inference mismatch explicitly: start from
the frozen weights and their 16/8 configuration, retain the same training and
development corpus, reset-balanced sampling, losses and two-frame unroll, and
run 1,000 updates at learning rate 0.0001, seed 31, evaluating at 500 and 1,000.
This changes no model architecture. Judge the run on development spatial,
temporal, brightness and cold-start evidence; it is not an audit-driven checkpoint
search. The first audit is now a regression/diagnostic set, not an untouched
holdout for any subsequent model. A newly frozen confirmation audit is required
before promoting a later candidate. The original crop/data lock and failed
selection record remain unchanged.
Supplementary development cold-start crops reuse the reference-checked smooth
wall/floor regions at frame zero in the static, object-motion, lighting and
catalog cases. The textured camera ceiling is not a smooth region in this
diagnostic. These are explicitly tuning data, not changes to the audit: the
frozen candidate's full-precision cold smooth-error ratio is 0.844 versus the
original runtime (`dev-cold-signed-history16.json`). Middle-frame development
results remain separately reported; they cannot stand in for cold-start quality.

The matched-history run completed. At 500 updates, development PSNR is 29.99455,
reset PSNR 26.03237, and temporal MSE 0.00035810. At 1,000, these are 30.16505,
26.12158 and 0.00036162, with energy ratio 1.00062. Development PNG diagnostics
give middle/cold smooth-error ratios 0.336 / 0.841 at 1,000. This is not a
material cold-start improvement over the frozen candidate. Matching the limit
alone does not resolve the remaining quality gap; do not send this checkpoint
to the confirmation audit as if the bottleneck had been fixed.

### More varied reset supervision

Test one bounded cold-coverage package within the same architecture: increase
the long corpus from 20 to 40 scenes and reset every other training window.
The original every-fourth policy supplies a true reset in only one eighth of
the two-frame loss slots. The new policy doubles that fraction while retaining
full-prefix warmup in the other half of windows. Sequence/start draws remain
uniform and are recorded. This tests the package, not isolated attribution to
either scene diversity or reset frequency.

Capture four additional 64-frame training scenes per case with base seeds
910001–950001 in increments of 10000, the same case settings, 1-spp 128→256
observations, eight bounces, 1,024-spp targets and the existing training-only
catalog pool. All 20 new scene seeds were checked against every current capture,
confirmation scenes and published ancestry before capture. Keep development
unchanged. Verify complete trajectories, membership and hashes before training.
No confirmation prediction may influence training or selection.

After a debug train/reload smoke, warm-start from `matched-history-fit/model.safetensors`
(weights only, fresh Adam), retaining diffuse/specular limits 16/8, losses and
two-frame unroll. Bound training to 4,000 updates, learning rate 0.0001, seed 31,
evaluating development every 1,000 updates. Select on cold and mature spatial
quality, motion, texture and brightness together. A higher mean PSNR alone is
not enough. The original failed audit remains fixed, and confirmation data/crops
are locked independently before any further candidate audit.

The additional captures passed membership, header and full-trajectory checks.
Training now contains 2,560 frames across 40 scenes; development remains the same
640 frames. The four new catalog scenes sample four training families, with
minimum visible coverage 5.46%. The two long training batches together contain
seven distinct sampled catalog families; the 22-family pool is not a claim that
all 22 appear in these captures. Dataset hashes and sidecars are recorded in
`cold-coverage-training/`, which uses the immutable `7e89075` executable and saves
full-precision development outputs at each 1,000-update checkpoint.

The eight-update reset-policy smoke and serialized reload produce identical
frame scores and all 24 PNGs, with zero Vulkan-validation errors
(`cold-coverage-{smoke,reload}-run/`). All 81 regular workspace tests pass on
Rust 1.92 (`cold-coverage-unit-tests/`). These correctness checks do not establish
that the new sampling policy improves quality; that remains a development test.

The bounded run completed all 4,000 updates. Full-precision development crop
reports are `dev-coldcoverage{1000,2000,3000,4000}-{middle,cold}.json`, using the
unchanged development rectangles and original published runtime as the control:

| Update | PSNR | Cold-start PSNR | Middle smooth MSE ratio | Cold smooth MSE ratio |
|---|---:|---:|---:|---:|
| 1,000 | 30.03709 | 26.10979 | 0.348 | 0.892 |
| 2,000 | 30.26950 | 26.22736 | 0.316 | 0.833 |
| 3,000 | 30.22411 | 26.26231 | 0.303 | 0.858 |
| 4,000 | 30.27080 | 26.26738 | 0.303 | 0.852 |

At 4,000 all 640 frames improve over the original implementation, by at least
0.438 dB. Temporal MSE is 0.00035853, gradient MSE 0.00059287 and mean energy
ratio 1.00423. Middle edge/texture MSE ratios are 0.699 / 0.297. These are useful
development gains, but the cold-crop result is not a material improvement over
the parent's approximately 0.84 ratio. Do not extend this run or evaluate a new
confirmation candidate on this evidence. The final weights are
`89e81df0b0a2228b04853c46c9aa0fb589bc364f8c6ba03935c9c8047ce51bbe`.

All ten development sequences at update 1,000 have lower temporal and
8×8-block-change errors than the original model (`dev-coldcoverage1000-sequences/`).
Matched review sheets show substantially cleaner accumulated surfaces but
remaining cold outliers. This is development evidence, not a new held-out pass.

If this four-checkpoint comparison does not materially improve cold starts, do
not extend broad training or consume the confirmation audit. First test cold
fitting capacity on a single frame of the existing diagnostic scene 310001,
captured with the same 1-spp 128→256 observations and 4,096-spp reference. A
one-frame sequence with unroll one guarantees empty history on every update
without introducing another trainer or architecture. Start from the matched
16/8-history parent, use 1,000 updates, learning rate 0.0003 and seed 31, and keep
the same loss. The separate diagnostic development scene remains disjoint.

Evaluate both the fitted input and the existing independent-noise stream of
scene 310001 with history reset on every frame, using the exact parent weights
as the matched control. Fitting one image tests optimization/representation,
not scene or noise generalization. A training-image gain without an
independent-noise gain is not success. Never use these overfit diagnostic weights
as a publication candidate or a warm start for broad training.
Use a lossless extraction of record zero from the existing eight-frame
`fit-train.omd` capture: change only the header's record count and sequence
length, preserving all observation, jitter and target bytes. Record the source
capture/sidecar hashes and record digest in the derived sidecar. This also avoids
changing the sampling footprint: the capture CLI disallows projection jitter
for a newly rendered one-frame sequence.
The first probe load was rejected by the shared loader's old minimum length of
two. Allow valid nonempty one-frame captures while retaining provenance and
input-source checks; recurrent training still rejects unrolls longer than the
sequence, and frozen benchmark checks still require their locked lengths.
Single-frame scores have zero temporal pairs and establish no temporal result.
The loader correction passes all 82 regular workspace tests, Clippy and formatting
on Rust 1.92. Eight debug one-frame training updates all have start/warmup zero;
training and reload have zero validation errors and 49 image/score files match
byte-for-byte (`cold-capacity-{smoke,reload}-run/`). On the existing eight-frame
control, the loader-only change preserves all 97 image/score files exactly.

The 1,000-update cold fitting probe completed (`cold-capacity-training/`). Every
update had empty history. Its loss fell from 0.00249290 to 0.00049699; training
frame PSNR rose from 28.10802 to 35.36028 dB. On eight independent-noise inputs
of the same scene, however, PSNR only rose from 27.98058 to 28.61715, SSIM fell
from 0.79671 to 0.79490, and gradient error increased 4.8%. The fitted image is
much cleaner, while independent-noise images retain visible streaks and colored
outliers. The model can fit cold output, but this is mostly memorization, not
robust denoising. These weights are rejected as a quality candidate.

### Controlled cold-noise generalization

Before another broad run, test varied cold observations of the same diagnostic
scene. Use the existing 64-frame `fit-train-long.omd` (scene 310001, input offset
256), with `--unroll 1 --reset-every 1`, and evaluate `fit-long.omd` (same scene,
independent offset 128) with reset before every frame. Keep the observed jitter,
4,096-spp references, model configuration and loss weights. A one-frame unroll
has no temporal-loss pairs; no recurrent-quality improvement is inferred.

Start again from the matched-history parent, never from the one-image overfit.
Bound the diagnostic to 2,000 updates, learning rate 0.0003 and seed 31. Evaluate
the fitted 64 inputs and the 64 independent-noise inputs against that exact
parent, with identical reset behavior. Retain the separate scene-310101 causal
development result as an overfitting warning, not a checkpoint-selection target.
The explicit positive reset interval defaults to the unchanged every-other
policy and is recorded in training provenance. Verify all-cold sampling and
serialized reload in debug before the run. No frozen audit is used here.

The diagnostic completed (`cold-noise-training/`). All 2,000 updates were cold,
covering all 64 inputs. Training-stream PSNR rises 28.08125 → 32.83757 dB;
independent-noise PSNR rises 28.12444 → 32.62485 dB, improving every test frame
by at least 3.684 dB. Test SSIM rises 0.80001 → 0.86077 and gradient MSE falls
0.00072651 → 0.00063685; energy ratio becomes 0.99996. Reference files match
byte-for-byte between parent and fitted evaluations. Broad surface blotches
decrease in the inspected image, but colored specular outliers remain.

This is **noise generalization within one scene**, not scene generalization:
the separate scene's causal PSNR falls 30.82542 → 23.39932 dB. The one-image
probe had fallen to 26.07836 there. Neither set of diagnostic weights is retained
for broad training. The explicit sampling control passes all 83 regular tests,
Clippy and formatting; its varied-start debug smoke and reload have zero
validation errors and 49 identical image/score files.

### Diverse all-cold supervision stage

Test whether the successful within-scene cold supervision transfers when trained
on the 40-scene corpus. Start from `cold-coverage-fit/model.safetensors`
(`89e81df0b0a2228b04853c46c9aa0fb589bc364f8c6ba03935c9c8047ce51bbe`),
not either overfit probe. The final broad checkpoint retains the accumulated-frame
improvements and supplies a known, unchanged cold-start baseline for this test.

Keep the same ten ordered training captures, five development captures, 16/8
history configuration, model, loss weights and seed 31. Use fresh Adam, learning
rate 0.0001, `--unroll 1 --reset-every 1`, and exactly 4,000 updates with
full-precision development evaluation every 1,000. This is a cold-supervision
stage: both history exposure and unroll/temporal-loss exposure change, so it does
not isolate reset frequency alone. Causal inference is unchanged. Select using
cold and mature spatial quality, texture, temporal error and illumination
response together; reject a cold-only gain that damages accumulated behavior.
Do not extend the budget or consume the confirmation audit unless development
evidence justifies freezing a new candidate. The single architecture and frozen
audit/confirmation contracts remain unchanged.

The all-cold stage completed with all 4,000 windows reset, covering all 40
sequences and all 64 start positions. It is **rejected as a quality candidate**.
The final checkpoint is
`ffd989e44a15047ec5b13c666e01f447186f9e22d03003776096796f2f8f06b2`.

| Update | Causal dev PSNR | Cold-start PSNR | Middle smooth MSE ratio | Cold smooth MSE ratio |
|---|---:|---:|---:|---:|
| Parent | 30.27080 | 26.26738 | 0.30318 | 0.85163 |
| 1,000 | 29.82490 | 26.18200 | 0.33909 | 0.97255 |
| 2,000 | 29.87633 | 26.38318 | 0.31041 | 0.78841 |
| 3,000 | 29.65549 | 26.35033 | 0.35042 | 0.84288 |
| 4,000 | 29.67396 | 26.39434 | 0.33251 | 0.83546 |

Crop ratios use the original published implementation as the denominator, not
the parent. The modest cold gain does not compensate for accumulated regressions:
the final model improves only 21/640 frames over its parent, and 610/640 over
the original baseline. Its worst regression against the original is 0.476 dB.
Energy ratio rises from 1.00423 to 1.02403; the second lighting sequence's final
frame falls from 29.42091 to 27.52758 dB. Temporal MSE is 0.000360410 versus
the parent's 0.000358526. Native chronological sheets show remaining colored
outliers and broader brightness errors. This is sheet inspection, not a claim
of video playback. Evidence: `all-cold-training/`, `dev-allcold*-{middle,cold}.json`
and `dev-allcold4000-videos/`. No confirmation predictions were generated.

Before changing capacity or training duration, compare cold-start fitting on
the training scenes with development generalization. A lossless diagnostic
keeps frame zero of every existing training/development sequence (40 training,
10 development frames), each reset independently. The frame choice is fixed by
sequence boundaries, not model errors. Preserve jitter, observations, surfaces
and targets byte-for-byte; record source hashes and record indices. Compare
the broad parent and the completed all-cold checkpoint with the same runtime.
Report case-level RGB and lobe errors, not just the pooled mean. Training targets
have 1,024 samples and development targets 4,096, so this is a diagnostic gap,
not an exactly matched reference-noise experiment. These derived records never
enter training or the locked audits.

The probe completed: training PSNR moves 26.95408 → 27.18130 dB, improving all
40 selected training observations, while development moves 26.26738 → 26.39434,
improving nine of ten. Training diffuse-illumination MSE falls only 4.6% and
specular MSE 4.7%. Thus the broad model still has a fitting limitation, not just
a large held-out generalization gap. This does not yet distinguish optimization
from capacity. All 50 references and fixed guides match between evaluations;
all ten development predictions/references match the full causal evaluations'
frame-zero files byte-for-byte. Evidence: `cold-corpus-{parent,allcold}-run/` and
`cold-corpus-report.json`, verified by `cold-corpus-score-run-v2/`.

### Bounded optimization control

Before changing capacity, repeat the diverse all-cold stage at learning rate
0.0003 instead of 0.0001. This rate substantially improved the controlled
same-scene independent-noise test; the broad cold fitting gain at 0.0001 was
small. This is an optimization hypothesis, not evidence that a larger step will
solve denoising or preserve recurrence.

Start independently from the same broad parent (`89e81df0…51bbe`), with fresh
Adam, seed 31, the same ordered 40 training / 10 development scenes, unchanged
16/8 history, loss weights, `--unroll 1 --reset-every 1`, cosine schedule and
4,000-update budget. Only the initial learning rate changes. Evaluate full
causal development at the same 1,000-update intervals and repeat the fixed
cold fitting probe afterward. Compare with both the lower-rate control and the
parent; reject cold-only gains accompanied by accumulated bias or lighting
regressions. Do not extend this run or evaluate confirmation without a justified,
development-frozen candidate. Retain the single architecture.

The rate control completed and is **rejected**, checkpoint
`73e1c99ae2b5c70bfab158808076090b1a7c6e4e116e5e342496c185795659dc`.
All 4,000 sampled sequence/start/reset triples exactly match the lower-rate run.
The last 1,000 updates' mean training loss falls 4.3% relative to that control,
but the extra fitting does not preserve causal reconstruction.

| Update | Causal dev PSNR | Cold-start PSNR | Middle smooth MSE ratio | Cold smooth MSE ratio |
|---|---:|---:|---:|---:|
| 1,000 | 29.52780 | 26.00347 | 0.41513 | 1.02193 |
| 2,000 | 29.70778 | 26.46294 | 0.40558 | 0.80421 |
| 3,000 | 29.16658 | 26.45310 | 0.41580 | 0.87640 |
| 4,000 | 29.23355 | 26.50591 | 0.38619 | 0.81874 |

Final causal PSNR is 1.03725 dB below the broad parent; energy ratio is 1.03002.
Only 22/640 frames improve over the parent, and 559/640 over the original
baseline. The worst original-baseline regression is 1.50370 dB. Lighting
sequence 7 ends at 26.90378 dB versus the parent's 29.42091. Temporal MSE is
0.000370022, also worse than the parent's 0.000358526. The inspected worst
frame retains broad brightness and reflection errors.

The fixed fitting probe gives 26.95408 → 27.37951 dB on the 40 training
observations and 26.26738 → 26.50591 on the ten development observations.
Every training observation improves, but two development cold starts regress.
References/fixed guides match, and extracted development predictions again
match causal frame-zero outputs exactly. Evidence: `cold-rate-training/`,
`dev-coldrate*-{middle,cold}.json`, and `cold-rate-probe-report.json`.
The higher rate therefore does not solve the fitting/recurrence trade-off;
it does not establish that all optimization improvements are exhausted.
No confirmation predictions were generated.

### Conditional same-family capacity control

If the completed rate control still gives little broad cold-fitting improvement
and loses causal quality, test width 32 in the **same existing U-Net graph**.
This is a capacity comparison, not proof that capacity is the bottleneck and
not a new architecture family. The retained implementation/default stays width
16 until evidence justifies changing the single published configuration.

Embed the broad parent's width-16 channels into width 32, preserving all old
connections and giving old channels zero connections from new channels. Handle
the decoder's two skip concatenations explicitly. New channels use seeded
Kaiming initialization (seed 314159); the radiance head initially ignores them.
Discard optimizer state. Parameter count becomes 685,056. The ignored offline
conversion's independent 4×4 scalar-core check has exactly zero output error
(`widen-scalar-check-run/`); this alone is not GPU parity or a quality result.

Before training, compare GPU outputs on the fixed 50-frame cold fitting probe
and an eight-frame causal diagnostic. Require identical references and maximum
absolute RGB difference divided by `1 + abs(parent RGB)` at most 1e-5. Verify
debug training/reload with zero Vulkan-validation errors. Do not train the
expanded model if those initialization/correctness checks fail.
Also run the existing independent loss/directional-gradient and every-parameter
GPU/reference check at width 32, using `OMMATIDIA_TEST_CHANNELS=32` for the
ignored `two_frame_training_matches_reference` test. Its default still tests
the retained configuration; this diagnostic override changes no model code.

The initial GPU parity check **failed**: normalized maximum error was
0.00368326 on the 50 cold frames and 0.000942414 on the eight causal frames,
both above the unchanged 1e-5 tolerance. References were byte-identical.
Evidence: `wide-cold-parity.json` and `wide-causal-parity.json`. No broad
width-32 training run is authorized by these results. The independent width-32
directional-gradient, loss, and every-parameter GPU/reference tests pass under
default and unfused lowering (`wide-gradient-check/`, zero validation errors).
An eight-update debug learning/reload smoke test also passes, with all 49
output/metric files identical after reload; its weights are not a broad-training
initialization.

Execution precision explains most of the discrepancy: Meganeura's default
`CoopPolicy::Auto` can choose f16-input inference kernels, while derivative
dispatches retain full precision. Explicit `NativeF32` passes the eight-frame
causal probe but leaves one of 9,830,400 cold RGB values just outside tolerance.
Separately disabling dispatch fusion changes the remaining rounding difference.
The complete probes, rerun for both widths with identical runtime settings, give:

| Inference policy | Cold maximum normalized error | Causal maximum normalized error | Gate |
|---|---:|---:|---|
| Automatic precision, fused | 0.00368326 | 0.000942414 | Fail |
| Native f32, fused | 0.0000108303 | 0.00000196753 | Fail |
| Native f32, unfused | 0.00000858246 | 0.00000243788 | Pass |

All reference images match exactly; the tolerance remains 1e-5. Preserve all
failed runs. Evidence: `f32-{cold,causal}-parity.json` and
`unfused-{cold,causal}-parity.json`, their evaluation/build manifests, and the
original runtime snapshots. The four-frame `unfused-subset-parity.json` was a
diagnostic on the failing case, not a substitute for the complete 50-frame gate.
Retain explicit f32, unfused inference for the controlled training comparison;
training's derivative/lowering policy is unchanged. This is not evidence of
an upstream contract violation: automatic reduced precision was permitted.

The width-16 parent itself changes by only +0.00000127 dB on the 50 cold frames
and -0.00046542 dB on the eight causal frames versus its original automatic
runtime. The precision choice is not a meaningful quality improvement. Before
comparing the wider trial against the lower-rate control, also re-evaluate
that control and the parent on all 640 development frames with the same new
runtime. Measure final GPU time/memory again; previous automatic-policy timing
does not apply.

The tiny full-model f64 inference check passes at both widths, but did not
exercise reduced-input kernels. A production-extent identity convolution on
the explicitly selected RX 7900 XT reproduces f16 input rounding under `Auto`
(`auto-precision-regression-rx7900/`, expected failure: 0.000100017 absolute
error). Native tests previously passed `None` to device selection, ignoring
`MEGANEURA_DEVICE_ID`; they now parse it at the test boundary and print the
actual adapter. Earlier native-test runs therefore do not establish coverage
on a requested hardware adapter. The independent gradient checker already
honored the environment, and CLI capture/training/evaluation explicitly selected
the device. Re-run the complete debug GPU suite on the selected RX 7900 XT and
LavaPipe, plus width-32 debug learning/reload, before broad training.

Those checks now pass: all seven ignored GPU tests on the explicitly selected
RX 7900 XT and on LavaPipe, with zero Vulkan-validation errors
(`f32-unfused-numerics-{rx7900,lavapipe}/`). The identity convolution has zero
error under the retained policy; maximum tiny-model f64 inference discrepancy
is below 5.8e-8. Width-32 debug smoke/reload also passes, with all 49 PNG/f32/CSV
files byte-identical (`wide-f32-{smoke,reload}-run/`). The independent width-32
gradient test remains the earlier `wide-gradient-check/`; training code did not
change. Rust 1.92 regular tests (83), Clippy, formatting, eight crop tests, three
catalog tests and the unchanged published-result verifier pass.

The clean release runtime is source `d1da76d`, executable SHA-256
`f24fe1fd2cce91bdeace1054b9071f3c557a0da14c892eb1b12cfe8fb908aab8`
(`precision-policy-release-build/`, `precision-policy-runtime/transport`).
Release repeats both complete parity probes successfully, and all 1,396
PNG/f32/frame-score files match their debug counterparts byte-for-byte.
The default-adapter observation explicitly identifies the Ryzen integrated GPU
(`default-adapter-observation/`), explaining why the earlier unselected identity
tests did not exercise the RX 7900 XT's reduced-precision path.

Both full 640-frame development controls were repeated with this release:

| Matched f32 control | PSNR | Cold PSNR | Middle smooth ratio | Cold smooth ratio | Temporal MSE |
|---|---:|---:|---:|---:|---:|
| Broad parent, width 16 | 30.27071 | 26.26739 | 0.30321 | 0.85154 | 0.000358527 |
| All-cold, width 16, update 4,000 | 29.67390 | 26.39442 | 0.33253 | 0.83550 | 0.000360407 |

These crop ratios still compare against the fixed original implementation.
Precision-only mean PSNR changes are -0.00009392 / -0.00006031 dB; all 640
references in each comparison match exactly. No per-frame PSNR change exceeds
0.00116 dB in magnitude. Evidence: `dev-{parent,allcold}-f32-run/` and the
corresponding `{middle,cold}.json` crop reports. The earlier quality verdicts
stand; the policy change does not remove the cold-start fitting limitation.

For a bounded capacity test, use the same broad parent function, fresh Adam,
seed 31, ordered 40 training / 10 development scenes, unroll 1, reset interval
1, loss weights, 16/8 history and 4,000 updates as the lower-rate all-cold
control. Keep learning rate 0.0001 and its cosine schedule fixed; do not combine
widening with the higher rate. Evaluate causal development every 1,000 updates
and repeat the fixed fitting probe. Preserve all cold/mature/texture/lighting
checks; no confirmation evaluation or publication follows from a fitting gain
alone. Experimental assets remain outside tracked source and result galleries.

The bounded run completed (`wide-cold-training/`, output `wide-cold-fit/`),
using that clean release runtime and the checked 685,056-parameter conversion,
not either smoke-test checkpoint. Its metadata confirms fresh Adam, seed 31,
2,560 training / 640 development frames, unroll/reset interval 1, and the fixed
4,000-update budget. All 4,000 sampled sequence/start/reset triples match the
width-16 control exactly. The last 1,000 updates' mean training loss is 4.5%
lower, but the completed capacity control is **rejected**:

| Update | Causal dev PSNR | Cold-start PSNR | Middle smooth MSE ratio | Cold smooth MSE ratio |
|---|---:|---:|---:|---:|
| 1,000 | 29.76918 | 26.20288 | 0.38359 | 0.95963 |
| 2,000 | 29.72870 | 26.43858 | 0.35540 | 0.77569 |
| 3,000 | 29.30267 | 26.41260 | 0.36572 | 0.83518 |
| 4,000 | 29.38472 | 26.47105 | 0.32724 | 0.82024 |

Final causal PSNR is 0.886 dB below the matched f32 parent; only 20/640 frames
improve over it. Against the original published implementation, 560/640 improve
and the worst regression is 1.804 dB (sequence 1, frame 62). The last lighting
frame scores 26.60589 dB versus the parent's 29.42. Temporal MSE is 0.000364736
and energy ratio 1.02036, both worse than the parent. Native-resolution cold
images still show colored outliers and blotches.

The fixed cold fitting probe gives 26.95409 → 27.38040 dB on 40 training
observations and 26.26739 → 26.47105 on ten development observations. All training
observations improve, but two development cold starts regress. References and
fixed guides are byte-identical, and the extracted development predictions
match causal frame zero exactly. Evidence: `wide-cold-probe-report.json` and
`dev-widecold*-{middle,cold}.json`. Final checkpoint SHA-256 is
`3db20d99401b5d251c0b5c3e4fb828b4c2ae56e2f72da1909eb56df1b74e3850`.
This small fitting gain does not justify the extra capacity or overcome the
all-cold recurrence regression. Retain width 16; no confirmation predictions or
new result gallery were generated.

An additional correctness diagnostic checks the scalar training objective at
the actual 128×128 input extent. Compare GPU backward gradients with forward-only
GPU finite differences in two directions per parameter tensor, using nonzero
head weights and fixed inputs. Run widths 16 and 32 on the explicitly selected
RX 7900 XT. Fixed central steps are 0.02, 0.01 and 0.005 with Richardson
extrapolation; require step-size convergence within 1% of the directional
derivative plus 1e-7 and agreement within 2% plus 1e-7. Nonconverged or zero-signal
probes are failures, not passes. This probes production-size backward kernels,
not every gradient coordinate, and supplements the independent small f64 oracle.
It does not change the active experiment or establish a training bottleneck by
itself. If it exposes a real gradient failure, stop quality interpretation and
diagnose it before another training run.

All 24 directional probes pass at each width on the selected RX 7900 XT, and
all 24 width-16 probes pass on LavaPipe, with zero Vulkan-validation errors.
Largest relative gradient discrepancies are 0.105%, 0.104% and 0.144%,
respectively (`production-gradient-directions{16,32,-lavapipe}/`). The software
check takes 43.71 seconds. All 83 regular Rust tests, Rust 1.92 Clippy and
formatting also pass. These synthetic-input probes find no backward-kernel
failure; they do not prove correctness for all captured inputs or identify the
remaining quality bottleneck.

### Bounded balanced-training exposure test

The diverse all-cold stages each visit only 2,030 of 2,560 observations in
4,000 updates (79.3% coverage, 1.5625 draws per observation); individual scenes
receive 76–117 draws. The successful same-scene cold-noise diagnostic supplies
2,000 draws over 64 observations (31.25 per observation). These are stage-local
counts, not the weights' complete training ancestry. Together with the modest
training-set fitting gain and passing gradient probes, they leave insufficient
optimization exposure as a concrete hypothesis; they do not establish it.

Test a bounded training package in the retained width-16 model: start from the
broad parent `89e81df0…51bbe`, use fresh Adam, seed 31, the unchanged 40-scene
training / ten-scene development corpus, two-frame unroll, reset every other
window, unchanged losses and 16/8 history. Use 16,000 updates, initial learning
rate 0.0003 and the existing cosine schedule. This restores mature-history and
temporal-loss supervision omitted by all-cold training, while increasing cold
exposure. It tests the package, not isolated attribution to duration, rate or
sampling. Evaluate all 640 causal development frames at 4,000-update intervals,
retaining full-precision outputs. Do not use width-32 or single-scene fit weights.

Before starting, pass an eight-update production-resolution debug train/reload
check using the retained parent and sampling policy. Use the already recorded
clean f32 release executable for the broad run. Stop on numerical or validation
failure; otherwise finish the fixed budget without automatic extension. Assess
cold/mature crops, edge/texture error, energy, lighting response, temporal scores
and worst frames together. Repeat the fixed 50-frame fitting probe for any
development-selected candidate. Do not consume confirmation data merely because
training loss or mean PSNR falls; a visibly useful, development-supported result
and a new immutable selection record are still required.

The debug gate passed with zero validation errors and 49 PNG/f32/frame-score
files byte-identical after reload (`balanced-exposure-{smoke,reload}-run/`).
The broad run was started in `balanced-exposure-training/`, writing
`balanced-exposure-fit/`, with the recorded clean `d1da76d` release executable.
Its initialization is the broad parent, not the smoke checkpoint. No quality
improvement was claimed before the development evaluations completed. It was
subsequently interrupted and closed without promotion; see the closure above.

The comparison renderer now verifies encoded frame count, dimensions and rate
with FFprobe. Optional review sheets cover all 64 frames consecutively, four
per page, instead of eight sampled frames. Three regression tests and a real
64-frame development rendering pass (`complete-review-smoke-run/`, 16 native
768×1152 sheets). First/last pages were inspected for layout; this is a tool
check, not complete visual review or a perceptual playback pass for a candidate.

## Progress and evidence

- Initial validation reproduction:
  `runs/quality-week-2026-09-26/validation-baseline/manifest.json` records failure:
  nine `VUID-StandaloneSpirv-None-10684` errors in the native recurrence test,
  despite numerical CPU/GPU agreement within approximately 7.8e-7. This was an
  initial failure; the later compiler repair is documented above.
- Controlled fit captures are complete: `data/fit-train.omd`, `data/fit-dev.omd`,
  `data/fit-noise.omd`, and `data/fit-reference-b.omd`, all under
  `runs/quality-week-2026-09-26/`. Their capture-run manifests record the exact
  commands, inputs and driver. The reference pair has 37.60 dB pairwise PSNR,
  implying an approximately 40.62 dB single-reference noise floor. Reference
  lobes composed with observed materials have compressed MSE about 3.0e-9.
- The original objective was tested for 1,000 updates, learning rate 0.0003,
  unroll 2, seed 31, starting from the published checkpoint. This is an explicit
  tiny-scene fit, not a candidate for publication. Result checkpoint:
  `fit/model.safetensors`, SHA-256
  `a29a6aa180cf5700e835beff627d14a1fb10c2940a61d25fac1471d7818f47bf`.

  | Diagnostic | Starting RGB PSNR | Fitted RGB PSNR | Starting specular MSE | Fitted specular MSE |
  |---|---:|---:|---:|---:|
  | Training inputs | 29.09 dB | 32.47 dB | 0.000632 | 0.001226 |
  | Independent input noise | 28.95 dB | 30.22 dB | 0.000682 | 0.001408 |

  The visually inspected result still has ceiling streaks and blotches. Combined
  RGB improves while separate specular error roughly doubles. This exposes a
  supervision ambiguity: the old objective supervises combined RGB and temporal
  lobe changes, but not absolute lobe values. A CPU regression now demonstrates
  that incorrect diffuse/specular values can compose to identical RGB.
- Test absolute compressed-lobe supervision with weight 0.5, holding the starting
  checkpoint, data, update count, learning rate and seed fixed. This changes the
  training objective, not the inference architecture. The new objective passes
  independent f64 directional-gradient and fused/unfused GPU-gradient checks
  (`lobe-gradient-check/manifest.json`). Those release numerical checks do not
  waive the outstanding debug validation failure.
  The completed experiment (`fit-lobes/model.safetensors`, SHA-256
  `ab6bf4527c63143041992c8a16c901666dee240b3d4ed679c234c0efc19ada7a`)
  scores 32.43 dB on training inputs and 30.14 dB on independent noise. On
  independent noise, diffuse-illumination MSE falls from 0.002927 to 0.001846 and
  specular MSE from 0.000682 to 0.000272 versus the starting checkpoint. This
  addresses component compensation but does not remove the visible ceiling
  streaks. Do not promote this tiny-scene checkpoint to the README.
- Meganeura now resolves directly to upstream `ee3aea4` (including its latest
  parameter-sharing correctness fix); the user's sibling branch is unchanged.
  All three release GPU numerical tests pass with this dependency, including
  full-model gradients, recurrence and training/reload
  (`current-dependency-numerics/manifest.json`). Re-evaluating the starting
  checkpoint gives identical per-frame PSNR CSV and the inspected PNG hash to
  the pre-update baseline (`updated-runtime-baseline/`).
- The static final-audit capture is complete: two 64-frame sequences, data hash
  `2d5b776cd48f292d10f315931a4ffb30d2eea8eeec3443a88858dc516ee634a7`.
  All five cases and crop locking were complete before any candidate evaluation;
  the first candidate's failed gate is reported above. A separate 64-frame
  diagnostic rollout of scene 310001 uses
  input sample offset 128 (`data/fit-long.omd`); it is not audit data.
- **The long-rollout diagnostic fails.** On that independent 64-frame noise
  stream, the starting checkpoint averages 28.73 dB. The RGB-only tiny-scene fit
  averages 27.97 dB and drops from 30.83 dB at frame 7 to 25.81 dB at frame 63.
  The lobe-supervised fit averages 24.30 dB and drops from 30.96 to 19.03 dB;
  its mean energy ratio is 1.129. Both fitted checkpoints are rejected as quality
  candidates, despite their short-clip gains. This also demonstrates that a
  lower temporal-change error alone can hide accumulated reconstruction bias.
  Evidence: `baseline-long/`, `rgb-fit-long/`, and `lobes-long/`.
  This led to testing the residual being added after history blending, and the
  mismatch between eight-frame training sequences and a 32-frame diffuse-history
  cap. The corrective experiments below are diagnostics, not final audit results.
- A separate spatial-support diagnostic found that the original depth cutoff
  heavily rejects valid neighbors on a grazing ceiling: at pixel (40,24), an
  eight-pixel horizontal tap has weight 0.923, versus 0.009–0.022 vertically.
  The sampled normals are identical; depth slope causes the rejection. This is
  a candidate explanation for horizontal streaks; the tested correction follows.
- The first corrected decoder corrects the incoming spatial observation before history
  blending. A 256-frame stationary regression verifies that a constant residual
  is not magnified by the accumulation length. With unchanged lobe-fit weights,
  this changes the independent 64-frame score from 24.30 to 30.28 dB and energy
  ratio from 1.129 to 0.9993 (`incoming-fitted/`). Fitting only eight frames with
  this decoder still drifts: 32.70 dB at frame 15 falls to 27.82 at frame 63
  (`incoming-trained-long/`). The incoming correction is history-dependent, so
  the algebraic regression alone does not prove learned recurrence stability.
- Spatial support now follows a robust local inverse-depth slope; quantized
  normals are normalized. Parallel-depth-edge and grazing-plane CPU regressions
  pass, as does 24-frame flat/sloped CPU/GPU parity (maximum relative discrepancy
  7.47e-7), full loss/gradient checks and training/reload (`slope-numerics/`).
  With the published starting weights, the spatial change raises the 64-frame
  score from 29.68 to 30.61 dB compared with the incoming-only decoder
  (`incoming-original/`, `slope-original/`).
  A conservative boundary guard requires both neighboring depths when estimating
  a slope; a single image-edge jump must not become valid spatial support.
  The added boundary regression and all GPU checks pass
  (`slope-boundary-numerics/`); `slope-boundary-noise/` rechecks the same long fit
  with the retained boundary behavior and saves full-precision outputs.
- The controlled training and separate development scene were extended to 64
  frames with input sample offset 256 (`data/fit-{train,dev}-long.omd`). Two
  matched 1,000-update runs start from the published weights, seed 31, learning
  rate 0.0003, unroll 2 and lobe loss weight 0.5. On the independent offset-128
  diagnostic stream, incoming-only scores 33.23 dB; adding slope-aware support
  scores 33.61 dB, versus 28.73 for the original model. The latter rises from
  32.65 dB at frame 7 to 34.32 at frame 63; mean energy ratio is 1.00015.
  The inspected surfaces are substantially cleaner, without the earlier
  accumulated drift. Evidence: `incoming-long-noise/`, `slope-long-noise/`.
  **These weights overfit one scene:** the separate development scene scores
  only 28.01 dB at update 1,000, down from 30.30 at 250. Neither diagnostic
  checkpoint is a publication candidate. Diverse long sequences are the next
  targeted data change; start from the published weights, not the tiny-scene fit.
- The original inference implementation is preserved outside tracked source.
  An evaluation-only full-precision-output addition was built from `049a7bd`
  (`initial-linear-build/`, `initial-linear-runtime/`). Its per-frame CSV and
  inspected PNG hash match the original baseline exactly on independent noise
  (`check-initial-linear/`). Before/after must use each version's own decoder,
  not both sets of weights through the new decoder. Use separate Cargo target
  directories when rebuilding historical worktrees: shared build artifacts can
  otherwise leave an executable linked to the wrong library revision.
- All 75 non-ignored Rust tests pass on the default and Rust 1.92 toolchains;
  Clippy and formatting pass on both toolchains, and all five crop-scoring
  Python tests pass. The three release GPU numerical tests also pass.
  Debug Vulkan conformance failed at this stage. The later patched-compiler
  checks above establish a separate conformance result; the original failure
  records are retained unchanged.
- Candidate crop scoring now supports `--selection FILE` to verify the frozen
  checkpoint, sidecar, executable, clean source/build record and chronology,
  including complete causal frame coverage. Eight Python tests cover crop math,
  changed/unrecorded artifacts, both locked audit hashes, bad timestamps,
  swapped checkpoints, dirty builds and reset/partial evaluations. Re-scoring
  the first audit with its original selection record produces exactly identical
  values for all 78 crop/frame pairs and every aggregate
  (`audit-selection-verification-run/`). The 44.8% smooth-error reduction still
  fails the quality gate; provenance verification does not change that verdict.
- Held-out learned quality gates, final uncontended timings, and videos remain
  pending. No README quality improvement is claimed by these diagnostics.
