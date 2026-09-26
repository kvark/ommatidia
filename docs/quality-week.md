# Quality sprint protocol

This implements [TASK.md](../TASK.md). Targets are not achieved results. The
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
locked in [quality-benchmark.json](quality-benchmark.json), SHA-256
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

The isolated [Naga correction](../patches/README.md) replaces decorated workgroup
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
  All five audit cases and crop locking are now complete; no candidate has been
  evaluated on them. A separate 64-frame diagnostic rollout of scene 310001 uses
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
- The retained decoder corrects the incoming spatial observation before history
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
- Held-out learned quality gates, final uncontended timings, and videos remain
  pending. No README quality improvement is claimed by these diagnostics.
