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

## Progress and evidence

- Initial validation reproduction:
  `runs/quality-week-2026-09-26/validation-baseline/manifest.json` records failure:
  nine `VUID-StandaloneSpirv-None-10684` errors in the native recurrence test,
  despite numerical CPU/GPU agreement within approximately 7.8e-7. No conformance
  gate has passed yet.
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
  No candidate has been evaluated on it. Other audit cases and crop locking
  remain pending. A separate 64-frame diagnostic rollout of scene 310001 uses
  input sample offset 128 (`data/fit-long.omd`); it is not audit data.
- **The long-rollout diagnostic fails.** On that independent 64-frame noise
  stream, the starting checkpoint averages 28.73 dB. The RGB-only tiny-scene fit
  averages 27.97 dB and drops from 30.83 dB at frame 7 to 25.81 dB at frame 63.
  The lobe-supervised fit averages 24.30 dB and drops from 30.96 to 19.03 dB;
  its mean energy ratio is 1.129. Both fitted checkpoints are rejected as quality
  candidates, despite their short-clip gains. This also demonstrates that a
  lower temporal-change error alone can hide accumulated reconstruction bias.
  Evidence: `baseline-long/`, `rgb-fit-long/`, and `lobes-long/`.
  Investigate the residual being added after history blending, and the mismatch
  between eight-frame training sequences and a 32-frame diffuse-history cap.
  Any correction must pass this long-rollout gate before broad training resumes.
- A separate spatial-support diagnostic found that the original depth cutoff
  heavily rejects valid neighbors on a grazing ceiling: at pixel (40,24), an
  eight-pixel horizontal tap has weight 0.923, versus 0.009–0.022 vertically.
  The sampled normals are identical; depth slope causes the rejection. This is
  a candidate explanation for horizontal streaks, not yet an evaluated fix.
- All 70 non-ignored Rust tests pass on the default and Rust 1.92 toolchains;
  Clippy, formatting and the existing published-evidence verifier pass. The
  three separate release GPU numerical tests also pass. Debug Vulkan conformance
  still fails and has not been reclassified as success.
- Final audit captures and crop lock, a visually clean diagnostic fit, held-out
  quality gains, validation repair, OIDN comparison, timings, and videos remain
  pending. No README quality improvement is claimed by these diagnostics.
