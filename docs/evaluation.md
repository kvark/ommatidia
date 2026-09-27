# Training and evaluation

Use fresh Blade captures with matching input/reference path depth, 1-spp sparse
inputs, split radiance, output-resolution primary surfaces, projection jitter,
and moving cameras, objects and lights. The trainer requires a matching
`.transport.json`, rejects entirely black/nonfinite reference frames, and
rejects overlapping training/development scene seeds or catalog families.
Canopy cameras stay on the open -X/-Z side, below the roof.
For catalog objects, require measured visible coverage, not just asset ids in a
sidecar. Current captures preflight camera trajectories and record coverage and
driver versions; the trainer rejects measured coverage below 1%. Missing
coverage in historical captures is not evidence that their assets were visible.

Hold out scenes **and** assets. Subdivide the asset holdout into development and
final audit before training. Development selects checkpoints; do not use final
audit frames for selection. Preselect illustrated frames, retain all per-frame
scores, and include resets, rejected history and longer-than-training rollouts.

```sh
cargo build --release --workspace
cargo run --release -p ommatidia-train --bin transport -- \
  --data data/train.omd --eval-data data/dev.omd --out runs/model \
  --steps 5000 --channels 16 --unroll 4 --batch 8 --crop 64 --lr 0.0003
cargo run --release -p ommatidia-train --bin transport -- \
  --checkpoint runs/model/model.safetensors --eval-only \
  --eval-data data/audit.omd --out runs/audit
```

Repeat `--data` and `--eval-data` to combine captures with equal dimensions and
sequence lengths. Training samples sequences uniformly across the combined
corpus (larger captures contribute more). Evaluation can use another resolution.
Default training uses eight persistent cursors, each with a fixed 64² LR crop
and four-frame gradient windows. Each life reserves 8–16 windows where possible;
its start is uniform among positions that fit that lifetime. Cursors respawn
cold, and each window additionally resets with probability 0.1. Actual cold
window counts/fractions are in `loss.csv`; the target is 10–20% (tiny CI sequences
necessarily reset more often). A per-life radiance gain of `2^U(-2,2)` scales
inputs, targets and emission together, without changing exposure from 1.
There is no prefix warm-up, no CPU history/feature round trip, and no geometric
augmentation. Out-of-crop warp taps are dropped/renormalized; losses exclude a
four-pixel HR margin and normalize over the retained pixels.

One clipped Adam update uses the mean gradient over the cursors, with a 500-step
linear warm-up then cosine decay to 10%. Read-only memory maps and a bounded
prefetch worker load crops only. **Do not modify or truncate captures while a
trainer or evaluator has them mapped.** Evaluation remains full-frame and causal;
`--reset-history`/`--reset-every` are evaluation-only diagnostics. Train captures
should retain the evaluation's pixel footprint even though the loss uses crops.

Training starts from scratch. Every 5,000 steps and at the requested stop point,
`checkpoints/step-NNNNNNNN/` saves weights, Adam moments/step, cursor GPU states,
sampler RNG/positions, schedule settings and ordered capture hashes. The final
bundle is also copied beside `training.json`. `--checkpoint FILE` resumes only
a complete matching bundle; it rejects weights-only warm starts or changed data
and schedule settings. Use a new output directory and repeat the same training
arguments, including the original total `--steps`; `--stop-after N` simulates an
interruption without changing that schedule. Existing checkpoints are not overwritten.
The [measured profile](training-profile.md#phase-3-cursor-loop) excludes startup,
checkpointing and evaluation, and counts only supervised pixels as gradients.

## What the numbers mean

v4 evaluates one live model (`learned`). The v3 guide and zero-head baseline
have been removed. Compare against archived v3 float outputs via `--control-run`;
historical reports retain their original `baseline` fields for reproduction.
By default frames are evaluated causally, with a reset at each sequence boundary.
The standard development protocol also runs `--eval-only --reset-every 16`,
simulating periodic cuts within each sequence. `frames.csv` records sequence,
frame, `frames_since_reset`, and per-frame metrics. `quality.json` reports cold
(age 0), early (1–7), settling (8–15), and warm (≥16) separately. Reset-16 has
no warm frames; that aggregate is null. Temporal pairs crossing resets are excluded,
so cold temporal MSE is also null, not zero.

`--control-run DIR` loads a previous evaluation's saved float RGB and reports
`control_*` and `delta_*` (learned minus control) columns. It requires the same
ordered capture provenance, extent, sequence length, complete frame ordering and
reset protocol. References must match **byte for byte** at every frame. Missing,
invalid or mismatched outputs fail the run; they are not silently skipped.
When updating a published result, also evaluate the previous trained checkpoint
on exactly the same new frames. A different dataset or stronger fixed-guide
comparison alone does not establish an improvement over the previous model.

- PSNR and SSIM use fixed `x/(1+x)` compression, averaged over frames.
- Low-frequency PSNR checks broad error after spatial averaging.
- Linear MSE, relative MSE and energy ratio catch brightness/energy bias.
- Detail ratio is a gradient-magnitude diagnostic, not proof of correct texture.
- Gradient MSE compares horizontal/vertical compressed-RGB differences against
  the reference, penalizing both missing edges and spurious detail.
- Temporal MSE compares motion-compensated output changes with reference changes,
  rejecting mismatched primary surfaces. It is not a perceptual video metric.
- Reset PSNR isolates cold starts. Rejected-history MSE is pixel-weighted over
  non-reset pixels with no in-frame geometric warp tap; empty regions are null,
  never perfect zeros. This v4 coverage mask is not a learned rejection or
  disocclusion measure and differs from v3's geometry/material-tested mask.

PNGs use the same compression followed by sRGB; no per-image exposure or
postprocessing. Candidate evaluation saves every frame unless `--no-images` is
set. Training checks are always causal, write no PNGs, and run every 10,000
updates by default (`--eval-every`), plus the final checkpoint. Use separate
`--eval-only` runs for the full candidate protocol and `--save-linear` for scoring.
README pictures must be copied from those
outputs with hashes and checkpoint/data provenance, never generated or retouched.

Each evaluation also writes `diagnostics.json`: per-frame diffuse illumination,
material-weighted diffuse radiance and specular radiance errors, and the error
from composing reference lobes with observed material data. v4 has no history ages.
Use `--save-lobes` for matched component PNGs. `--eval-only --reset-history`
resets the model before every frame; all temporal pairs are excluded. It is a
spatial diagnostic, not the causal production result.

For frozen comparisons, save unquantized images with `--save-linear` and first
verify the recorded dataset/benchmark hashes with `scripts/score-regions.py` as
described in the [archived quality protocol](archive/quality-week.md). For a selected learned
candidate, also pass `--selection FILE`, and include that freeze file as an
input when recording its evaluation. The scorer checks the selected checkpoint,
actual `model.transport.ron` sidecar, executable and clean build revision; the
build must precede selection, and selection must precede evaluation. It rejects
partial or reset-every-frame candidate evaluations. A follow-up freeze records
both `benchmark_sha256` and `confirmation_benchmark_sha256`. These are identity
and chronology checks, not proof of sound checkpoint selection or visual quality.

## Development protocol and confidence intervals

Use the five development captures, in the exact order of `docs/dev-crops.json`,
with the same checkpoint for both causal and `--reset-every 16` evaluations.
Record the executable, checkpoint/config, benchmark, ordered captures and sidecars
as inputs with `scripts/record-run.py`. Do not use confirmation data here.

Install `scripts/evaluation-requirements.txt` in an isolated Python environment
(the recorded environment uses Python 3.12). `score-flip.py` uses the
[official NVIDIA FLIP implementation](https://github.com/NVlabs/flip), version 1.7,
at 67 pixels/degree, on unquantized float RGB after the same compression and sRGB
transform as the pictures. It writes a new score directory, preserving the source.

```sh
target/evaluation-env/bin/python scripts/test-flip.py --reference-dir target/flip-reference
target/evaluation-env/bin/python scripts/score-flip.py --run runs/control/causal --out runs/control/causal-scores
python3 scripts/bootstrap-evaluation.py \
  --before runs/control/causal-scores/frames.csv \
  --after runs/candidate/causal-scores/frames.csv --out runs/comparison.json
```

The bootstrap uses 1,000 **paired, whole-sequence** resamples (seed 31), retaining
frame weighting within each sample. It reports after-minus-before differences and
percentile 95% intervals for PSNR, FLIP, temporal MSE and linear energy ratio, both
overall and by reset age. Matching frame identities, protocol and per-frame
reference hashes are required. Missing temporal pairs stay missing; unsupported
strata are null. To measure the v3 control against its zero-head guide, use the same
CSV twice with `--before-prefix baseline`. This is not a v4 improvement claim.

The FLIP test pins and hashes upstream reference images and checks the published
mean 0.159691 within 1e-4. The local CPU wheel gives 0.159714609385 (absolute error
2.36e-5). Its quantized magma visualization is not bit-identical to the upstream
C++ fixture (0.76% of channels differ, at most 3/255); that additional diagnostic
is disclosed, not used to replace the prescribed mean tolerance.

Crop scoring retains all historical rectangles and frame selections. The owner
approved adding early coverage on 2026-09-27: reference-only inspection at frame 3
confirmed the same sixteen rectangles still have their declared content. Their
reference PNG hashes and selection method are in `docs/dev-crops.json`. These
sixteen additions precede the complete protocol evaluations; no learned outputs
were viewed to choose them. Scoring adds reset-age breakdowns and supports
`--before-role base` for guide/control ratios. Cold edge/texture crops and reset-16
warm frames remain absent (null, not a pass). Coverage is five of ten sequences,
at ages 0/3/31 in causal runs, not exhaustive coverage of every early frame.

On those verified outputs,
`score-sequences --benchmark FILE --before BEFORE_OUTPUTS --after AFTER_OUTPUTS
--out NEW_DIRECTORY` recomputes the same temporal metric for every frame and
sequence. It reads observed capture motion and rejects differing references;
the preceding provenance check is required, not replaced by shape checks.
Its supplementary motion-compensated 8×8-block score distinguishes broad
fluctuation from grain. Neither score is a perceptual video pass.

`python3 scripts/render-comparisons.py --before BEFORE_OUTPUTS --after
AFTER_OUTPUTS --out NEW_DIRECTORY --sequences 0 1 --review-sheets` renders the
current 64-frame, 24-fps comparison format with FFmpeg. It checks identical
references, preserves native image dimensions and only adds labels outside the
images. It decodes each encoded video to verify dimensions, frame count and
playback rate. Review sheets cover all 64 frames in order, four per page;
earlier sparse overview sheets did not establish complete frame inspection.
Sheets support frame-by-frame review but do not establish a perceptual playback
pass. Lossy video encoding is presentation-only; metrics never read the videos.

The training objective now includes absolute compressed-lobe supervision
(`--lobe-weight 0.5` by default). RGB alone cannot identify the split: diffuse and
specular errors can cancel after material composition. Set the weight to zero
only for an explicit loss ablation. The currently published checkpoint predates
this objective change; its recorded training configuration remains authoritative.

## Reference noise

Capture a second matched reference with the same arguments but
`--reference-sample-offset N`, where N is at least `--canonical-frames`.
Compare with:

```sh
cargo run --release -p ommatidia-train --bin reference-noise -- \
  --a data/reference-a.omd --b data/reference-b.omd
```

Input planes must match exactly. Report reference-pair error as well as model
error; finite-sample reference grain is not recoverable truth. The half-variance
noise-floor estimate assumes independent equal-variance references.

## Reproducibility and limits

Wrap captures, training and tests with `scripts/record-run.py RUN --input FILE
-- COMMAND...`. It records executable/input hashes, source revisions/diffs,
environment, output, exit status and validation messages. Numeric success with
GPU validation errors is recorded as failed, not silently promoted.

These are small synthetic and object-in-procedural-room corpora, not production
game traces or broad indoor coverage. Matching path depth alone does not prove
identical integrators. No matched DLSS/OIDN comparison or real-time speed claim
is established by these runs.
