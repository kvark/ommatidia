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
  --steps 4000 --channels 16 --unroll 2 --lr 0.0003 --eval-every 500
cargo run --release -p ommatidia-train --bin transport -- \
  --checkpoint runs/model/model.safetensors --eval-only \
  --eval-data data/audit.omd --out runs/audit
```

Repeat `--data` and `--eval-data` to combine captures with equal dimensions and
sequence lengths. Training samples sequences uniformly across the combined
corpus (larger captures contribute more). Evaluation can use another resolution.
Prefer training at the target evaluation resolution: identical parameter shapes
do not imply identical pixel-footprint or history statistics.
`--checkpoint` during training is a weights-only warm start, **not** an Adam
resume. The serialized config, training provenance and intermediate checkpoints
are saved with the run. Existing final checkpoints are not overwritten.

## What the numbers mean

The control is the same multiscale/recurrent guide with a zero residual head,
running its **own** history. Both methods receive identical observations.
All frames are evaluated causally, with a reset at each sequence boundary.
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
  non-reset pixels with no accepted reprojection in either lobe; empty regions
  are null, never perfect zeros. Reactivity suppression is not disocclusion.

PNGs use the same compression followed by sRGB; no per-image exposure or
postprocessing. All frames are saved. README pictures must be copied from those
outputs with hashes and checkpoint/data provenance, never generated or retouched.

Each evaluation also writes `diagnostics.json`: per-frame diffuse illumination,
material-weighted diffuse radiance and specular radiance errors, mean history
ages, and the error from composing reference lobes with observed material data.
Use `--save-lobes` for matched component PNGs. `--eval-only --reset-history`
resets both models before every frame while still measuring inter-frame changes;
it is explicitly labeled a diagnostic, not the causal production result.

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
