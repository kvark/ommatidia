# Sampling, capacity and objective controls

These are construction/development experiments, not new audit sets or speed
benchmarks. Keep RGB-only field observations and supplied realtime geometry.

## Field

`field --stratified` samples once uniformly inside each fixed ray interval while
training. Widths, ray bounds and surface-target classes do not move. A separately
seeded stream preserves fitting-pixel and auxiliary-probe choices. Evaluation
uses deterministic midpoints. No target depth enters sampling.

| Arm | Training queries | Core / hidden | Compare against |
|---|---|---|---|
| field-midpoint | 32 midpoints | 8 / 32 | Previous surface-supervised control |
| field-stratified | 32 stratified | 8 / 32 | Midpoint: quadrature locations |
| field-fine | 64 stratified | 8 / 32 | Stratified: ray resolution |
| field-wide | 64 stratified | 16 / 64 | Fine: capacity |

All use the same 64x64 construction scene, source/fitting/validation cameras,
32 camera rays per update, 16 emission probes, 0.05 surface weight, and 2,048
updates by default. Doubling samples also refines termination-label bins and
evaluation quadrature; it is not a pure compute-matched or inference-only test.
Compare RGB/depth, not raw categorical NLL between different bin counts. The
wider model changes initialization shape and compute, not just one stored weight.

## Realtime

`transport` exposes nonnegative `--compressed-weight`, `--physical-weight`,
`--low-frequency-weight`, `--confidence-weight` and `--temporal-weight`. Defaults
remain 1, 0.1, 0.05, 0.01 and 0.01. They are training inputs, not checkpoint or
inference settings. Zeroing a term retains the same graph, sampling and initial
parameters; recurrent states naturally diverge after optimization.

The four arms use fixed-exposure normalization: the existing control, then
independent removals of compressed, confidence and temporal losses. Use the
same eight fitting scenes and four development sequences, 512 updates, two-frame
BPTT and eight channels. Seed 50000 has already been inspected: do not call it
untouched. The full physical/low-frequency objective remains active in all arms.

## Reproduce

```
for arm in field-midpoint field-stratified field-fine field-wide \
           transport-control transport-no-compressed transport-no-confidence transport-no-temporal; do
  bash benchmarks/quality-controls.sh "$arm"
done
python3 benchmarks/report-controls.py target/quality-controls
```

`QUALITY_DATA`, `QUALITY_RESULTS`, `QUALITY_UPDATES` and `QUALITY_SEED` override
paths/budgets. For identical capture hashes across both tracks, first run both
`field-midpoint` and `transport-control` with `QUALITY_CAPTURE_ONLY=1`. The manual
`controlled reconstruction study` workflow performs that preparation once,
distributes the exact captures to all eight arms, then verifies the reports.
Delete or rename an old data directory deliberately before changing capture
options: existing files are reused, never silently overwritten.

All checkpoints reload before scoring. Recipes contain capture hashes and exact
commands. Reports include parameter/query counts, fitting and validation quality,
geometry and source-image ablations. The summary refuses mixed capture hashes
or different deterministic transport baselines. Use each arm's full images and
per-sequence scores as well as means. No automatic checkpoint promotion occurs.
