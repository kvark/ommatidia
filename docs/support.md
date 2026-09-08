# Learned source support and projected-colour supervision

Both modes are opt-in; published checkpoints and the default runtime are unchanged.
[Measured two-seed results and limitations](results/support-lavapipe-2026-09-08.md).

## Field: `--view-fusion visible-rgb`

The shared RGB/camera-ray encoder predicts a 17-way distribution for every source
pixel: 16 finite ray intervals through the declared acquisition bounds plus escape.
This is a monocular, image-conditioned depth head, not a multiview cost volume.
Bilinear projection samples the distributions and integrates survival at each 3D
query with a fixed half-bin surface tolerance. Weighted first/second feature
moments feed density and appearance. Source-copy authority is also multiplied
by absolute mean survival: a lone nearly occluded view cannot normalize itself
back to full authority. Density/visibility never see the requested view direction.

`--visibility-weight` weights training-only source-camera termination labels.
Hits outside acquisition bounds and uncertified misses are masked. Labels never
enter observations, queries or exported RGB contexts. Evaluation/recovery needs
no labels for the forward pass. Width, positions, cameras and samples stay
unchanged; this does not make the field physically relightable. Source-depth and
implicit density are not yet tied by a consistency loss.

The new head is zero-initialized without RNG draws, preserving common parameter
initialization. Moments and LateRgb keep their schema; VisibleRgb needs new
weights. `build_points` appends per-source pixel-major probabilities after its
original four outputs for this variant only.

## Selector: `--projected-weight`

For each actual training candidate/history state, project target linear RGB onto
the convex closure of available candidates. Supervise the unique projected colour,
not nonunique mixture weights. Auxiliary MSE uses fixed exposure. Real targets,
reference temporal changes, confidence labels and runtime history stay unchanged.
The teacher is detached and never enters recurrence. The coefficient-zero arm
computes the same teacher and uses the same graph. No new learned parameters or
inference inputs are introduced.

This tests whether an attainable target is easier to learn, not an unbiasedness
guarantee. Per-observation squared-error minimization already has the same
constrained optimum; it is alternative optimization/supervision, not new scene
information. Averaging over noisy inputs can change the statistical objective.

## Reproduce

Run `bash benchmarks/support-lavapipe.sh ARM` with `SUPPORT_SEED=7` or 11. The recipe
expects cache-corrected captures in `target/quality-control-data`; the existing
quality-controls capture recipe produces them. It refuses to overwrite results
and records commands, hashes and source revision. Field arms are `late-rgb`,
`visible-rgb`, `visible-unsupervised`; selector arms are `selector-control`,
`selector-projected`, `selector-linear`. The last doubles ordinary physical loss
instead of adding projected targets. All default to 512 updates. Images, input
ablation checks and candidate-oracle diagnostics are retained.

`python3 benchmarks/report-support.py ROOT` summarizes all 11 completed runs and
rejects missing reports, unequal field contexts/references or unequal deterministic
selector baselines. An interrupted post-training evaluation can be recovered with
`transport --eval-only --eval-data DATA --out COPIED_RUN --candidate-oracle` after
copying the saved checkpoint and `.transport.ron` to a separate directory. Keep
original training metadata and verify unchanged checkpoint bytes.

Field image-ray/update budgets match, but source-depth supervision adds all valid
source pixels and the head adds computation: not equal label exposure/FLOPs. The
field remains a construction-scene experiment; the denoiser reuses development
sequences, not a fresh audit. Report every seed and energy/detail/temporal failure.
No LavaPipe speed claims.

Primary context: [NeuRay, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Neural_Rays_for_Occlusion-Aware_Image-Based_Rendering_CVPR_2022_paper.html)
uses ray visibility for image-based fields; [IBRNet, CVPR 2021](https://ibrnet.github.io/)
motivates retaining source observations until aggregation. This compact
implementation is not a reproduction of either system.
