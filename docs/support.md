# Learned source support and projected-colour supervision

Both modes are opt-in. Published checkpoints and the default runtime are unchanged.
[Original two-seed study](results/support-lavapipe-2026-09-08.md) ·
[Geometry consistency and corrected noise study](results/geometry-noise-lavapipe-2026-09-08.md).

## Field: `--view-fusion visible-rgb`

The shared RGB/camera-ray encoder predicts 16 finite termination bins plus escape
per source pixel. Bilinear projection samples these distributions; their survival
probabilities weight geometry features and source-colour copying. Absolute mean
survival also limits copy authority, so normalized weights cannot revive an
occluded source. Density/visibility never see the requested viewing direction.
This is a monocular depth head, not a multiview cost volume or a relighting solver.

`--visibility-weight` supervises source-camera termination. Out-of-bounds hits and
uncertified misses are masked. Labels never enter observations, rendering queries
or exported RGB contexts. The zero-initialized head preserves common parameter
initialization; Moments/LateRgb retain their original weight schemas. VisibleRgb
adds per-source pixel-major probabilities after the four `build_points` outputs.

Optional `--consistency-rays N --consistency-weight F` compares source and volume
termination CDFs along observation-selected rays, including escape. It adds no
weights or inference inputs, and requires samples divisible by 16. Weight zero
keeps the same training graph and rays. Agreement is not proof of correct depth.

## Selector: `--projected-weight`

This rejected-as-an-improvement experiment projects true linear RGB lobes onto
the available candidate hull and supervises the unique projected colour rather
than nonunique weights. The teacher is detached; reference losses stay unchanged,
and teacher colours never enter recurrence. Weight zero retains the graph and
teacher computation. No inference parameters or inputs are added. Per-example
linear MSE already has the same constrained optimum; this is not new scene information.

## Reproduce

`bash benchmarks/support-lavapipe.sh ARM` accepts `late-rgb`, `visible-rgb`,
`visible-unsupervised`, `selector-control`, `selector-projected`, `selector-linear`.
The last increases ordinary physical weighting instead of adding projected targets.
Use `SUPPORT_SEED=7` or `11`; default is 512 updates. Corrected captures come from
`benchmarks/quality-controls.sh`. Results cannot overwrite an existing checkpoint.
`benchmarks/report-support.py ROOT` requires all 11 original study reports.

For the newer experiments use `benchmarks/geometry-lavapipe.sh` and
`benchmarks/report-geometry.py`; the result document specifies exact budgets.
These are construction/development studies, not new-scene audits. No LavaPipe
speed claim or improved shared-weight transfer is established.

Primary context: [NeuRay, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Neural_Rays_for_Occlusion-Aware_Image-Based_Rendering_CVPR_2022_paper.html)
and [IBRNet, CVPR 2021](https://ibrnet.github.io/). This compact experiment is not
a reproduction of either system.
