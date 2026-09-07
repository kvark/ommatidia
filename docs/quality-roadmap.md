# Quality roadmap

Two tracks: realtime denoising and posed-RGB field reconstruction. Shared code
is not yet successful weight transfer. Neither new path has a promoted checkpoint.
Keep [representative images](../README.md) visible and
[historical results](results-overview.md) separate from new measurements.

## Completed: surface targets and construction diagnostics

[Surface supervision](surface.md) trains existing field density through first-hit
termination and certified misses. Emitter support, centre-ray RGB/depth agreement,
RGB-dependence controls, geometry metrics, reload and GPU tests are implemented.
All truth stays out of inference and sample placement; a hit does not declare
occluded space behind it empty.

The [64×64 paired construction run](results/surface-lavapipe-2026-09-07.md) used
2,048 updates, an eight-channel core and 32-wide query network. Surface weight
0.05 raised validation-camera PSNR from 17.15 to 18.08 dB, reduced linear RGB
MSE 55.8%, and depth MAE 48.9%. This is one construction scene, not an unseen-scene
result. Fitting cameras still lack sharp boundaries. Keep the loss opt-in while
evaluating more scenes and schedules; the larger default remains untested.

A post-fit CPU replay matched saved GPU images. Denser inference alone (32 to
128 samples) did not consistently help. A nondeployable true-surface oracle
still had substantial appearance error. Sampling, appearance and correspondence
remain unresolved. The earlier [incident ablation](results/incident-lavapipe-2026-09-07.md)
was mixed; that loss also remains opt-in.

## Next field experiments

**1. Reach a sharp fitting result.** Independently increase training exposure,
capacity and training sample density at 64 pixels before moving to 128. Test
stratified intervals or predicted coarse-to-fine sampling to avoid fitting only
a fixed midpoint grid. Denser inference on a coarse-quadrature checkpoint is
not that experiment. Never use target depth to position queries. Log pixel/ray
exposure and separate loss terms.

**2. Resolve multiview correspondence.** Compare per-view visibility-weighted
aggregation or a small depth-hypothesis volume against unconditional feature
mean/variance pooling. Frustum membership is not occlusion visibility. Use
true-surface, fit-camera and RGB-permutation controls to separate geometry from
appearance failure; no oracle is a deployable result.
[MVSNeRF](https://apchenstu.github.io/mvsnerf/) supplies a correspondence-volume
baseline; [DS-NeRF](https://www.cs.cmu.edu/~dsnerf/) is prior termination-supervision work.

**3. Generalize, then enrich transport.** Evaluate independent geometry families,
camera layouts and lighting seeds. Re-run incident supervision after recognizable
structure and emitters emerge. Expand directional environments and materials.
Arbitrary relighting needs material/visibility/illumination factorization; an
appearance field under captured lighting is not automatically a transport solver.

## Realtime: remove energy loss, then beat fixed reconstruction

The [expanded native run](results/transport-lavapipe-2026-09-07.md) used eight
fitting scenes, four fitting-disjoint sequences, 512 updates, 1-spp input and
32×32 to 64×64 reconstruction. Temporal error fell 37%, but energy ratio fell
from 1.007 to 0.969 and low-frequency PSNR from 34.86 to 32.45 dB. It is not
promoted. The baseline is the fixed candidate mixture, not SVGF.

The physical/low-frequency/temporal losses used target-dependent relative weights.
The [paired fixed-exposure test](results/normalization-lavapipe-2026-09-07.md)
gained 0.62 dB spatial and 1.44 dB low-frequency PSNR over that learned arm on
four fresh validation sequences. Energy improved from 0.936 to 0.953 but stayed
below the deterministic baseline's 0.998. Temporal error worsened versus the
relative arm. Keep `--fixed-exposure-loss` opt-in: a better spatial tradeoff is
not a quality promotion or complete removal of bias.

Next isolate the remaining compressed objective and confidence/temporal terms
on construction controls. Include real meshes from initialization, longer
trajectories, cuts, disocclusions, thin geometry and moving reflections. Check
expected previous-view depth under translation before loosening rejection.
Require broad lighting and temporal gains without darkening, blur or ghosting;
then audit the selected recipe on fresh families.

Compare with matched-input SVGF before DLSS Ray Reconstruction. ReSTIR+SVGF is
a separate pipeline comparison. LavaPipe is for quality and correctness;
production-GPU speed and memory are separate later gates.

## Shared weights: after useful standalone models

Compare independent training, field-pretrained initialization and joint training
at controlled budgets. Keep supplied realtime geometry authoritative. Group all
views/lighting variants of one geometry into one split; changed lighting must
not be forced into invariant radiance features. Retain shared weights only if
both task audits improve. OLATverse and blade-volume integration are not prerequisites.

## Evidence rules

Construction data diagnoses implementation; validation selects designs; fresh
family-disjoint audits test final claims. Retire an opened audit into development.
Report full frames, depth/opacity, energy, detail, temporal error and per-case tails
alongside means. Name RGB transforms and block extents; quantify reference noise.
Reload checkpoints before scoring. Repeat optimization seeds and use larger
independent scene/camera/light sets before promotion.
