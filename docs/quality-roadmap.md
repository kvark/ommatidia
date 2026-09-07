# Quality roadmap

Two quality tracks: realtime denoising and posed-RGB field reconstruction.
Shared code is not yet successful weight transfer. Neither new path has a
promoted checkpoint. Keep [representative images](../README.md) visible and
[historical results](results-overview.md) separate from new measurements.

## Completed: surface targets and meaningful construction diagnostics

[Surface supervision](surface.md) now trains the existing field density through
first-hit termination and certified misses. Exact emitter support, centre-ray
RGB/depth agreement, zero/reassigned-RGB controls, geometry metrics, reload and
GPU tests are implemented. All surface/light truth stays out of inference and
sample placement; a hit does not label occluded space behind it empty.

The [64x64 paired construction run](results/surface-lavapipe-2026-09-07.md) used
2048 updates, an 8-channel core and 32-wide query network. Surface weight 0.05
raised validation-camera PSNR 17.15 -> 18.08 dB, reduced linear RGB MSE 55.8%,
and depth MAE 48.9%. This is one construction scene, not an unseen-scene result.
Fitting cameras still lack sharp boundaries, and the larger default is untested.
Keep surface supervision opt-in while evaluating more scenes and schedules.

A post-fit CPU diagnostic matched the saved GPU images. Denser inference alone
(32 -> 128 samples) did not consistently help. A nondeployable true-surface
oracle still had substantial appearance error. These point to unresolved
sampling/appearance/correspondence, not a reason to add another lighting loss.
The earlier [incident ablation](results/incident-lavapipe-2026-09-07.md) was mixed;
incident supervision also remains opt-in.

## Next field experiments

**1. Reach a sharp fitting result.** Independently increase training exposure,
core/query capacity and training sample density at 64 pixels before moving to
128. Use stratified ray intervals or a properly evaluated coarse-to-fine sampler
to avoid fitting only a fixed midpoint grid. Denser inference on a checkpoint
trained with coarse quadrature is not that experiment. Never use target depth
to position queries. Log ray/pixel exposure and separate loss terms.

**2. Preserve multiview evidence until correspondence is resolved.** Compare
per-view visibility-weighted aggregation or a small depth-hypothesis volume
against unconditional feature mean/variance pooling. Frustum membership is not
occlusion visibility. Use exact-surface, fit-camera and RGB-permutation controls
to separate geometry from appearance failure; no oracle is a deployable result.
[MVSNeRF](https://apchenstu.github.io/mvsnerf/) is a correspondence-volume baseline;
[DS-NeRF](https://www.cs.cmu.edu/~dsnerf/) is prior work on termination supervision.

**3. Generalize, then enrich transport.** Evaluate independent geometry families,
source/target camera layouts and lighting seeds. Re-run the incident-loss ablation
only after recognizable structure and emitters emerge. Expand directional
environments and material/lighting diversity. Arbitrary relighting still needs
material/visibility/illumination factorization; an appearance field under captured
illumination is not automatically a transport solver.

## Realtime: remove energy loss, then beat the fixed reconstruction

The [expanded native run](results/transport-lavapipe-2026-09-07.md) completed on
LavaPipe: eight fitting scenes, four fitting-disjoint sequences, 512 updates,
1-spp input, 32x32 -> 64x64. Temporal error fell 37%, but energy ratio fell
1.007 -> 0.969 and low-frequency PSNR fell 34.86 -> 32.45 dB. Do not promote it.
The deterministic candidate mixture is the baseline here, not SVGF.

The physical/low-frequency/temporal loss scales were target-dependent relative
weights. `--fixed-exposure-loss` isolates replacing only those scales, retaining
the graph, initialization, primary compressed loss and confidence term. The
[paired normalization recipe](../benchmarks/normalization-lavapipe.sh) uses fresh
evaluation seeds rather than selecting on the inspected first study. This is
an ablation, not a claim that every source of compressed-loss bias is removed.

After an energy-preserving result, isolate scale selection, confidence and BPTT.
Include real meshes from initialization, longer trajectories, cuts, disocclusions,
thin geometry and moving glossy reflections. Check expected previous-view depth
under camera translation before loosening rejection thresholds. Require broad
lighting and temporal gains without darkening, blur or ghosting.

Compare with matched-input SVGF before DLSS Ray Reconstruction. ReSTIR+SVGF is
a separate pipeline comparison. Software Vulkan is for quality and correctness;
production-GPU speed and memory are separate later gates.

## Shared weights: only after useful standalone models

Compare independent training, field-pretrained initialization and joint training
at controlled budgets. Keep supplied realtime geometry authoritative. Group all
views/lighting variants of a geometry into one split; changed illumination must
not be forced into an invariant radiance representation. Retain shared weights
only if both task-specific audits improve without negative transfer. OLATverse
and blade-volume integration are not prerequisites.

## Evidence rules

Construction data diagnoses implementation; validation selects designs; fresh
family-disjoint audits test final claims. Retire an opened audit into development.
Report full frames, depth/opacity, energy, detail, temporal error and per-case tails
alongside means. Name the RGB transform and block extent; quantify reference
noise independently. Checkpoints must reload before scoring. Repeat optimization
seeds and use larger independent scene/camera/light sets before promotion.
