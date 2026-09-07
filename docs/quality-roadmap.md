# Quality roadmap

## Status after the September 7 architecture change

The native multiscale denoiser, posed-RGB field, and incident-light supervision
are implemented. Contract tests pass; neither new path has a promoted quality
checkpoint. Keep realtime denoising and offline reconstruction as separate
quality tracks. Shared code is not yet successful weight transfer.

The field's latest [paired ablation](results/incident-lavapipe-2026-09-07.md)
used 16x16 images, a four-channel core and 128 updates. Incident supervision
reduced total incident log-error 1.29%, but image PSNR fell 0.13 dB and indirect
error rose 3.58%. Both outputs remain blurry. Keep that loss opt-in; diagnose
geometry, sampling and learning capacity before adding another lighting loss.

Historical measurements remain in [results-overview.md](results-overview.md).
This roadmap replaces the older immediate experiment order; negative tests of
old models are not blanket rejections of new architectures.

## Field track: recover structure before richer transport

### 1. Establish a useful fit, not another tiny smoke result

Use a construction-only scene with visible emitters, occlusion and texture.
Start at 64x64, then 128x128, with several overlapping posed source views. Run
fixed-budget controls for a wider model and denser ray sampling, changing one
factor at a time. Log image rays/pixels seen, steps, gradient norms and separate
image/emission/incident losses. The 64-channel/128-hidden default has not yet
been evaluated by the tiny ablations.

First require a sharp fit to fitting cameras. Then test distinct validation
cameras. Shuffle/zero source images to check whether predictions actually depend
on the scene evidence rather than a spatial average. If fitting fails, use an
explicitly oracle-only true-surface rendering diagnostic to distinguish decoder,
geometry and sampling failures. Never count that oracle as the RGB-only model.
Do not inspect the next untouched audit set while making these decisions.

### 2. Supervise geometry without providing it at inference

Add training-only first-hit distance, hit/miss, and geometric-surface records to
synthetic captures. Supervise the ray-termination distribution around the true
surface, free space before it, and transmittance of miss rays. Do not mark
occluded space behind the first hit as empty or assign an arbitrary ground-truth
volume density to a triangle surface. Treat transparent materials separately.

Tie emitter supervision to occupied surface support as well as emitted radiance;
an emission value at a point alone does not establish an opaque light source.
Keep all labels out of runtime contexts and ray sampling. The inference contract
remains posed RGB plus declared bounds. Measure depth, silhouette and emitter
localization in addition to RGB, so incorrect geometry cannot hide behind colour.
[DS-NeRF](https://www.cs.cmu.edu/~dsnerf/) is a primary reference for supervising
ray termination rather than relying on RGB alone.

### 3. Improve correspondence and ray coverage where controls justify it

Compare denser uniform samples against coarse-to-fine sampling driven by predicted
weights, never target depth. Test thin surfaces and small emitters explicitly.
If useful surface structure still fails despite a good fitting-camera result,
replace unconditional multiview mean/variance pooling with a small plane-swept
correspondence volume or learned visibility-weighted aggregation. Camera-frustum
membership is not occlusion visibility. Preserve per-view evidence until the
model can resolve disagreements. [MVSNeRF](https://apchenstu.github.io/mvsnerf/)
provides a concrete correspondence-volume baseline, not a novelty claim.

Only after this control recovers boundaries and sources should incident
supervision be rerun against the same stronger model. Next extend directional
environments and lighting diversity; arbitrary relighting additionally requires
a material/visibility/illumination factorization, not just a radiance head.

## Realtime track: beat the deterministic reconstruction

Do not let field experiments displace the original denoising goal. Run the new
native path on fresh matched-depth captures, comparing its fixed candidate
mixture against learned multiscale lobe selection and confidence/BPTT ablations.
Use identical noisy frames, scene families and ray budgets. The
[existing LavaPipe recipe](../benchmarks/transport-lavapipe.sh) is a starting
harness, not a broad quality study.

Include real meshes from initialization and longer causal sequences with camera,
object and light motion, cuts, disocclusions, thin geometry and glossy reflections.
Check expected previous-view depth under camera translation before tuning
rejection thresholds. Preserve actual linear HDR energy, exact albedo/emission
and separate lobe histories. Require gains in broad lighting error and temporal
stability without ghosting, darkening or lost detail.

Compare with matched-input SVGF before DLSS Ray Reconstruction. ReSTIR+SVGF is a
separate pipeline control, not a same-input denoiser comparison. Use LavaPipe to
render, train and judge quality; production GPU timing is a later, separate gate.

## Shared-core experiment comes after useful standalone models

At equal architecture and budget, compare independent training, field-pretrained
core initialization, and joint training. Keep geometry/material inference in the
field adapter and supplied surface data authoritative in the realtime path.
Light changes should preserve geometry, not force radiance features to be
invariant. Group all views and lighting variants of a geometry into one split.
Retain shared weights only if fresh task-specific evaluations improve without
negative transfer. OLATverse and blade-volume integration are not prerequisites.

## Evidence for every quality decision

Use fitting data for optimization, validation for choices, and fresh scene/family
holdouts for the final audit. Retire an opened audit set into development rather
than repeatedly calling it untouched. Measure reference convergence separately.
Publish full frames, fixed crops, motion sequences, per-scene tails, named RGB
transforms, linear energy and structural/temporal metrics beside means. Promote
only after repeat seeds and a larger independent scene/camera/light evaluation;
no LavaPipe timings or tiny smoke gains are product claims.
