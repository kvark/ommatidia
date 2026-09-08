# Quality roadmap

Two tracks: realtime denoising and posed-RGB fields. Shared code is not proven
weight transfer. Neither new path has a promoted checkpoint. Keep [README
comparisons](../README.md) visible and [historical results](results-overview.md)
separate from current experiments.

## Latest decision

[Two-seed source-support study](results/support-lavapipe-2026-09-08.md): RGB-predicted
visibility improves field colour but worsens depth; its extra labels do not
explain the whole gain. Projected-colour targets and stronger ordinary linear
loss both fail the denoiser's PSNR/energy gates. All remain opt-in. Do not repeat
larger auxiliary-weight sweeps without a new diagnosis.

[Earlier controls](results/late-fusion-lavapipe-2026-09-08.md) recovered the wider
checkpoint without retraining and established conditional selector headroom.
Surface targets, stratified sampling, late RGB, source visibility, candidate
oracles and checkpoint recovery are implemented. Procedural textures now have
content-based identities; only cache-corrected captures support new comparisons.

## Field: sharp fitting and consistent geometry

The short visibility arms saw only 16,384 image rays each across four 64x64 fitting
views. Increase declared image-ray exposure and retain intermediate fitting
metrics before inferring a capacity ceiling. Compare the wider core and late/
visible RGB at matched data, rays and optimizer budgets, first 64 then 128 pixels.
Report RGB, depth, opacity, source dependence and emitter localization together.
No sharp fit or correct separate-camera boundaries has yet been demonstrated.

The current source-depth distribution and implicit density can disagree. Measure
that disagreement on source and separate cameras, then compare a consistency
constraint with explicit per-view correspondence/depth-hypothesis aggregation.
A monocular source-depth head is not a multiview cost volume. Source copying
must not conceal wrong geometry, and added labels must have a zero-weight control.
Keep independent-noise/repeated-seed controls because training is not bit-repeatable.

Only then introduce predicted coarse-to-fine samples, retaining uniform coverage
for missed surfaces. Truth depth belongs in training targets, never runtime
observations or deployment query placement. Repeat incident-light supervision
and expand lighting/material diversity once structure survives multiple scenes.
Relighting, OLATverse ingestion and blade-volume integration are not prerequisites.

## Realtime: distinguish oracle headroom from learnable selection

A per-realization oracle gives a conditional bound, not additional observable
information. The projected-target experiment did not realize that headroom.
First test a construction-only selector fit and independent noisy realizations
of the same clean scene. Separate predictable candidate risk from noise-specific
oracle choices; do not supervise arbitrary nonunique weight vectors. Preserve
actual learned history in evaluation and never feed an oracle into recurrence.

Log signed linear error, candidate risk and weight use by lobe, scale and history
age. Check objective/parameterization saturation before adding capacity. Require
broad lighting, energy, detail and temporal gains together; the current learned
models still darken and blur relative to the deterministic candidate mixture.
Expand candidate support only where conditional oracle residuals justify it.

After construction works, train with real meshes and audit fresh scene families:
thin geometry, glossy reflections, moving lights, cuts and longer histories.
Beat SVGF on identical noisy inputs. ReSTIR+SVGF is a separate pipeline control;
DLSS Ray Reconstruction follows with an explicit quality/runtime envelope.

## Evidence and shared weights

Use construction data to debug, validation to select, fresh families to audit.
An inspected holdout is development data. Preserve all views/lighting variants
of one geometry in one split, exact capture hashes, full images and failure tails.
Reload fixed-budget checkpoints before scoring. LavaPipe measures quality and
correctness; production-device speed and memory are separate gates.

Compare independent training, field-pretrained initialization and joint training
only after useful standalone baselines. Keep supplied realtime geometry
authoritative. No single metric or visually smoother image promotes a model.
