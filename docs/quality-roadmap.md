# Quality roadmap

Two tracks: realtime denoising and posed-RGB fields. Shared implementation is not
proven weight transfer. Keep [README comparisons](../README.md) visible and
[historical results](results-overview.md) separate. No new checkpoint is promoted.

## Latest decision

[Geometry/held-noise study](results/geometry-noise-lavapipe-2026-09-08.md): source
and volume distributions agree better, but geometry still fails. The corrected
linear-risk diagnostic loses to the existing denoiser in causal evaluation.
The earlier noise report is superseded: it fit log-risk and used held availability
in a common candidate mask. The verified rerun fixes both and matches native RGB.

[Prior controls](results/support-lavapipe-2026-09-08.md) rejected projected-colour
supervision and stronger ordinary linear loss as improvements. Stop increasing
these auxiliary weights. Capture-cache and checkpoint-loading fixes are complete;
all future experiments must use corrected data and exact saved configurations.

## Field: correspondence before another loss

The 65,536-image-ray construction runs still lack sharp fitting images. Preserve
per-camera learning curves and compare declared ray exposure, capacity and samples
before inferring a representation ceiling. Agreement alone can make two wrong
geometry estimates consistent; report truth-depth accuracy and RGB separately.

Next test explicit per-view depth hypotheses/correspondence, with the current
VisibleRgb model as the control. Hold scene, ray budget and common initialization
fixed, retain source RGB detail, and inspect occlusion boundaries. Ground-truth
geometry stays a training target, never an inference input or sample-placement
shortcut. Require sharp fitting cameras first, separate cameras second, fresh
scene families third. Repeat seeds; same-seed training is not bit-reproducible.

Only then test predicted coarse-to-fine samples with uniform fallback, repeat
incident-light supervision, and expand scene/material/lighting diversity. Relighting,
OLATverse and blade-volume integration are deferred, not prerequisites.

## Realtime: fit the actual output before scaling

A conditional lobe oracle is neither a deployable selector nor an RGB-PSNR bound.
The simple cross-noise risk model does not establish a limit on neural selection.
Test a native-selector construction fit with full available geometry, variance
and history inputs. Compare linear remodulated RGB with illumination-lobe losses;
validate energy, broad lighting, detail and temporal errors together. Separate
fitting noise from held noise and keep each method's actual causal history when
claiming an end-to-end improvement. Same-state diagnostics remain labeled as such.

Log candidate risk, signed RGB error and selection saturation by lobe, scale and
history age. Expand candidate support only where oracle residuals justify it.
After construction works, train on real meshes and audit fresh scene families,
long histories, cuts, moving lights, thin geometry and glossy reflections.
Compare SVGF on identical noisy inputs; ReSTIR+SVGF is a separate pipeline arm.

## Evidence and shared weights

Construction data debug; validation selects; fresh families audit. Inspected
holdouts become development data. Keep each geometry's cameras and lighting
variants in one split. Save captures, hashes, all outputs and failures; reload
fixed-budget weights before scoring. LavaPipe establishes quality/correctness,
not production speed. Defaults require joint quality gates, not one better metric.

Compare independent training, field-pretrained initialization and joint weights
only after useful standalone baselines. Supplied realtime geometry stays authoritative.
