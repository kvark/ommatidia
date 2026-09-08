# Quality roadmap

Two tracks: realtime denoising and posed-RGB fields. Shared implementation is not
successful weight transfer. Neither new path has a promoted checkpoint. Keep
[README comparisons](../README.md) visible, measured failures explicit, and
[historical results](results-overview.md) separate from current experiments.

## Latest evidence

[Late fusion, recovered wider field, and selector oracle](results/late-fusion-lavapipe-2026-09-08.md):

- The wider saved field now loads without retraining after the upstream Winograd
  checkpoint fix; its separate-camera PSNR is 20.98 dB on one construction scene.
- Late source RGB at 512 updates raises PSNR 17.55 -> 18.51 dB, but linear RGB
  error rises 32% and geometry worsens. It stays opt-in, not a quality winner.
- The fixed-history denoiser oracle reduces diffuse/specular lobe error 32%/29%
  versus learned selection. This is conditional headroom, not deployable output.

Surface supervision, stratified sampling, independent denoiser loss controls,
checkpoint recovery and target-only diagnostics are implemented. Procedural
texture identities now depend on content; warm/cold-cache equality is tested.
Old texture-contaminated captures cannot be repaired by changing metadata.
Use corrected capture bytes for future comparisons; preserve older reports as
historical evidence, not proof of capture-order-independent reproduction.

## Realtime priority: learn the available selector headroom

First keep candidates fixed and diagnose selector optimization. Compare direct
reconstruction supervision against candidate-aware targets that respect
nonunique mixtures and uncertainty. Oracle labels are noisy per-example optima;
they are not guaranteed predictable from the model's observations. Do not force
one arbitrary weight vector when several produce the same radiance.

Log error and selection by lobe, scale, coverage and history age. Isolate whether
confidence targets conflict with reconstruction, and monitor scene-linear signed
bias and energy alongside detail and temporal error. Any controlled change must
beat the same-state and full-trajectory deterministic comparisons without buying
smoothness through darkening or broad blur. No oracle enters the evaluated state.

Expand actual spatial/temporal support only where oracle residuals show it is
needed. Include real meshes, thin geometry, glossy reflections, moving lights,
cuts and longer sequences. Train/selector/audit splits are scene-family-disjoint.
Then beat SVGF on identical noisy inputs; ReSTIR+SVGF is a separate pipeline
control. DLSS Ray Reconstruction is a later matched-input quality/runtime gate.

## Field priority: trustworthy multiview support and a sharp fit

Use the recovered wider baseline and the late-RGB variant at matched ray exposure
and update budgets; vary width, samples and duration independently. Start with
64-pixel fitting/validation cameras, then 128. Preserve full-resolution source
colour, but resolve which sources actually observe a queried surface. A frustum
mask and learned appearance score are not calibrated occlusion visibility.

Geometry still pools source features early. Test per-view correspondence across
depth hypotheses or density-informed source visibility before another generic
lighting head. Keep fitting/held RGB, termination/depth, source permutation and
true-surface diagnostics separate. Source copying must not hide wrong geometry.

Compare uniform/stratified sampling with coarse-to-fine proposals driven by
predicted density, retaining uniform coverage for missed surfaces. Ground-truth
depth stays in losses, never observations or query placement. A sharp fitting
result and recognizable separate-camera boundaries remain unpassed gates.

Once structure works on repeated independent scenes/seeds, repeat incident-light
supervision and expand illumination/material diversity. A captured-appearance
field is not automatically a relightable transport model. OLATverse ingestion and
blade-volume integration are not prerequisites for these quality decisions.

## Shared weights and evidence

Compare independent training, field-pretrained initialization and joint training
only after useful standalone baselines. Keep supplied realtime geometry
authoritative and all views/lighting variants of one geometry in one split.
Retain shared weights only if both tasks improve on fresh audits.

Use construction data to debug, validation to select, and fresh family holdouts
to audit. An inspected holdout becomes development data. Name image transforms,
measure reference noise, publish full images/motion and per-scene tails, and
reload saved weights before scoring. LavaPipe measures correctness and quality;
production-GPU speed/memory are separate gates. No metric alone promotes a model.
