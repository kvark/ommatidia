# Quality roadmap

Two tracks: realtime denoising and posed-RGB fields. Shared implementation is not
proven weight transfer. Keep [README comparisons](../README.md) visible and
[historical results](results-overview.md) separate. No new checkpoint is promoted.

## Latest finding: correct normalization before more architecture

[Frozen fitting](results/frozen-fitting-lavapipe-2026-09-08.md) found that the native
selector could lose all candidate mass and invent black at negative logits.
The GPU graph now matches the scalar positive-multiplier floor and denominator
clamp. This changes affected experimental transport outputs, not published legacy
Upscaler behavior. The regression reproduces the old failure and checks the fix.
Historical scores remain tied to their recorded runtime.

With the guard, free logits close about 99% of the frozen candidate improvement
gap. The native network closes 84%/96% for RGB on reset/history frames. This is
fitting, not generalization; even the candidate optimum is not the reference.
Do not interpret a violated candidate constraint as superior reconstruction.

## Realtime: corrected native recurrence, then conditioning

Repeat held-noise and full causal evaluation with corrected normalization before
claiming a quality gain. Every method must own its recurrent state. Keep energy,
broad lighting, detail and temporal gates together, not just compressed PSNR.

Use frozen logits/network controls to measure saturation, feature magnitudes and
remaining attainable loss. The native network still reaches extreme logits;
compare stabilized parameterization/scales only against the corrected control.
Expand candidate evidence only where its conditional optimum remains inadequate.
Do not repeat rejected auxiliary-loss sweeps without a changed diagnosis.

After construction works, train on real meshes and audit fresh families, longer
histories, cuts, moving lights, thin geometry and reflections. SVGF gets identical
noisy inputs; ReSTIR+SVGF is a separate pipeline comparison. DLSS RR follows with
explicit quality and production-GPU runtime limits.

## Field: improve geometry fitting before another head

Isolated known-surface appearance fits 48 colours to 42.5 dB, but this privileged
point test is not a sharp full-frame reconstruction. Source depth reaches 88%
correct bins on 12,288 fitted pixels; volume termination reaches 70% on 64 rays.
Its loss still fluctuates and gradients are connected. Source feature magnitudes
and finite-difference curvature also flag conditioning to investigate.

First improve fixed-ray volume fitting and track per-ray errors, mass and gradient
scales. Then restore joint RGB/geometry training at declared adequate ray budgets.
Compare schedules/stabilization with equal observations and keep isolated controls;
more parameters or supervision must earn a measured improvement. Truth positions
belong only to the explicitly privileged diagnostic; deployment queries remain
observation-selected. Require sharp fitting cameras, then separate cameras, then
fresh scene families. No single-scene probe establishes cross-scene capacity.

The prior stereo/visibility/consistency and light-supervision results remain
negative or mixed. Do not add another correspondence or illumination head before
resolving geometry optimization. Relighting, OLATverse and blade-volume integration
are deferred, not prerequisites.

## Evidence and shared weights

Construction data debug; validation selects; fresh families audit. Inspected
holdouts become development data. Keep a geometry's cameras and lighting variants
in one split; retain captures, hashes, complete outputs and failures. Reload fixed-
budget weights before scoring. LavaPipe establishes correctness/quality, not speed.

Compare independent training, field-pretrained initialization and joint weights
only after useful standalone baselines. Supplied realtime geometry stays authoritative.
