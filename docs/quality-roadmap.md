# Quality roadmap

Two tracks: realtime denoising and posed-RGB fields. Shared implementation is not
proven weight transfer. Keep [README comparisons](../README.md) visible and
[historical results](results-overview.md) separate. No new checkpoint is promoted.

## Latest: numerical correctness before another architecture

[Stable-softplus study](results/stable-softplus-lavapipe-2026-09-09.md) closes the
frozen-fitting audit. The native mixture's missing multiplier floor could invent
black; Meganeura separately lost representable negative-tail values and gradients.
Both are fixed. All saved mixture weights now match an independent f64 reference
within 2.4e-7. Old reports remain evidence of their recorded runtime, not the fix.
Checkpoint schemas remain supported, but numerical fixes can change old outputs.

The seven repeated frozen fits and four own-history held-noise runs are complete.
Free logits realize about 99% of the attainable loss reduction, while the native
network ranges from 40% to 96% for RGB. The candidate optimum still loses detail.
Held-noise quality is essentially unchanged; no default or checkpoint is selected.

## Realtime: saturation and candidate support are separate problems

The reset-RGB fit drives 72% of legal multipliers to their floor and selects the
broadest scale with mean weight 0.923. First compare a centered, masked-softmax
parameterization against the corrected softplus control on the same frozen batch.
Preserve legal candidates and initial priors; measure saturation, gradient scale,
finite differences and loss relative to the actual attainable optimum. This is
not evidence for another auxiliary target or more backbone capacity.

Keep the direct-logit control. Expand candidate evidence only where its conditional
optimum remains inadequate. Then repeat independent held noise and full native
rollout, with every method owning its history. Require linear energy, broad-lighting,
detail and temporal gates together, not a compressed-PSNR increase alone.

After construction succeeds, train on real meshes and audit fresh families,
longer histories, cuts, thin geometry, moving lights and reflections. SVGF gets
identical noisy inputs; ReSTIR+SVGF remains a separate pipeline comparison.
Production-device runtime is measured separately from LavaPipe correctness.

## Field: stabilize geometry fitting

At 48 known surface points appearance reaches 42.67 dB. That is privileged point
fitting, not a full image or novel view. Source depth fits 88% of 12,288 bins;
volume termination is unstable even on 64 fixed rays. The corrected-runtime
repeat ends at 39% correct bins after earlier loss dips and late spikes. Do not
select an earlier minimum or mistake agreement between estimates for truth.

First stabilize fixed-ray volume optimization and record complete per-ray errors,
probability mass, feature magnitudes and gradient scales. Compare learning-rate
schedules and conditioning at equal observation budgets before changing field
representation. Preserve source-depth and known-surface appearance controls, with
truth positions explicitly excluded from deployment. Only then restore joint
RGB/geometry fitting, require sharp fitting cameras, test separate cameras, and
expand to fresh scene families. The current short fits do not bound capacity.

More correspondence/illumination heads, relighting, OLATverse and blade-volume
integration are deferred. The earlier mixed/negative controls remain documented.

## Evidence and shared weights

Construction data debug; validation selects; fresh families audit. Inspected
holdouts become development data. Keep a geometry's cameras and lighting variants
in one split; retain captures, hashes, full outputs and failures. Reload fixed-budget
weights before scoring. Numerical correctness and quality promotion are different
gates. Compare independent, pretrained and jointly trained weights only after
useful standalone baselines. Supplied realtime geometry remains authoritative.
