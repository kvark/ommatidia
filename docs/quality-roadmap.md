# Quality roadmap

Two tracks: realtime denoising and posed-RGB fields. Keep [README comparisons](../README.md)
visible and [historical results](results-overview.md) distinct. No new checkpoint
is promoted; shared implementation has not demonstrated weight transfer.

## Latest decision

[Correspondence/RGB construction study](results/correspondence-lavapipe-2026-09-08.md):
stereo slightly helps fitting-camera PSNR but worsens held colour/depth. Final-RGB
selector training slightly helps fitting loss but worsens held-noise linear error.
Both use actual observations and pass GPU contracts. Neither establishes sharp
fitting or a production improvement. Stop adding heads/losses before diagnosing
that fitting failure.

## Immediate gate: construction convergence

For the selector, freeze one real candidate/history batch and fit the native
network against final RGB. Measure gradient norms, output-logit/prior saturation,
learning curves and distance to an attainable conditional optimum. Change learning
rate, training exposure and capacity independently. A per-batch oracle is a
construction diagnostic, not additional inference information or a global recurrent
bound. Then repeat on held noise, with each method's own native history.

For the field, log all fitting cameras rather than relying on one aggregate or
one held image. Separate source-depth accuracy, volume termination and appearance
error; two geometry estimates agreeing does not make them correct. Check fitting
at declared ray exposure and capacity, first 64 then 128 pixels, with repeat seeds.
The new compact stereo head is not a full cost-volume network.

If representation remains limiting, test explicit surface-likelihood/free-space
evidence in the density decoder rather than only visibility-weighted feature
averages. Then consider spatial/depth regularization and predicted coarse-to-fine
sampling with uniform fallback. These are hypotheses, not established diagnoses.
Truth geometry stays training-only; never position deployment samples with it.

## After fitting works

Audit fresh scene families and longer native histories: real meshes, thin geometry,
glossy reflections, moving lights and cuts. Require energy, broad-lighting, detail
and temporal quality together. Compare SVGF on identical noisy inputs; ReSTIR+SVGF
is a separate pipeline control. LavaPipe measures correctness and quality; production
GPU time and memory require separate measurements.

Keep each geometry's cameras and lighting variants in one split. Construction data
debug, validation selects, fresh families audit; inspected holdouts become development
data. Save hashes, all images, failure tails and fixed-budget reloadable checkpoints.
Only after useful standalone baselines compare independent, pretrained and joint
weights. Supplied realtime geometry remains authoritative. OLATverse, relighting
and blade-volume integration are deferred, not prerequisites.
