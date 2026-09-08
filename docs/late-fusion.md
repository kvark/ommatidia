# Late source-view appearance and candidate diagnostics

Two bounded experiments, not a new default or a blade-volume integration.

## Field

`--view-fusion late-rgb` preserves individual projected source features until a
shared scoring MLP assigns appearance weights. It receives per-view features,
feature mean/variance, the geometry latent, bounded source colour and source/query
ray directions. It blends the original scene-linear RGB, not compressed RGB.
A learned gate retains the existing radiance decoder as a fallback. The existing
emission field is added once: source RGB is converted to a nonnegative scattered
candidate by subtracting predicted emission before blending.

Density and emission still use the moment-pooled geometry branch and remain
independent of query viewing direction. This is late **appearance** fusion, not
an occlusion-aware geometry solver. A projected view can be occluded; frustum
validity is not visibility. The new head can learn rejection but has no guarantee
of calibrated visibility, material separation or physically correct relighting.
No target depth, light metadata or oracle enters inference observations.

Missing source coverage closes the source gate exactly. New parameters are tied
across views and appended after historical parameters to preserve common-weight
initialization in paired runs. `moments` remains the serialized/default mode;
`late-rgb` needs new weights. Compare data, sampled camera pixels, quadrature
stream and update count, while reporting the extra parameters.

`field --eval-checkpoint PATH` reads the sibling `model.field.json` and evaluates
saved weights without optimization. Put outputs in a separate directory. This
recovers completed training runs whose earlier inference compilation failed;
it does not select or retrain a checkpoint. Evaluation sampling can be changed
explicitly and must be recorded.

## Native denoiser

`transport --eval-only --candidate-oracle` reads the actual GPU-prepared candidates,
priors and selected weights after each normal native frame. Each lobe's linear RGB
target is projected onto the convex closure of its available candidates. A
zero-prior candidate, including rejected history, is excluded. The solver uses
float64 affine-face projections (up to four vertices in RGB), enumerates lower-
dimensional faces for degeneracies, and reports a first-order dual-gap certificate.

This measures the conditional minimum **linear lobe error**, with the learned
model's history fixed. It is neither a bound on final compressed RGB PSNR nor a
bound for a different recurrent trajectory. Oracle images are diagnostics only;
they never enter next-frame history, checkpoint selection or training. Weights
can be nonunique, so the selected RGB point/error matters more than one oracle
weight vector. An oracle gap is possible headroom, not proof it can all be learned
from the available noisy observations.

The report includes same-state fixed/learned/oracle error, per-candidate errors
with availability counts, signed lobe RGB bias and mean candidate shares. A large
selector gap motivates learning better selection; a poor conditional oracle
motivates different candidates/support. Both problems can occur together.

## Runtime dependency

[Meganeura PR #160](https://github.com/kvark/meganeura/pull/160) fixes duplicate
Winograd caches for tied convolutions and keeps declared caches out of the logical
checkpoint schema. Original weights remain strict and caches are regenerated at
execution; other derived parameters are not blanket-filtered. No checkpoint
rewriting or disabling validation. CI pins its tested revision `3d424e0`.

## Reproduce

`benchmarks/late-fusion-lavapipe.sh` provides `recover-wide`, `moments`, `late-rgb`
and `candidate-oracle` modes with explicit data/checkpoint paths. These are
construction/development controls. LavaPipe is used for quality and correctness,
not hardware speed claims. Artifacts record commands, input hashes and scores.
