# Goal: DLSS 4 Ray Reconstruction-level quality

**Active goal; not achieved.** Reconstruct noisy, jittered path-traced frames with
DLSS 4 RR-level detail, temporal stability and responsiveness. This covers
denoising, reconstruction and antialiasing, not frame generation. Passing CI or
winning one metric does not complete the goal.

Denoising is primary. Preserve the offline field experiment, but pause separate
field/relighting work unless it directly tests transfer to denoising. Keep the
Rust/Meganeura/Blade path portable. Quality comes before latency: a larger model
is allowed; LavaPipe speed is not a quality gate.

## Completion contract

First declared envelope: **960x540 input to 1920x1080 output**, 2x per axis,
with 1 path/pixel primary and 2/4-path conditions reported separately. This is an
engineering target, not existing benchmark evidence or parity across every game.
Tiny construction renders remain diagnostics, not full-resolution substitutes.

Pin the actual DLSS 4 transformer RR version: SDK/library hash, model/preset,
settings, driver, device and overrides. A latest SDK or automatic preset alone
cannot identify the executed network. Missing DLSS results mean **not measured**.

Replay identical noisy radiance, sample phases, jitter, motion, exposure and
resets. Validate coordinate/radiometric conversions and record optional guides.
Ommatidia's high-resolution surface guides require a separate signal-advantaged
arm unless the comparator receives equivalent observations. Each method owns
its causal history; no future/reference frames or method-specific tone mapping
and sharpening. Compare converged path-traced references and matched-input SVGF;
ReSTIR+SVGF is a separate sampling-plus-denoising pipeline.

Archive actual DLSS outputs from supported hardware for comparison with LavaPipe
outputs. Do not substitute marketing screenshots. Success requires:

- Comparable geometry, texture, reflection, shadow and broad-lighting detail,
  checked in linear HDR and a common display transform.
- Comparable stability **and response** to disocclusions, cuts, moving lights/
  reflections and exposure changes. Measure delay separately from warped temporal
  error; blur or lag must not masquerade as stability.
- No systematic brightness loss, severe failing subsets or visible preference for
  DLSS in blinded normal-speed playback. Inspect startup, thin/glossy geometry,
  dark scenes and small emitters, not merely averages or static crops.

Before the sealed audit, freeze scenes, clip lengths, metrics, numerical
non-inferiority margins and visual-review protocol using development data.
Require multiple families and training seeds; report paired uncertainty by
sequence/family, not correlated pixels. A nonsignificant difference alone is not
equivalence. Publish every sequence and the worst cases. Actual DLSS comparator
capture and the final audit have **not** been performed.

## Next: optimization and comparator, in parallel

The [numerical audit](results/stable-softplus-lavapipe-2026-09-09.md) fixed lost
mixture mass and softplus-tail arithmetic. Use the corrected pinned runtime.
Reset-frame selection still saturates, and even optimal candidates lose detail.

Compare centered, prior-aware masked softmax with corrected softplus on identical
frozen batches. Preserve legal candidates, initial priors and common weights;
measure gradients, saturation and attainable-loss gaps. Retain direct-logit
controls, then test each model's own recurrence on held noise. Softmax is a
hypothesis, not a presumed fix or permission to reinterpret old checkpoints.

In parallel, build the recorded-input DLSS/SVGF adapter and reference-convergence
checks. Comparator infrastructure must not remain an indefinitely deferred last
step. Use construction clips to validate integration; keep final-audit families
unseen until the protocol is frozen.

## Raise the reconstruction ceiling

Separate selector failure from missing evidence using the candidate bound. Do
not restrict the quality-first model to already blurred candidates when perfect
selection cannot recover detail. Preserve sparse radiance, sample phase and
surface guides beside deterministic fallback estimates. Test multiscale temporal
features, lobe-aware recurrent state and a decoder capable of sharp fitting
images. Learned residual reconstruction or local temporal attention must beat the
corrected control, not just resemble a paper or add parameters.

Train on real meshes/materials early, with camera/object/light trajectories,
independent noise, matched transport and converged HDR targets. Increase capacity,
resolution and ray exposure independently. Track learning curves and sharp
fitting images before interpreting generalization. Recheck reference gradients
when a tiny construction task fails. No new auxiliary-loss sweep without a
specific diagnosis and a predefined decision rule.

## Win sequences, then optimize deployment

Beat matched-input deterministic/SVGF controls on spatial, energy, detail and
temporal gates together. Run the pinned DLSS comparison at the declared envelope
and complete the sealed audit. Expand that envelope before broader claims.
Distill or optimize the quality winner only afterward; production latency/memory
remain separate measurements. Do not preserve a small model at the expense of
the quality target.

Each experiment declares its failure mode, control, budget and decision rule.
Keep concise negative results, retire redundant experimental paths after explicit
compatibility decisions, and stop accumulating heads/flags without measured value.
Keep [README comparisons](../README.md) visible, capture hashes and full outputs
retained, and truth out of inference. Main and published weights need explicit
review/promotion. Field integration and shared-weight claims stay secondary.

## Primary comparator references

[NVIDIA's DLSS 4 report](https://research.nvidia.com/labs/adlr/DLSS4/) describes
transformer RR and stability versus responsiveness.
[The Vulkan RR sample](https://github.com/nvpro-samples/vk_denoise_dlssrr) documents
input buffers and provides a reference for the optional comparison adapter.
Neither establishes Ommatidia's quality or a reproduced model.
