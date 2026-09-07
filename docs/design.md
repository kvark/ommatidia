# Ommatidia: reconstruction contracts

Ommatidia is a portable research reconstructor using Blade and Meganeura on
Vulkan/Metal. The published checkpoint is a spatial 2× model. Recurrent 2×
reconstruction exists but is not yet a demonstrated DLSS replacement.

The active architecture review and experiment order are in
[reconstruction-review-2026-09-06.md](reconstruction-review-2026-09-06.md).
Historical diffusion and early spatial reasoning is retained in
[design-legacy.md](design-legacy.md), not treated as the current design.

## Current and experimental contracts

`ModelConfig` is the checkpoint's interpretation, not just training metadata.
Missing `fusion` selects `Legacy`; missing `backbone` selects `GroupNorm`.
Never relabel trained weights to opt into a different interpretation.

| Axis | Existing contract | New opt-in contract |
|---|---|---|
| Fusion | Compressed-space guide/history mixing | Linear radiance mixing |
| History storage | Compressed f16 | Linear demodulated f16 in the same texture |
| History interpolation | Compressed bilinear | Linear bilinear, then compress for features |
| Gate | Positive scalar odds predicted from LR conditioning | Signed affine coefficients applied to actual candidate disagreements |
| Backbone | Image-wide GroupNorm residual U-Net | Local residual U-Net, no spatial statistics, branch scale 0.1 |

`--fusion linear` isolates the radiance contract from the new gate.
`--fusion candidate` adds candidate-aware gates. Both require direct kernel
prediction, linear gather taps, `--guide-mix` and `--previous-output`.
`--backbone local` is an independent direct-regression experiment.

## Data flow

```text
sparse paths + G-buffer + accumulated LR evidence
    -> LR residual U-Net -> positive spatial kernels + gate coefficients
    -> actual spatial gather C + deterministic guide G + warped output H
    -> candidate disagreement features + validity
    -> guide/gather fusion -> valid previous-output fusion
    -> linear HDR history / exact current albedo remodulation / output
```

Gate features are `[1, mean((C-G)^2), valid*mean((C-H)^2),
valid*mean((G-H)^2)]`, with compressed RGB for numerical range. Their signed
coefficients are feature-major, then output-subpixel-major. The final image
averages decoded linear candidates, not compressed values. Features do not
provide per-lobe motion, learned recurrent feature state, or general attention.

The runtime still uses the existing pack/network/unpack path and history
textures. A wider output head has a cost; unchanged dispatch/buffer counts do
not imply unchanged frame time. The existing inverse-transform HDR ceiling is
retained and must be addressed explicitly in a future exposure contract.

## Training and evaluation

The new path receives gradients through physical image reconstruction. The
`confidence_target` utility constructs identifiable, detached linear-mixture
oracle labels; an auxiliary confidence-loss trainer is not implemented yet.
A label utility is not evidence that confidence supervision has been trained.

Causal rollout currently detaches prior predictions: it is state-distribution
training, not backpropagation through time. Compare both equal-frame and
equal-optimizer-update budgets, use longer sequences, and evaluate full-frame
causal runs on untouched scene families.

Retain legacy tests, three-way CPU/image-graph/WGSL parity, cut/disocclusion
coverage, fractional-history interpolation, nonzero-head crop invariance,
and a backward/optimizer smoke test before any quality experiment. New
checkpoint promotion additionally requires the quality and performance gates
in the review. No pretrained weights are changed by this architecture patch.
