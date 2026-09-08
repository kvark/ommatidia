# Source visibility and projected-selector controls

**No promotion.** Predicted source visibility improves field RGB in both seeds,
but worsens depth. Projected-colour supervision does not improve the denoiser.
Defaults, published weights and blade-volume are unchanged.

## Reproduction

[Main study 34174073553](https://github.com/kvark/ommatidia/actions/runs/34174073553)
completed all nine quality jobs. Model source is `3d961fb` (study head `3e95b47`
adds only its control workflow), merged Meganeura `43b606f`, Rust 1.92.0,
published Blade graphics 0.9/render 0.6, and LavaPipe/llvmpipe LLVM 20.1.2.
No software-device timing is interpreted as production GPU performance.

The stronger ordinary-linear arms in
[34174073455](https://github.com/kvark/ommatidia/actions/runs/34174073455)
completed all 512 updates and saved weights, but their 30-minute jobs ended
during evaluation. Both were successfully scored in
[34175903249](https://github.com/kvark/ommatidia/actions/runs/34175903249), using
`--eval-only` and the same original experiment binary. Recovery verifies unchanged
weight bytes; no retraining, partial-frame scoring or checkpoint selection.

`benchmarks/support-lavapipe.sh ARM` records commands and hashes and refuses to
overwrite an existing result. `benchmarks/report-support.py ROOT` requires all
11 completed reports, checks paired RGB contexts/references and deterministic
baselines, and writes per-seed scores and arithmetic means. Complete evidence
includes all comparison PNGs, loss curves, checkpoints, reference metadata,
candidate-oracle reports and shared capture hashes. The corrected capture files
are from run 34153941146, artifact `corrected-control-data`; do not substitute
older texture-cache-contaminated captures.

## Field: absolute visibility rather than normalized copying

The new `--view-fusion visible-rgb` predicts 16 source-ray termination bins plus
escape from each RGB/camera view. Their survival probabilities gate geometry
features and late RGB copying; absolute mean survival also caps copying strength.
Source termination labels are training-only. This is a monocular visibility head,
not a multiview cost volume. Implicit density and source-depth distributions are
not yet constrained to agree.

One construction scene, eight 64x64 centre-ray views, 256 reference spp; three
context views, four fitting cameras and one separate validation camera. Core8,
hidden32, 32 image rays x 64 stratified samples, 16 emission probes, surface
weight 0.05, emitter-pixel fraction 0.25, 512 updates, seeds 7 and 11. Each arm
sees 16,384 image rays (roughly one average pass over the fitting pixels before
emitter bias). This is a short construction diagnostic, not convergence or
unseen-scene generalization.

Common parameter initialization is GPU-tested identical. VisibleRgb adds only
153 parameters (47,439 -> 47,592), but its depth loss also supervises all 12,288
valid source pixels per update. Image-ray/update budgets match; label exposure
and computation do not. No incident loss is used.

| Separate validation camera | PSNR x/(1+x) | Linear RGB MSE | Depth MAE, world units | Hit/miss accuracy |
|---|---:|---:|---:|---:|
| Late RGB, seed 7 | 17.8992 | 1.39629 | 1.83685 | 92.09% |
| Visible RGB, seed 7 | 19.1612 | 0.89462 | 1.98381 | 95.73% |
| Late RGB, seed 11 | 15.4326 | 1.98390 | 1.84237 | 93.12% |
| Visible RGB, seed 11 | 17.0064 | 1.58185 | 1.98809 | 96.09% |
| Late RGB, mean | 16.6659 | 1.69010 | 1.83961 | 92.60% |
| Visible RGB, mean | 18.0838 | 1.23824 | 1.98595 | 95.91% |

Mean PSNR rises 1.4179 dB and linear error falls 26.74%, but depth error rises
7.95%. Mean termination NLL also slightly worsens (2.6761 -> 2.7080).
The inspected images remain very blurry and do not recover correct object
boundaries. An RGB gain is not a geometry success.

A same-architecture **zero visibility-loss** arm at seed 7 scores 18.2521 dB,
linear MSE 0.89918, depth MAE 2.03329 and hit/miss accuracy 93.43%. Thus most of
that seed's linear RGB gain already exists without source-depth supervision;
adding the labels improves PSNR and occupancy, but only slightly changes linear
error (0.89918 -> 0.89462). The extra labels cannot receive credit for the entire
architecture improvement. This control was run for only one seed.

Training is not bit-reproducible: the new LateRgb seed-7 control has the same
untrained output and first loss as the earlier 512-update run, but later losses
diverge and final PSNR differs by 0.61 dB. Do not treat a small single-seed gain
as decisive. The exact cause of that divergence has not been isolated.

## Denoiser: projected RGB is not additional scene information

For actual available candidates and the current learned history, the teacher
projects reference RGB onto their convex closure. The auxiliary loss supervises
that unique colour, not nonunique mixture weights. It is detached and never
feeds recurrence. There are no new inference parameters. The zero-weight arm
still computes the teacher and uses the same graph. Ordinary per-example linear
MSE already has the same constrained optimum; this is an optimization/statistical
objective experiment, not a guaranteed route to the oracle result.

Eight fitting scenes x eight frames; four existing development sequences x
16 frames. Channels8, unroll2, fixed exposure, confidence loss disabled, 512
updates and seeds7/11. All other losses and sampled sequences match. The
projected arm adds weight0.1; the ordinary-linear control instead raises the
physical loss from0.1 to0.2. No audit set is used for selection.

| Mean over seeds; all 64 frames per seed | PSNR | Low-frequency PSNR | Linear energy ratio | Detail ratio | Temporal MSE |
|---|---:|---:|---:|---:|---:|
| Deterministic recurrent baseline | 22.2493 | 29.0211 | 0.997665 | 1.02295 | 0.00360159 |
| Learned control | 22.3347 | 26.9019 | 0.955861 | 0.94480 | 0.00261632 |
| + projected-colour loss | 22.1283 | 26.4726 | 0.952842 | 0.94822 | 0.00267866 |
| Stronger ordinary linear loss | 22.1839 | 26.5084 | 0.951859 | 0.94902 | 0.00270526 |

PSNR uses x/(1+x); low-frequency PSNR averages 8x8 output blocks in that space.
Energy is scene-linear; ideal energy/detail ratios are1. Detail is a diagnostic,
not a perceptual guarantee. Temporal errors use reference changes and accepted
reprojection; the 60 temporal comparisons exclude the four reset frames.

| Seed | Control PSNR | Projected PSNR | Stronger linear PSNR |
|---|---:|---:|---:|
| 7 | 22.4447 | 22.1425 | 22.2442 |
| 11 | 22.2246 | 22.1140 | 22.1236 |

Both changes worsen PSNR, broad-lighting PSNR and energy in both seeds. The
projected loss raises mean diffuse lobe MSE 0.30004 -> 0.31177, while slightly
reducing specular MSE 0.02757 -> 0.02721. More ordinary linear weight reduces the
reported relative MSE slightly but does not fix the energy loss. Neither is
selected. The learned control still sacrifices energy/broad fidelity versus the
deterministic baseline despite lower temporal error. Full reference and baseline
PNG sets match across all arms; every final report includes all64frames.

These are development controls, not four newly independent scene families or
proof that all oracle-related supervision must fail. Conditional oracle headroom
is not proof that its exact per-noise-realization choices are predictable.

## Validation and decision

The suite includes 25 Ommatidia GPU tests plus physical incident-capture tests,
source-label isolation, normalized source probabilities, shared initialization,
occlusion suppressing copying, view permutation, direction-independent density,
visibility gradients/reload, coefficient-zero parity, projected-only learning,
and unchanged runtime parameter schema. Normal CI also covers capture-cache
independence, moving geometry, original field/surface training, and VisibleRgb
training/reload. Final status is recorded in PR19; no tolerances are relaxed.

Next: establish a sharper field fit at a declared adequate ray budget, then test
source-depth/implicit-density consistency and explicit multiview correspondence.
For the denoiser, stop escalating these auxiliary weights; isolate achievable
selector fitting with independent noise realizations and measure the stability
of candidate risk/selection before expanding architecture. Preserve energy,
spatial detail and temporal gates. No new default, shared-weight result,
blade-volume integration, SVGF win or DLSS parity is claimed.

[Input and loss contracts](../support.md) · [Roadmap](../quality-roadmap.md).
