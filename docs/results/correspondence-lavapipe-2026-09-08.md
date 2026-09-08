# Correspondence and final-RGB construction

**Implemented and tested; neither experimental option is promoted.** Stereo
matching does not improve the field's held-camera quality. Final-RGB training
gives a tiny PSNR gain but worse held-noise linear error. Defaults and published
checkpoints are unchanged; no blade-volume integration.

## Execution and controls

All eight quality jobs completed in [34185179547](https://github.com/kvark/ommatidia/actions/runs/34185179547).
Model source is `0e34e0c`, Meganeura `43b606f`, published Blade graphics 0.9/render
0.6, Rust 1.92.0, LavaPipe/llvmpipe LLVM 20.1.2. Binaries use dev opt-level 1;
software-device time is not a production GPU measurement. Seeds 7 and 11 refer
to optimization, not new scenes. These are construction controls, not convergence
or unseen-scene generalization studies. Each fixed-budget checkpoint is reloaded
before evaluation; no held metric selects a checkpoint.

## Field: explicit cross-view depth hypotheses

`--view-fusion stereo-rgb` compares learned features and RGB across source views
at 16 hypothetical distances along each source ray. A shared residual matching
head corrects source termination logits. The quarter-resolution hypothesis map
is geometry-only; RGB/features remain live. Missing overlap or parallax cannot
override the monocular fallback. Ground-truth geometry stays training-only.
The head starts at zero, preserving common initialization and initial outputs.

One corrected construction capture: eight 64x64 centre-ray cameras at 256 reference
spp, three context views, four fitting cameras and one separate validation camera.
Both arms use core8/hidden32, 64 image rays x 64 stratified samples, 16 emission
probes, 1,024 updates (65,536 image rays), surface/visibility weights 0.05, and no
incident/consistency rays. Stereo adds 241 parameters (47,592 -> 47,833) and
correspondence computation, but no additional truth labels.

| Separate validation camera | PSNR x/(1+x) | Linear RGB MSE | Depth MAE, world units |
|---|---:|---:|---:|
| Visible RGB, seed 7 | 19.6871 | 0.735309 | 1.31682 |
| Stereo RGB, seed 7 | 19.6862 | 0.852689 | 1.40665 |
| Visible RGB, seed 11 | 19.6359 | 0.632771 | 1.51916 |
| Stereo RGB, seed 11 | 19.2037 | 0.684436 | 1.51582 |
| Visible RGB, mean | **19.6615** | **0.684040** | **1.41799** |
| Stereo RGB, mean | 19.4450 | 0.768562 | 1.46124 |

Mean held PSNR falls 0.2165 dB; linear RGB error rises 12.4% and depth error 3.0%.
Hit/miss accuracy is essentially unchanged (96.191% -> 96.179%). Fitting-camera
PSNR increases slightly (20.4208 -> 20.6569 dB), but inspected fitting and held
images remain very blurry, with incorrect boundaries and emitter streaks.
This compact matching head is not a successful multiview reconstruction result.
It is also not a test of every cost-volume architecture: there is no learned
occlusion rejection, 3D cost regularizer or hierarchical depth refinement.

## Native selector: final RGB versus illumination lobes

The native network is trained from scratch with its ordinary observed geometry,
variance, candidates and history. `--rgb-loss --fixed-exposure-loss` scores
`diffuse * observed_albedo + specular + observed_emission`. The control scores
six illumination-lobe channels. Physical weight is 1; compressed, low-frequency,
confidence and temporal weights are zero. No projected teacher or risk regression.

Two scenes, four frames each, six nonoverlapping recorded sparse-path ranges:
three fitting and three held-noise captures with equal references and guides.
Input is one path per 16x16 pixel; output 32x32, reference 256 spp. Channels8,
two-frame differentiable unroll, 512 updates, seeds7/11. Both models and the
independent deterministic baseline run their **own causal history**. All 24 fitting
and 24 held-noise frames per arm are scored, including six resets and 18 temporal
comparisons. This is explicitly same-scene held-noise construction fitting.

| Held-noise mean over two seeds | PSNR x/(1+x) | Linear RGB MSE | Energy ratio | Low-frequency PSNR | Temporal MSE |
|---|---:|---:|---:|---:|---:|
| Independent deterministic recurrence | 18.6327 | 0.636756 | 1.030414 | **25.4180** | 0.0114586 |
| Linear lobe objective | 18.8864 | **0.545772** | **0.992787** | 25.0350 | 0.0110653 |
| Final-RGB objective | **18.9328** | 0.649006 | 0.986174 | 25.0446 | **0.0107144** |

The 0.0464-dB PSNR gain does not compensate for **18.9% worse held RGB MSE** and
a larger energy deficit. MSE worsens in both seeds (0.505400 -> 0.705332 for7;
0.586144 -> 0.592680 for11). Temporal error falls 3.2%. Mean detail ratios are
1.1644/1.1591; noise can raise this diagnostic, so it is not a sharpness guarantee.
The inspected 32x32 images remain noisy, without a decisive visual winner.

Fitting RGB MSE improves only 1.8% (0.370985 -> 0.364259); fitting PSNR is only
19.2192 -> 19.4067 dB. A successful construction fit has not been established.
Changing loss space changes gradient/channel weighting; equal update counts and
learning rates do not prove convergence or rule out another optimizer schedule.

A separate CPU check on the captured references finds `D*A+S+E` agrees with stored
reference RGB at roughly 70 dB compressed PSNR: linear MSE 7.99e-7 and energy ratio
1.000012. A gross target-remodulation mismatch is not the explanation. This check
does not establish unbiased sampling or rule out every training/runtime defect.

## Verification and reproduction

`benchmarks/report-correspondence.py ROOT` verified all eight complete runs,
contiguous optimizer logs, finite checkpoint values, matching capture hashes,
source-only runtime contexts, configurations, initial field predictions, and all
paired reference/baseline images. Local negative tests reject missing reports,
partial frame counts, wrong loss/sampling settings and extra runtime truth inputs.
All native reports include fitting and held streams separately.

Normal [CI 34186210419](https://github.com/kvark/ommatidia/actions/runs/34186210419)
passed all five jobs at `91d7243`: cross-platform builds, formatting/Clippy/workspace
checks and C ABI, **28 Ommatidia LavaPipe GPU tests**, incident capture, cache-order
regression and bounded training/reload. New CLI tests reject duplicate noise,
ordinary same-scene split leakage and modified references posing as noise variants.
The final-RGB smoke reload reproduces weights and quality exactly. Stereo reload
is tested but its default evaluation sample count differs; bit-identical stereo
images are not claimed. The initial test-only serialization dependency error was
fixed using existing RON support. No tolerances were relaxed.

`bash benchmarks/correspondence-lavapipe.sh capture` creates fresh field and noise
captures without a prior checkpoint. Other arms are `field-visible`, `field-stereo`,
`selector-lobes`, `selector-rgb`; use `CORRESPONDENCE_SEED=7` or `11` and the recorded
budgets. Explicit paths permit verified existing captures. Overwrites are refused.
Temporary implementation workflows and payloads are removed; README images remain.
Final branch validation is recorded in PR19.

Next: diagnose a sharp fitting/convergence failure before adding another head.
Use a frozen real candidate batch for selector gradient/logit checks against its
conditional optimum, then held noise and native recurrence. For the field, separate
source-depth accuracy, implicit density and appearance fitting; supply explicit
surface likelihood only if those diagnostics establish a representation bottleneck.
No fresh-family victory, useful shared weights, SVGF win or DLSS parity is claimed.

[Architecture and commands](../correspondence.md) · [Quality roadmap](../quality-roadmap.md).
