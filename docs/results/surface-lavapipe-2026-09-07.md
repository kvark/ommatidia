# Surface-supervised field: 64-pixel construction test

**Geometry supervision helps; reconstruction remains blurry.** This is one
construction scene, not an unseen-scene generalization or checkpoint promotion.

Run `FIELD_UPDATES=2048 bash benchmarks/surface-lavapipe.sh`.
[Completed LavaPipe run 34146682487](https://github.com/kvark/ommatidia/actions/runs/34146682487)
used code `6170eab` and recipe commit `0f9daad`; the `surface-lavapipe-long`
artifact retains captures, targets, both checkpoints, diagnostics and hashes.
Rust 1.92, Meganeura `d903bba`, published Blade 0.9/render 0.6 and Naga 30.
No timings are interpreted as production performance.

## Paired protocol

One procedural geometry (seed 20260907), CLI lighting seed 43, eight 64x64
centre-ray views at 256 reference spp. Source cameras 0/2/4, fitting 1/3/5/6,
validation 7. Both arms: core 8, hidden 32, 32 camera rays x 32 midpoint samples,
16 emission probes, 25% visible-emitter pixel proposals, optimization seed 7,
2048 updates. This is 65536 sampled fitting rays, about four nominal passes
before repeated and emitter-biased sampling. The larger default was not trained.

Only the termination-loss weight changes, 0 versus 0.05. Graph, initialization,
query coordinates, sampling strata and update budget match. Neither arm receives
target depth as an observation or uses it to place samples. Save/reload precedes
evaluation; no validation result selects an update. Separate fitting-view
numbers below describe view 1, not an average over the four fitting views.

## Measured results

| Metric | Weight 0 | Weight 0.05 |
|---|---:|---:|
| Fitting-view PSNR | 19.8174 dB | **21.3862 dB** |
| Validation-view PSNR | 17.1494 dB | **18.0843 dB** |
| Validation linear RGB MSE | 3.68629 | **1.62973** |
| Validation log1p RGB MSE | 0.144644 | **0.077881** |
| Fitting depth MAE, world units | 3.66421 | **0.87715** |
| Validation depth MAE, world units | 2.60089 | **1.32963** |
| Validation hit/miss accuracy | 74.46% | **96.48%** |
| Validation termination NLL | 4.20126 | **1.77921** |
| Validation early termination mass | 0.68581 | **0.32751** |
| Validation mass at visible emitter surface | 0.08647 | **0.51964** |

PSNR uses x/(1+x), not linear HDR PSNR. Depth MAE compares the conditional
expected termination distance on ground-truth hit rays; opacity/NLL accompany
it because expected depth alone can hide a diffuse or transparent volume. All
4096 rays in each scored view have valid labels; no exclusions. The acquisition
cube radius is 12 world units. Ground-truth triangle surfaces do not have an
arbitrary target volume density.

Validation camera, **64 native pixels enlarged**, same display transform:

| RGB/source only | + surface termination | Reference |
|---|---|---|
| <img src="../surface-preview/control.png" alt="Construction field without surface targets" width="256" height="256"> | <img src="../surface-preview/surface.png" alt="Same field with surface termination loss" width="256" height="256"> | <img src="../surface-preview/reference.png" alt="Held camera of the same construction scene" width="256" height="256"> |

Linear image error falls 55.8% and depth error 48.9% on this validation view,
but the images still lack clear object boundaries and material detail. The
surface-supervised field scores 15.7711 dB with zero RGB and 17.3415 dB when RGB
images are reassigned to unchanged camera poses, versus 18.0843 dB normally.
There is image dependence, not proof of correct correspondence or transfer.
A single scene can still be memorized through position and camera features.

## Post-fit diagnosis, not another learned result

An independent CPU PyTorch replay matched the four saved LavaPipe image outputs
within one 8-bit value (0--2 differing components out of 12288). No optimization
was performed. Increasing evaluation samples alone from 32 to 128 gave:

| Arm/view | Normal 32-sample PSNR | Normal 128-sample PSNR | True-surface oracle PSNR |
|---|---:|---:|---:|
| Control / fitting | 19.8174 | 19.6025 | 13.2110 |
| Control / validation | 17.1494 | 17.2947 | 12.5348 |
| Surface / fitting | 21.3862 | 20.7030 | 20.0204 |
| Surface / validation | 18.0843 | 17.9953 | 19.0689 |

The **nondeployable oracle** evaluates learned radiance at true first-hit
positions and uses true misses. It exposes remaining appearance/correspondence
error even with exact support; it is not an upper bound on a co-adapted field's
RGB score. More evaluation samples alone do not consistently improve the fixed
checkpoint. These post-fit observations are construction-only: do not mistake
them for new GPU quality measurements or tune an untouched audit on them.

## Decision

Retain opt-in surface supervision and the target-only contract. Next compare
capacity and sampling exposure separately, using stratified/finer training rays
before judging denser inference. Diagnose per-view correspondence/visibility
rather than add another independent lighting head. A sharp fitting result and
multiple independent scenes/seeds remain gates, not completed milestones.

[Target contract](../surface.md) · [Roadmap](../quality-roadmap.md).
Earlier 16-pixel incident images remain archived: [control](../field-preview/control.png),
[incident](../field-preview/incident.png), [reference](../field-preview/reference.png).
