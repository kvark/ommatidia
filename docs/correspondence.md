# Correspondence and final-RGB construction

Both paths are opt-in. Runtime observations, production checkpoints and default
selection are unchanged. [Completed paired study and limitations](results/correspondence-lavapipe-2026-09-08.md).

## Field: `--view-fusion stereo-rgb`

For each source camera, a quarter-resolution lattice samples 16 distance
hypotheses through the acquisition bounds. Each hypothetical 3D point is projected
into every other source view. Bilinear learned-feature and compressed-RGB
mismatches, normalized depth and valid-pair support feed a shared small MLP. Its
scores correct the source termination logits before the existing visibility
aggregation. The matching MLP is shared across cameras and hypotheses.

This is actual cross-view correspondence, not another target-depth input. Camera
maps are cached using exact camera/bounds/extent keys; RGB and learned features
remain live, so changing source images changes the matching evidence. Out-of-view,
zero-parallax and invalid pairs contribute zero support. With no usable peer,
stereo cannot override the monocular result. The new output layer starts at zero,
so shared initialization and initial predictions match the VisibleRgb control.

Hypotheses are distance intervals along each camera ray, not frontoparallel planes.
Scores are computed at quarter resolution and nearest-upsampled. There is no 3D
cost-volume regularizer, learned occlusion test or hierarchical depth refinement.
A valid projection is not proof of visibility. Reflective/view-dependent surfaces
can disagree even at the correct depth. This compact model is not a reproduction
of [MVSNet, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Yao_Yao_MVSNet_Depth_Inference_ECCV_2018_paper.html).

`stereo-rgb` uses the same source termination and image/surface supervision as
`visible-rgb`. No extra geometric labels are added by the comparison. New weights
are needed for the matching head; existing modes retain their parameter schemas.
Density never sees the requested viewing direction. No blade-volume integration.

## Denoiser: `--rgb-loss --fixed-exposure-loss`

The spatial objective can now score the actual final linear image:

```
RGB = diffuse_illumination * observed_albedo + specular_radiance + observed_emission
```

This replaces the spatial lobe objective rather than adding another auxiliary
weight. Compression, physical and low-frequency coefficients apply in that
space; the temporal/confidence objectives retain their lobe contracts. Rendering,
network parameters and recurrent state do not change. Albedo/emission are ordinary
renderer observations, while reference RGB remains training-only. Fixed exposure
avoids target-dependent reweighting.

Repeated `--data` and `--eval-data` arguments allow multiple captures. Normal
training still rejects overlapping scene seeds/catalog families. Only explicit
`--construction-noise` permits the same scenes, after checking equal reference
pixels, observed surfaces, motion, jitter and provenance, and nonoverlapping
recorded path ranges that end before the reference range. This is held-noise
construction fitting, never a claim of unseen-scene generalization.

Each evaluated model runs its own causal history in this comparison. A construction
run also saves fitting-stream images/scores in `fitting/`; both fitting and held
evaluation use the serialized checkpoint. Do not confuse these end-to-end results
with the previous same-history candidate-risk diagnostic.

## Reproduce

`bash benchmarks/correspondence-lavapipe.sh capture` generates fresh matching field
and noise captures without requiring a previous checkpoint. Capture destinations
must be new. Existing, verified captures can be supplied explicitly instead.

`bash benchmarks/correspondence-lavapipe.sh ARM` accepts `field-visible`,
`field-stereo`, `selector-lobes` or `selector-rgb`. Set `CORRESPONDENCE_SEED` to 7
or 11. The field uses the corrected construction capture; selector arms use the
six recorded noise streams from the geometry study. Input paths and update budgets
can be overridden explicitly; exact commands, captures and checkpoint hashes are
saved. No existing output directory is overwritten.

The completed selector pair uses physical weight 1 and all other weights 0. It
isolates objective space, not loss-weight tuning or a production-ready balance.
All metrics retain their named evaluation spaces, including final RGB MSE, energy,
broad lighting, detail and temporal error. Software Vulkan tests establish
correctness and quality only. `benchmarks/report-correspondence.py ROOT` requires
all eight complete paired runs before producing a summary.
