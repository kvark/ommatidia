# Learned source support and projected-colour supervision

Both modes are opt-in; published checkpoints and the default runtime are unchanged.

## Field: `--view-fusion visible-rgb`

The shared RGB/camera-ray encoder predicts a 17-way distribution for every source pixel: 16 finite ray intervals through the declared acquisition bounds plus escape. It is a monocular, image-conditioned depth head; it is not a multiview cost volume. Bilinear projection samples the distributions and integrates a survival probability at each 3D query, with a fixed half-bin surface tolerance. Weighted first/second feature moments feed density and appearance; source-copy authority is also multiplied by absolute mean survival so a lone nearly occluded view cannot normalize itself back to full authority. Density and visibility never see the requested viewing direction.

`--visibility-weight` weights training-only source-camera termination labels. Hits outside the acquisition bounds and uncertified misses are masked. Labels never enter runtime observations, queries, or exported RGB contexts. Evaluation/recovery needs no labels for the forward pass. Width, positions, cameras and ray samples stay unchanged; this does not make the field physically relightable. The two representations of geometry (source depth distribution and implicit density) are not yet tied by a consistency loss.

The new head is zero-initialized without consuming RNG draws, preserving common parameter initialization. Moments and LateRgb keep their existing schema; VisibleRgb needs new weights. `build_points` appends per-source pixel-major probabilities after its original four outputs for this variant only.

## Selector: `--projected-weight`

For each actual training candidate/history state, project the target linear RGB lobe onto the convex closure of available candidates. Supervise the unique projected colour, not a nonunique set of mixture weights. The auxiliary MSE uses fixed exposure. Real RGB targets, reference temporal changes, confidence labels and runtime history stay unchanged. The teacher is detached and never enters recurrence. The coefficient-zero arm computes the same teacher and uses the same training graph. There are no additional learned parameters or inference inputs.

This tests whether an attainable target is easier to learn; it is not an unbiased-estimator guarantee. Per-observation squared-error minimization already has the same constrained optimum, so this is an alternative optimization/supervision experiment, not new scene information. Averaging over noisy inputs can change the statistical objective.

## Evaluation

Use `bash benchmarks/support-lavapipe.sh ARM` with `SUPPORT_SEED=7` or `11`. The recipe expects cache-corrected captures in `target/quality-control-data` (the existing quality-controls capture recipe produces them). It refuses to overwrite checkpoints/results and records commands, data hashes and source revision. Field arms: late-rgb, visible-rgb, visible-unsupervised. Selector arms: selector-control, selector-projected, selector-linear. The last doubles the ordinary physical loss instead of using projected colours, testing whether a result is just stronger linear weighting. Both use 512 updates by default. Full images, source-ablation checks and candidate-oracle diagnostics are retained.

Field image-ray and optimizer budgets match, but source-depth supervision adds all valid source pixels and the new head adds computation. This is not equal label exposure or equal FLOPs. The field is still a construction-scene experiment. The denoiser uses the established four development sequences; reuse makes them development data, not a fresh audit. Report each seed, all metrics and energy/detail/temporal failures. No LavaPipe speed claims.

Primary context: [NeuRay (Liu et al., CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Neural_Rays_for_Occlusion-Aware_Image-Based_Rendering_CVPR_2022_paper.html) uses predicted ray visibility to improve image-based field construction; [IBRNet (Wang et al., CVPR 2021)](https://ibrnet.github.io/) motivates retaining source observations until aggregation. This compact implementation is not a reproduction of either system.
