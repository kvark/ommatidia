# Stable softplus: correct weights, unchanged quality verdict

**Two numerical defects are fixed; no checkpoint is promoted.** This report
supersedes the corrected-runtime interpretation of the earlier
[frozen fits](frozen-fitting-lavapipe-2026-09-08.md), whose arrays still describe
Meganeura `43b606f`. Their unit weight sum did not establish correct weights.
Main and published assets are unchanged. Existing checkpoint schemas remain
supported, but corrected numerical operations can change their output/training.

## Defects and tests

The first defect was Ommatidia's missing positive-multiplier floor: extreme
negative logits could lose candidate mass and invent black. Commit `73fb954`
aligns the graph with the existing scalar floor and denominator clamp.

Independent inspection then found a second defect in Meganeura's softplus:
`-log(sigmoid(abs(beta*x)))` cancels small positive values near beta*x=-17; the
expanded backward similarly loses the negative-tail gradient. A real fitted
pixel had weight 0.300000 instead of 0.595345, despite a unit total weight.
The previous absolute-error tolerance and a parity test fed from GPU multipliers
hid this. The revised Ommatidia test computes its reference directly from logits
with f64 log1p, checks each multiplier and normalized weight, and covers the tail.

[Meganeura #164](https://github.com/kvark/meganeura/pull/164), revision `8c6a545`,
evaluates log1p through a bounded eight-term atanh series and uses a cancellation-
free sigmoid derivative. It invalidates version-5 compiled plans, so cached
execution cannot silently retain the old lowering. The learned-parameter schema
is unchanged. One pointwise dispatch is retained; no latency claim is made.
The original regressions fail before and pass after the change in
[34313737236](https://github.com/kvark/meganeura/actions/runs/34313737236).
All six upstream CI jobs pass in
[34314004049](https://github.com/kvark/meganeura/actions/runs/34314004049), including
LavaPipe and Metal tests. No numerical tolerance was loosened; a bit-exact test
against the faulty expansion was replaced by mathematical forward/gradient tests.

## Repeated experiments

[34314109253](https://github.com/kvark/ommatidia/actions/runs/34314109253) completed
all seven frozen-fit and four causal jobs. Ommatidia source `80a2746`, Meganeura
`8c6a545` (0.3), published Blade graphics 0.9/render 0.6, Rust 1.92.0, LavaPipe.
Same original captures and fixed budgets; no new architecture. Only Meganeura's
package version changes in Cargo.lock; transitive resolutions are unchanged.
All model checkpoints are reloaded before scoring. These are construction
experiments, not new-scene generalization or production-speed measurements.

### Frozen native selector

Seed 7, 512 updates, one 32x32 output frame. Native candidates/observations and
zero-head warmup history are frozen, not fed from a teacher. The free per-pixel
logits and the ordinary core8 network start at the same mixture, with learning
rates 0.03 and 0.001 respectively. Equal updates are not equal convergence.
The RGB bound accounts jointly for diffuse/specular mixing and remodulation.

| Objective / frame | Initial MSE | Conditional optimum | Free logits | Native network |
|---|---:|---:|---:|---:|
| Lobes / reset | 2.296399 | 1.964121 | 1.966891 | 2.092701 |
| Lobes / frame 3 | 0.258796 | 0.113954 | 0.115121 | 0.123556 |
| Final RGB / reset | 0.696045 | 0.607629 | 0.608397 | 0.660932 |
| Final RGB / frame 3 | 0.146332 | 0.053816 | 0.054812 | 0.057286 |

All saved weights now agree with independent stable-softplus reconstruction to
within **2.4e-7 absolute**, versus discrepancies up to **0.295** before.
Free logits realize 98.9–99.2% of the initial-to-optimum loss reduction. The
network realizes 39.7%/96.2% for RGB and 61.3%/93.4% for lobes. These percentages
are not image accuracy: the candidate optimum itself still loses visible detail.
Neither an optimization ceiling nor a generalization result is established.

The reset-RGB network is heavily saturated: 71.9% of legal multipliers hit the
floor; its mean broadest-scale weight is 0.9232 and logits span -204.6 to +332.6.
That is observed conditioning/parameterization trouble to isolate, not proof that
another loss or larger network will fix it. Some final-network finite differences
remain inconclusive at small gradients/high curvature. The full nonsaturated
mixing-gradient check and upstream relative tail checks pass; this is not a blanket
claim that every finite-difference probe passes.

### Field component isolation

One 64x64 construction scene, original VisibleRgb core8/hidden32, three context
cameras (0,2,4), fitting camera 1. Separate objectives and label counts:

| Component | Fixed observations | Initial → final loss | Fitting result |
|---|---|---:|---|
| Source termination | 12,288 source pixels, 17 classes | CE 2.833213 → 0.327206 | 88.25% correct bins; 99.49% hit/miss |
| Volume termination | 64 rays, 64 intervals + escape | CE 4.247359 → 1.992199 | 39.06% correct bins; 96.88% hit/miss |
| Surface appearance | 48 valid true-surface points | log1p MSE 0.218008 → 0.00010853 | 42.67 dB compressed PSNR; linear MSE 0.0005154 |

**The appearance row deliberately uses true positions.** It is a privileged
fitting diagnostic, not an inference path, full image or novel-view result.
Source termination is unchanged; appearance remains a good fit on these few
points. Volume fitting remains unstable: last-64-update mean loss is 4.0892 and
earlier losses dip below 0.42 before later spikes. The final fixed-budget result
is worse than the old runtime's 0.8841, and is reported rather than selecting an
earlier minimum. This single repeat cannot attribute optimization instability
solely to the tail correction or establish a representation limit.

### Own-history held-noise denoising

Repeat the earlier construction recipe: two scenes, three fitting and three held
noise streams, 512 updates per arm, seeds 7/11, core8 and two-frame unroll. All
spatial weight is fixed-exposure physical loss (lobe or final RGB); other losses
are zero. Each trained model and the deterministic baseline own their histories.
All 24 fitting and 24 held frames per arm are scored.

| Held-noise mean | PSNR x/(1+x) | Linear RGB MSE | Energy ratio | Temporal MSE |
|---|---:|---:|---:|---:|
| Independent deterministic recurrence | 18.6327 | 0.636756 | 1.030414 | 0.0114586 |
| Corrected lobe objective | 18.8804 | 0.546332 | 0.992521 | 0.0109578 |
| Corrected final-RGB objective | 18.9328 | 0.649010 | 0.986174 | 0.0107145 |

Results are essentially unchanged from the earlier own-history study: old lobe
PSNR was 18.8864 and RGB PSNR 18.9328. The fixed implementation is necessary for
correctness, but this test does not demonstrate a denoising breakthrough. Final
RGB still has 18.8% higher linear error than the lobe control. No default changes.

## Verification and reproduction

Archive SHA-256 values match GitHub. The independent verifier recomputes losses,
weight masses, radiance, remodulation, probability metrics and stable-softplus
weights from saved arrays; finite checkpoints and complete update logs are checked.
Causal reports have matching capture metadata, configurations, references and
independent baseline images, including against the earlier study. All comparison
images were decoded and representative frozen/causal outputs inspected. The
candidate-optimum image still differs visibly from the reference. Verification
scripts and raw arrays are retained with the experiment artifacts/evidence bundle.

Use `cargo run -p ommatidia-train --bin fit -- --task TASK --data DATA --out OUT
--frame 0 --seed 7 --steps 512` for selector-rgb, selector-lobes, source-depth,
volume-depth or surface-appearance (also selector frame 3). The unchanged
`benchmarks/correspondence-lavapipe.sh` reproduces causal arms on declared captures.
Use the pinned corrected runtime, not old compiled plans or mislabeled archives.

Normal CI retains 29 Ommatidia GPU tests, physical incident capture, and bounded
fitting/training/reload checks. Final exact-head status is in PR #19. The temporary
integration workflow is removed; only the four retained workflows remain. README
images remain visible. No checkpoint, default, joint-weight, SVGF or DLSS claim.

Next isolate selector saturation and volume-fit instability with the corrected
control. A centered masked-softmax mixture is a parameterization experiment, not
another prediction head; it must preserve legal candidates and baseline priors.
For fields, stabilize fixed-ray geometry before restoring joint reconstruction.
