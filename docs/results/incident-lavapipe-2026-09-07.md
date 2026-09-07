# Incident supervision: matched LavaPipe ablation

**Implemented and tested; not a quality promotion.** Incident supervision remains
opt-in. Images are still blurred and weakly scene-specific. No blade-volume
integration, relighting, or realtime improvement is established.

## Reproduce

```
bash benchmarks/incident-lavapipe.sh
```

[Validation run 34141837198](https://github.com/kvark/ommatidia/actions/runs/34141837198)
passed workspace checks, tests, Clippy, all 16 GPU tests and the complete paired
capture/train/reload/evaluation recipe. Its tested source was committed as
`c28a219f90fd8c1e90a836f7e8030578c3c14cb6`. The `incident-validation` artifact
contains source, logs, captures, labels, checkpoints, PNGs, hashes and full JSON
scores. No held score selects a checkpoint or changes a hyperparameter.

Rust 1.92; Meganeura `d903bba`; published Blade graphics 0.9, render 0.6,
asset 0.2.2 and Naga 30. LavaPipe/llvmpipe, LLVM 20.1.2. Quality only.

Two fitting geometries appear under lighting seeds 19 and 31. Two fitting-disjoint
geometries use lighting seed 23. Each has six static 16x16 RGB views, 128 image
reference paths/pixel, and 24 incident probes with eight four-path batches.
Context and held cameras are separate. These are held-out scenes, not a claim
that every scene is new to the project's previous audits.

Both arms use the same graph, data, four incident rays, eight camera rays, sixteen
samples/ray, eight emission probes, four core channels, sixteen hidden units and
128 optimizer updates. Only the incident-loss weight changes: 0 versus 0.1.
Repeat with optimization seeds 7 and 11. Initial-control scores match exactly
within each pair. The larger default field is not tested here.

## Held results

Equal-weight means over two scenes x two optimization seeds; these four rows are
not four independent scene families. Lower error is better; higher PSNR is better.
PSNR uses the named transform x/(1+x), not linear HDR PSNR.

| Metric | Weight 0 | Weight 0.1 | Change |
|---|---:|---:|---:|
| Image log1p MSE | 0.132625 | 0.132245 | -0.29% |
| Image PSNR | 17.8735 dB | 17.7436 dB | -0.1299 dB |
| Total incident log1p MSE | 0.553613 | 0.546458 | -1.29% |
| Direct incident log1p MSE | 0.622452 | 0.614315 | -1.31% |
| Indirect incident log1p MSE | 0.047722 | 0.049431 | +3.58% |

| Optimization seed | Held geometry seed | Image PSNR delta | Total incident error change |
|---|---:|---:|---:|
| 7 | 10000 | +0.0072 dB | -1.25% |
| 7 | 2654428841 | +0.0141 dB | -3.33% |
| 11 | 10000 | -0.0865 dB | +0.43% |
| 11 | 2654428841 | -0.4545 dB | -0.45% |

Direct log-error improves in all four rows, but indirect log-error regresses in
three. The small total-error improvement does not justify a default change.
Visual inspection shows broad colour fields instead of recovered boundaries and
emitters in both arms. Neither a physically identifiable scene nor beneficial
cross-task transfer follows from these metrics.

## What the tests establish

The dedicated capture test checks exact visible emission, constant sky, zero
emission through an opaque blocker, reflected illumination, and distance-invariant
radiance. CPU/GPU integration checks direct + indirect = total. The incident-only
initial learning signal reaches the existing core, density, emission and appearance
parameters; its fixed-batch test loss falls 0.223666 -> 0.002153 and reload works.
No test tolerance was relaxed. These are contract/gradient checks, not image gains.

Labels come from the same canonical material and path tracer as image capture.
They are bounded-depth, radiance-clamped Monte Carlo estimates, not exact
infinite-bounce truth. Variance is estimated from independent batch means and
retains the paired total; it cannot quantify truncation or unsampled rare paths.
Probe strata are pointwise supervision, not angular quadrature.

## Decision

Keep the target format, capture tests, coupled renderer and opt-in loss as an
experimental baseline. Do not increase its default weight or tune on these held
rows. Source localization, surface support and capacity need construction-only
controls before a larger independent scene/camera/light evaluation. Merely
supervising the same field does not guarantee that its density is correct.

[Architecture and label contract](../field.md).
