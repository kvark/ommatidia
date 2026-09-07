# Image-conditioned field: first LavaPipe result

**Pipeline works; geometry/detail quality is not established.** No blade-volume
integration or production checkpoint promotion.

Run `bash benchmarks/field-lavapipe.sh`. Measurements below are from
[34135818542](https://github.com/kvark/ommatidia/actions/runs/34135818542);
the same corrected source passed the complete validation and commit in
[34136772532](https://github.com/kvark/ommatidia/actions/runs/34136772532),
producing `1d0933739f7e168f0b262f1dc866a3d25b1a04e3`.

## Fixed experiment

LavaPipe/llvmpipe, LLVM 20.1.2; Rust 1.92.0; Meganeura `d903bba`; published
Blade graphics 0.9.0, render 0.6.0, asset 0.2.2 and Naga 30.0.1. Quality only.
Two fitting scenes (seeds 7, 8), one untouched scene (10000), six cameras each,
16×16 linear RGB, 128 reference samples/pixel. Lighting seeds 19/23. Two source
views; separate fitting cameras; final camera held. The new scene's images or
labels never enter optimization. Its two source RGB views are legitimate
inference inputs, not a zero-shot-without-observations claim.

The deliberately small model uses channels=4, hidden=16, eight rays × sixteen
samples, eight emission probes and 64 updates. This does not test the larger
64-channel/128-hidden default. Save and reload precede held scoring; the held
metric does not select the checkpoint. The artifact contains recipe, dataset
hashes, loss CSV, weights, runtime-only RGB contexts and three comparison PNGs.

## Held camera of the unseen scene

| Method | PSNR, x/(1+x) | Linear MSE | log1p MSE |
|---|---:|---:|---:|
| Black | 6.8891 | 4.55076 | 0.59601 |
| Constant source-view RGB mean | 14.9628 | 3.28713 | 0.21370 |
| Untrained field | 14.2284 | 3.63398 | 0.25929 |
| Trained field | **16.1860** | **3.22769** | **0.19363** |

The inspected prediction is still very blurry: broad colour improves, but
boundaries and source detail are absent. One tiny scene and one training seed
cannot establish geometry recovery, generalization, relighting, cross-task
transfer, or a benefit specifically attributable to light supervision.

## Contract tests

Workspace tests and Clippy pass. All 13 GPU tests pass: nine legacy/runtime,
two multiscale transport and two field tests. Field checks cover projection,
label isolation, direction-independent density, source-view permutation,
CPU/GPU volume integration, gradients into the shared encoder/density/emission
heads, and checkpoint reload. A separate fixed-batch field test reduces loss
0.102511→0.014806; that is a gradient/optimizer check, not an image-quality score.

[Architecture, labels and next controlled experiments](../field.md).
