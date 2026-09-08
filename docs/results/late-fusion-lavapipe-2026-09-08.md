# Late appearance fusion, recovered capacity, and candidate ceiling

**Working implementations; no checkpoint promotion.** Late source RGB improves
compressed-colour PSNR but worsens linear error in this short construction test.
The denoiser oracle establishes conditional selector headroom, not a learned gain.

All four experiments completed successfully in [LavaPipe run 34171548358](https://github.com/kvark/ommatidia/actions/runs/34171548358).
Code: `820467cd4bd6de72eb0a1fd4a8d4675a0296d810`; Rust 1.92.0;
Meganeura `10b5961` (the same numerical fix as final `3d424e0`, which only adds
explicit-reference lint patterns and removes temporary workflows); published
Blade graphics 0.9/render 0.6, Naga 30. Software device: llvmpipe, LLVM 20.1.2.
Artifacts retain complete scores, images, commands, input hashes and weights.
No LavaPipe timings are interpreted as hardware performance.

## Wider checkpoint recovered without retraining

[Meganeura PR #160](https://github.com/kvark/meganeura/pull/160) deduplicates tied
Winograd transforms and excludes declared execution caches from logical
checkpoints. Original weights remain strictly checked; no suffix-based filtering,
checkpoint rewriting or disabling of Winograd.

The earlier `control-field-wide` checkpoint had completed 2,048 training updates
but failed at inference loading. `field --eval-checkpoint` now evaluates it with
zero optimizer updates. Its original SHA256 stayed unchanged:

```
543f4cb78cb3a9e80c4448fff0c1c5392d969631fd1069357a2ea9d92c8cfde7
```

Core16/hidden64, 176,906 parameters, 64 midpoint samples per ray at evaluation:

| Metric | Fitting camera | Separate validation camera |
|---|---:|---:|
| PSNR in x/(1+x) | 23.5378 dB | 20.9840 dB |
| Linear RGB MSE | 0.67862 | 0.69011 |
| Conditional hit-depth MAE, world units | 0.66826 | 1.01594 |

Validation hit/miss accuracy is 97.63%. This is one construction scene, not a
new-scene audit. Do not compare it to the 512-update pair below as a controlled
capacity result. Recovery writes a separate directory; a CLI regression prevents
accidental overwrite of the original run, including a `child/..` path alias.
The `late-recover-wide` artifact is evaluation-only; original weights remain in
run 34152574900, artifact `control-field-wide`.

## Paired source-view fusion

One construction geometry (seed 20260907), eight 64x64 centre-ray views at 256
reference spp; three context views, four fitting cameras and a separate validation
camera. Core8/hidden32, 32 image rays x 64 stratified samples, 16 emission probes,
surface weight 0.05, 25% emitter-pixel proposals, seed7, **512 updates** each.
Both see 16,384 image rays and 1,048,576 rendering queries. Query jitter uses an
independent stream. No incident loss. Neither checkpoint is selected by validation.

`moments` has 45,258 parameters; `late-rgb` has 47,439. Their common parameter
prefix retains the same initialization order. Data, context RGB, fitting/held
reference PNGs and ray/pixel sampling recipe match. The new architecture adds
per-view scoring and a linear source-RGB path to the appearance head; geometry
still uses pooled features. It is not a full visibility-aware geometry model.

| Metric | Moments | Late RGB |
|---|---:|---:|
| Fitting-camera PSNR | 18.2895 dB | 19.2547 dB |
| Validation-camera PSNR | 17.5538 dB | 18.5112 dB |
| Validation linear RGB MSE | 0.95095 | 1.25685 |
| Validation log1p RGB MSE | 0.06732 | 0.07400 |
| Validation depth MAE, world units | 1.80639 | 1.87930 |
| Validation hit/miss accuracy | 94.26% | 92.90% |

PSNR uses x/(1+x), not linear HDR. The +0.9574 dB validation PSNR comes with
**32.17% more linear RGB error** and worse depth/occupancy accuracy. Both inspected
images remain blurry. Source copying produces visible streaks where support is
wrong; this is not evidence that projection alone resolves occlusion.

Zeroing source RGB reduces validation PSNR to 13.46/10.62 dB for moments/late RGB.
Reassigning images to unchanged poses scores 17.59/17.75 dB respectively. These
are input-dependence diagnostics, not proofs of correct correspondence. Keep
late RGB opt-in. Test longer matched budgets and multiple scenes/seeds before
attributing the tradeoff to capacity, optimization or geometric ambiguity.

Artifacts: `late-moments`, `late-late-rgb`. Neither result replaces the README's
explicitly labeled, longer surface-supervision experiment.

## Denoiser: conditional convex candidate oracle

Use the saved cache-corrected `transport-control` checkpoint and the same four
16-frame development sequences from seed50000. After each normal native frame,
read the actual GPU candidates, prior availability and selected weights. Project
each target RGB lobe onto the convex closure of available candidates. Rejected
history is excluded. The oracle never replaces history or feeds another frame.

The objective is linear RGB lobe squared error. Means over all 64 frames:

| Same actual learned-history state | Diffuse lobe MSE | Specular lobe MSE |
|---|---:|---:|
| Fixed prior mixture | 0.40568 | 0.03012 |
| Learned selector | 0.30282 | 0.02710 |
| Convex oracle | 0.20615 | 0.01924 |

Oracle error is **31.92% lower for diffuse and 29.00% lower for specular** than
learned selection. Maximum first-order dual gap is 2.39e-12. The final displayed
oracle image averages 25.0947 dB versus 22.2144 dB learned, but the solver did not
optimize displayed RGB PSNR. Oracle images retain residual noise/error.

Normal quality.json, all normal learned PNGs and checkpoint bytes match the
non-oracle run exactly. The fixed-prior row above uses the learned trajectory,
not the separate deterministic recurrent baseline. The bound does not apply to
a different recurrent trajectory; oracle weights can be nonunique. It is possible
headroom, not proof that noisy available observations predict the optimum.

Artifact: `late-candidate-oracle`, including per-frame availability, candidate
errors, signed lobe bias, mean shares, and all oracle/normal/reference images.

## Validation and next decision

The experiment's workspace checks/tests/Clippy and all 20 then-enabled Ommatidia
GPU tests passed. Normal CI additionally covers wider tied-weight loading and
physical incident capture. The final upstream fix passed all six Meganeura CI
jobs, including Linux/LavaPipe and macOS/Metal tests. No tolerances were relaxed.

For the denoiser, test learning achievable selector improvements on construction
data before widening candidate support. Preserve energy/detail/temporal gates.
For the field, late RGB does not remove geometric or visibility ambiguity; finish
sharp fitting and explicit multiview support diagnostics before richer lighting
losses. No shared-weight-transfer or blade-volume result is claimed.

[Contracts and commands](../late-fusion.md) · [Roadmap](../quality-roadmap.md) ·
[Reproduction script](../../benchmarks/late-fusion-lavapipe.sh).
