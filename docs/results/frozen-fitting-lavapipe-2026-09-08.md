# Frozen fitting: a real normalization bug, then component limits

**Correctness fix, not checkpoint promotion.** The native selector could lose
all candidate weight and invent black. This is fixed and regression-tested.
Isolated appearance fits well; volume termination is still imperfect. No new
architecture, production weights, shared-weight result or blade-volume integration.

## Execution

The new `ommatidia-train --bin fit` appends isolated losses to existing model
graphs. Source `7ade5db` introduced the diagnostics; `73fb954` fixes normalization.
Dependencies: merged Meganeura `43b606f`, published Blade graphics 0.9/render 0.6,
Rust 1.92.0, LavaPipe/llvmpipe LLVM 20.1.2. Experiment binaries use dev opt-level 1.
CPU reference replays below are additional checks, not software-Vulkan runs.

Original selector/source/appearance reports are from
[34282104186](https://github.com/kvark/ommatidia/actions/runs/34282104186).
The completed volume report is from
[34283375566](https://github.com/kvark/ommatidia/actions/runs/34283375566), whose
field diagnostic code is unchanged. An overlapping temporary workflow cancelled
an earlier volume attempt; no partial result is used. The four repaired selector
fits and before/after regression are from
[34283375770](https://github.com/kvark/ommatidia/actions/runs/34283375770).

All measurements use seed 7 and 512 fixed-batch updates. Each checkpoint is saved
and reloaded before scoring. These are fitting diagnostics, not held-noise,
new-camera or new-scene evaluations. No LavaPipe speed claim.

## Native mixture defect

The scalar reference already floored each positive multiplier at `1e-8` and
clamped the denominator. The GPU graph instead divided raw positive weights by
`sum + 1e-12`. Numerically zero softplus values at sufficiently negative logits
could make every legal weight zero: the output became black, outside the intended
candidate hull. Smaller nonzero sums could also lose mass through the epsilon.

This occurred in actual frozen fitting, not only an artificial test. The old
lobe/frame-0 network reached MSE **1.916356**, below the candidate optimum
**1.964121**; its mean total candidate weight was **0.962890**. That was a violated
constraint, not an extraordinary reconstruction. An independent CPU replay of
the old RGB models found 19 fully collapsed lobe/pixel weights per case, including
nonzero-albedo pixels; replayed logits and RGB match the GPU measurements.

The graph now uses the existing scalar multiplier floor and denominator clamp.
`MIN_MULTIPLIER` is shared, not another tunable option. The new `mixture_mass`
regression first fails on the old implementation, then passes offsets from -1000
to +1000 with unit total weight and CPU/GPU reconstruction parity. The four
corrected fits retain total weights within **2.4e-7** of one.

Published legacy Upscaler/C-ABI behavior is unchanged. **Experimental transport
checkpoints can render differently at extreme logits**; historical images remain
measurements of their recorded runtime, not the corrected one. This guard does
not repair every softplus-tail/gradient issue or guarantee an unbiased estimator.
Selectors can still saturate, and normalized selection can still darken an image.
A post-hoc floor on one old RGB fit actually worsened its MSE; do not equate the
correctness repair with an established full-sequence quality gain.

## Frozen selector: how much is attainable?

Freeze actual native features, candidates, priors and history from frames 0 and 3
of one corrected noise capture (16x16 input, 32x32 output). History is the zero-head
native recurrence up to that frame and stays fixed during fitting. Compare free
per-pixel logits with the ordinary core8 native network; both start from the same
mixture. Learning rates are 0.03 and 0.001 respectively, so equal updates do not
mean equal convergence. This is not either model's learned causal rollout.

The RGB optimum projects onto the joint diffuse/specular candidate set after
observed albedo remodulation and emission, not two independent lobe optima.
Numerical dual-gap certificates are recorded. The lobe optimum is separate.

| Objective / frame | Initial MSE | Conditional optimum | Free logits | Native network |
|---|---:|---:|---:|---:|
| Lobes / reset | 2.296399 | 1.964121 | 1.966891 | 2.092968 |
| Lobes / frame 3 | 0.258796 | 0.113954 | 0.115121 | 0.132985 |
| Final RGB / reset | 0.696045 | 0.607629 | 0.608397 | 0.621652 |
| Final RGB / frame 3 | 0.146332 | 0.053816 | 0.054812 | 0.057309 |

Free logits close **98.9–99.2% of the initial-to-optimum gap**. The network closes
84.1%/96.2% for RGB, versus 61.2%/86.9% for lobes. Neither means 99% accurate
reconstruction: even the candidate optimum differs visibly from the reference.
There is both irreducible error for these fixed candidates and remaining network
optimization/parameterization error. Repaired RGB logits still span roughly
-329 to +324; the guard does not prevent saturation.

Full analytic mixing gradients agree with GPU backpropagation at relative L2
error <=1.1e-6 on non-saturated test logits. A separate PyTorch CPU replay of the
stored network matches final losses and logits; selected gradient norms agree
within 0.16%. That is not an exhaustive proof for every graph or gradient element.
Finite-difference probes retain all three step sizes and failures rather than
asserting that every small gradient passed a float32 difference check.

## Field: isolate three different problems

One 64x64 construction capture. Original VisibleRgb core8/hidden32, three RGB
context cameras (records 0,2,4); fitting camera 1. Each row trains a separate
existing model with only the indicated objective. Label counts, ray counts and
loss definitions differ, so these are not interchangeable quality scores.

| Isolated component | Fitting observations | Initial -> final loss | Result |
|---|---|---:|---|
| Source termination | 12,288 source pixels, 17 classes | CE 2.833213 -> 0.327206 | 88.25% correct bin; 99.49% hit/miss |
| Volume termination | 64 fixed rays, 64 intervals + escape | CE 4.247359 -> 0.884129 | 70.31% correct bin; 100% hit/miss |
| Known-surface appearance | 48 valid hits on an 8x8 ray grid | log1p MSE 0.218008 -> 0.0001157 | 42.52 dB compressed PSNR; linear MSE 0.0004943 |

**Known-surface appearance deliberately queries true hit positions.** It is a
privileged diagnostic to bypass density, never a new inference contract. The
reported PSNR covers those 48 fitted colours, not a sharp 64x64 image or a novel
view. The original appearance path can fit this small problem; that does not
rule out capacity limits on larger texture sets.

Volume queries stay observation-selected, with fixed midpoint intervals and no
truth-guided placement. Its loss reaches density, geometry features and the RGB
adapter. At epsilon 0.001, their directional finite differences agree within
0.62%, 0.34% and 0.07%. Loss is not settled: later training has spikes, and the
last 64-step mean is 0.9868 versus the final 0.8841. Continue diagnosing geometry
optimization before declaring a representation ceiling.

Source-depth finite differences initially looked wrong at large perturbations.
An independent float32/float64 CPU replay matches its loss and selected gradient
norms. Features reach magnitude 2580; reducing the normalized-direction step from
0.001 to 1e-5 makes the CPU head difference converge from 10.55 to 3.113, agreeing
with the approximately 3.112 derivative. This is evidence of strong curvature/
conditioning, not a demonstrated Meganeura backward bug. Fine-step CPU checks
must not be represented as additional GPU checks.

## Reproduce and validation

Create corrected inputs with `bash benchmarks/correspondence-lavapipe.sh capture`.
Run the following for `selector-lobes` and `selector-rgb`, frames 0 and 3:

```sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
cargo +1.92.0 run -p ommatidia-train --bin fit --locked -- \
  --task selector-rgb --data target/noise-data/noise0.omd \
  --out target/frozen-fitting/selector-rgb-0 --frame 0 --steps 512 --seed 7
```

For `source-depth`, `volume-depth`, and `surface-appearance`, use
`target/quality-control-data/field.omd` and separate fresh output directories.
The executable refuses overwrites and labels its checkpoints diagnostic-only.
The evidence bundle includes all seven completed corrected/component reports,
original pre-fix selector archives, raw predictions, gradients, captures/hashes,
PNG comparisons and an independent standard-library verifier. It recomputes
weight mass, prediction losses, depth classification and CE from raw arrays;
checks complete logs, finite weights and identical observations; and rejects
missing runs, old unnormalized reports and corrupted mass/metrics.

Normal CI retains the extreme-logit regression and an eight-update frozen RGB
fit/reload smoke, in addition to the existing 28 GPU tests and capture contracts.
No tolerance was relaxed. The temporary source-transfer and execution workflows
are removed; README images remain. Final head/check status is recorded in PR19.

Next: rerun real held-noise/causal denoising with corrected normalization before
reusing historical selector conclusions. Measure saturation and feature scales;
change parameterization or stabilization only with fixed-batch controls. For the
field, improve volume-termination fitting and then reintroduce RGB jointly at
matched ray budgets. Do not add another light/correspondence head to explain a
geometry optimization failure.
