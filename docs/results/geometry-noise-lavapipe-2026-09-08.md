# Geometry consistency and held-noise selection

**Validation complete; no quality promotion.** Geometry estimates agree more,
but the field is still blurry. The simple risk predictor does not beat the
existing denoiser. Defaults, main, published weights and blade-volume are unchanged.

## Field: compare distributions, not just mean depth

`--consistency-rays 8 --consistency-weight 0.1` couples the RGB-predicted source
termination distribution to volume integration along the same source-camera ray.
Fine termination masses are summed into 16 bins plus escape; the loss compares
their cumulative distributions. Both branches receive gradients. Source cameras,
pixels and bounds determine these rays, never truth depth. No new parameters or
runtime inputs. A zero coefficient retains the same graph and ray budget.

[Study 34178844726](https://github.com/kvark/ommatidia/actions/runs/34178844726),
model source `0d45e66`, merged Meganeura `43b606f`, published Blade 0.9/render 0.6,
Rust 1.92.0, LavaPipe/llvmpipe LLVM 20.1.2. All four field jobs completed.
One construction scene, eight 64x64 centre-ray views at 256 reference spp: three
context views, four fitting cameras, one separate validation camera. Core8,
hidden32, 64 image rays and eight source rays per update, 64 stratified intervals,
16 emission probes, 1,024 updates, seeds 7/11. Surface/visibility weights are 0.05;
only consistency weight differs. Each arm sees 65,536 image rays, four times the
previous short study, not a convergence guarantee or new-scene evaluation.

| Separate validation camera | PSNR x/(1+x) | Linear RGB MSE | Depth MAE, world units | Source/volume CDF MSE |
|---|---:|---:|---:|---:|
| Control, seed 7 | 19.5398 | 0.894267 | 1.35974 | 0.0488195 |
| Consistent, seed 7 | 20.2085 | 0.635246 | 1.44339 | 0.0183987 |
| Control, seed 11 | 19.9172 | 0.889919 | 1.37109 | 0.0347258 |
| Consistent, seed 11 | 19.9512 | 0.900802 | 1.32804 | 0.0167794 |
| Control, mean | 19.7285 | 0.892093 | 1.36542 | 0.0417726 |
| Consistent, mean | 20.0799 | 0.768024 | 1.38572 | 0.0175890 |

Mean CDF error falls 57.9%; RGB gains 0.3514 dB and 13.9% lower linear error.
Mean depth error is 1.5% worse: seed 7 worsens, seed 11 improves. Linear RGB also
slightly worsens for seed 11. Hit/miss accuracy rises 96.18% -> 97.94%. Fitting
PSNR is only 20.37 -> 20.62 dB. Inspected images remain very blurry, with wrong
object boundaries. Agreement between two estimates is not evidence they are right.

CDF evaluation uses 512 fixed random source pixels, with duplicates allowed,
not all source pixels or an independent geometry audit. The report script
recomputes this metric from every saved distribution and checks matching contexts,
references, configurations, budgets and recorded capture hashes.

## Noise diagnostic: corrected before drawing conclusions

The original `geometry-noise-risk` artifact is superseded. Its regression learned
log-risk instead of mean linear MSE; its common candidate mask also inspected
held-noise availability. The corrected implementation fits linear risk, freezes
cross-noise choices using fitting streams only, and falls back to the evaluated
stream's ordinary prior when a fitted history candidate is unavailable. Single
oracles now use each evaluated stream's own legal candidates. No held targets
choose the fitted predictor, and diagnostic outputs never enter recurrent state.

[Verified rerun 34181706513](https://github.com/kvark/ommatidia/actions/runs/34181706513)
produced `361f485`. It reused the six fresh captures and the unchanged
`selector-control-7` checkpoint. Candidate recomposition equals native linear RGB
exactly in both reset and causal modes; all learned PNGs match the first run
byte-for-byte. Checkpoint bytes are unchanged. Tests reject overlapping streams;
missing Blade fixtures in the earlier validation jobs were restored, not skipped.

Two scenes, four frames each, six non-overlapping recorded LR frame-index ranges:
three fit diagnostic risk, three supply held noise. Input is one path per 16x16
pixel; output/reference is 32x32 with 256 reference spp. Clean references and
surface guides match across streams. Disjoint pseudorandom sample ranges are not
a mathematical independence proof. Risk fitting sees these same scenes, so this
is not unseen-scene generalization. There is no neural-denoiser retraining here.

| Causal mode, mean over 24 held-noise frames | PSNR | Final linear RGB MSE | Energy ratio |
|---|---:|---:|---:|
| Existing learned selector | 19.0678 | 0.514667 | 0.992276 |
| Fixed prior, same learned histories | 18.8049 | 0.604402 | 1.016784 |
| Observable linear-risk regression | 17.6098 | 0.655626 | 0.955621 |
| Reference-fitted per-pixel cross-noise choice | 18.3626 | 0.966384 | 1.014918 |
| Held-target single-candidate oracle | 20.8091 | 0.409932 | 0.955700 |
| Held-target convex lobe oracle | 21.3881 | 0.403393 | 0.964136 |

Each alternative uses the existing model's candidate/history state, not its own
causal rollout. Fixed-prior is therefore not the independent deterministic
recurrent baseline. Oracle rows use held targets; they are not deployable and
minimize lobe error, not final RGB PSNR. Energy is the arithmetic mean of per-frame
linear ratios. In reset mode the regression improves PSNR 17.8951 -> 18.1052 dB,
but worsens linear RGB MSE 0.852923 -> 0.908421. No candidate-risk result passes
all quality gates. Failure of this small linear model is not a bound on neural
selector learnability. All reset/causal rows, lobe errors, bias/variance terms,
images and 255 unavailable-history fallbacks are retained in the evidence.

## Reproduce and decide

`bash benchmarks/geometry-lavapipe.sh field-control` and `field-consistent`, with
`GEOMETRY_SEED=7` or `11`, use the existing corrected field capture. The `noise`
arm needs `GEOMETRY_CHECKPOINT=.../model.safetensors` and generates its six captures.
`python3 benchmarks/report-geometry.py FIELD_ROOT NOISE_QUALITY_JSON --out SUMMARY`
requires all four field reports and both complete noise modes; obsolete log-risk
reports, unequal references and partial metrics fail rather than being averaged.

Normal CI retains the isolated CDF gradient test and a bounded visibility-plus-
consistency train/reload smoke. Long studies remain manual; temporary validation
workflows are removed. The explicit budget-counter type fix affects metadata,
not the completed field weights. No tolerance was relaxed.

Next: explicit multiview correspondence, not another agreement loss, and a native
selector construction fit using full observed guides and final remodulated RGB.
Neither these tests nor earlier auxiliary losses demonstrate a sharp fit, an
SVGF win, useful shared weights or DLSS parity.
