# Masked softmax: correct normalization, not a saturation cure

**Implemented and tested; no quality promotion.** The prepared patch is now
installed, not merely staged. Source `f795d3a` uses merged Meganeura `185b101`,
with published Blade graphics 0.9/render 0.6 and unchanged Cargo.lock.
[Run 34364507785](https://github.com/kvark/ommatidia/actions/runs/34364507785)
completed preparation and all eight quality jobs. No experiment is still pending.

## Contract

`--mixture masked-softmax` enables a prior-aware softmax, centered over legal
candidates before adding log-priors. Unavailable history has zero weight and
gradient. Common logit shifts cannot erase relative priors. Gain is
`0.5 / ln(2)`, matching normalized softplus's initial sensitivity; zero heads
produce the same prior mixture and common parameter initialization is identical.
No observation, learned parameter, spatial support or recurrent state is added.
Centering prevents numerical offset trouble, not saturation of relative logits.

Softplus remains the default and keeps version 1. Masked softmax requires a
version-2 sidecar; missing fields still select the old interpretation and
mismatched versions are rejected. Evaluation obtains the mode from the saved
sidecar, never an override flag. Training, frozen fitting and native recurrence
all use the same mixer. No existing checkpoint is relabeled or promoted.

## Frozen fitting

Same stored observations, candidates, priors, references and zero-head warmup
history; output is 32x32. Four jobs compare both modes on reset and frame 3.
Each job independently fits free per-pixel logits (LR 0.03) and the original
core8 network (LR 0.001), seed 7 and 512 updates. Equal update budgets do not
establish convergence. Reference-based optima are diagnostic bounds, not models.

| Final linear RGB MSE | Initial | Candidate optimum | Softplus | Masked softmax |
|---|---:|---:|---:|---:|
| Reset, free logits | 0.696045 | 0.607629 | 0.608397 | **0.608166** |
| Reset, network | 0.696045 | 0.607629 | **0.660932** | 0.662024 |
| Frame 3, free logits | 0.146332 | 0.053816 | 0.054812 | **0.054005** |
| Frame 3, network | 0.146332 | 0.053816 | **0.057286** | 0.062423 |

Free softmax logits recover 99.4%/99.8% of the initial-to-optimum improvement.
The network recovers only 38.5%/90.7%, versus 39.7%/96.2% for softplus. These
percentages are not image accuracy. Even optimum candidates visibly miss detail.

The reset softmax network is nearly one-hot: mean maximum row weight 0.999962,
79.6% of legal normalized weights below 1e-6, logits -214.8 to +302.7, and head
gradient norm 3.25e-9. Softplus has 73.8% below the same normalized-weight threshold.
Changing normalization did not fix the network's optimization trajectory.

Independent reconstruction from saved logits reproduces every mixture weight
within 2.25e-7 absolute and verifies lobe mixing, remodulation and final losses.
An additional CPU PyTorch replay of the saved convolutional network in f32/f64
reproduces the saturated result and negligible reset-softmax gradients. Full
replay losses differ by less than 7e-8; convolution-order rounding means this is
not bit-exact activation/gradient equality. Some final-network finite-difference
probes remain inconclusive at tiny gradients; their raw values are retained.

## Each model's own held-noise history

Same six recorded captures as the preceding numerical study: two construction
scenes, four frames per scene, three fitting and three held-noise streams. Input
is 16x16; output is 32x32. Each mode is trained for 512 updates with seeds 7/11,
core8, two-frame unroll, fixed-exposure linear lobe loss only. Other objective
weights are zero. Every learned model and the independent deterministic baseline
owns its own causal history. Saved weights are reloaded before all 24 fitting
and 24 held frames are scored. These are held-noise, not unseen-scene results.

| Held-noise mean over two seeds | PSNR x/(1+x) | Linear RGB MSE | Energy ratio | Low-frequency PSNR | Temporal MSE |
|---|---:|---:|---:|---:|---:|
| Deterministic recurrence | 18.6327 | 0.636756 | 1.030414 | 25.4180 | 0.0114586 |
| Corrected softplus | **18.8804** | 0.546332 | **0.992521** | **25.0487** | 0.0109578 |
| Masked softmax | 18.8320 | **0.505086** | 0.986852 | 24.8529 | **0.0106539** |

Softmax lowers mean linear error 7.55% and temporal error 2.77%, but loses
0.0485 dB spatial and 0.1957 dB low-frequency PSNR, with more brightness loss.
Temporal error alone does not establish improved responsiveness or less ghosting.
The mixed results do not meet joint promotion gates.

| Seed | Softplus PSNR | Softmax PSNR | Softplus linear MSE | Softmax linear MSE |
|---|---:|---:|---:|---:|
| 7 | **18.9075** | 18.7193 | **0.505157** | 0.507017 |
| 11 | 18.8534 | **18.9446** | 0.587507 | **0.503155** |

Direction varies by seed. Both use the same repeatedly inspected development
scenes. This is not a confidence interval, fresh audit, or a DLSS/SVGF comparison.
The field track was not expanded. No production-GPU performance claim is made.

## Verification and reproduction

All eight ZIP hashes match GitHub. The independent report verifies identical
frozen inputs, paired capture metadata/loss settings, full update logs, finite
checkpoints, versioned sidecars, per-frame PSNR aggregation and identical
reference/baseline images. All 596 original PNGs decode and representative
fitting and causal outputs were inspected. Six deliberately corrupted reports,
weights, sidecars, logs or baseline images are rejected; originals are restored.

The added GPU tests cover masked values/gradients, large common offsets,
independent mathematical weights, identical initialization, zero-head recurrence
and reset. The normal CI suite retains these plus a bounded masked fit, native
training, save/reload and override-admission test. Exact final CI status is in
PR #19. No existing tolerance was loosened.

`benchmarks/masked-lavapipe.sh fit MODE` uses `MASKED_FRAME=0` or `3`;
`benchmarks/masked-lavapipe.sh causal MODE` uses `MASKED_SEED=7` or `11`.
MODE is `softplus` or `masked-softmax`; defaults are 512 updates. `MASKED_DATA`
points to the six unchanged `noise0.omd` through `noise5.omd` captures and their
sidecars. The default path matches `benchmarks/geometry-lavapipe.sh noise`.
Fresh captures are a new dataset, not identical historical reproduction.
The harness refuses overwrites and records commands, revisions and data hashes.

`python3 benchmarks/report-masked.py ROOT` requires NumPy and Pillow, all four
fitting archives and all four causal archives extracted into named directories.
It independently recomputes the mixture arithmetic and writes `summary.json`.
The optional independent CPU-network replay and its environment version are
retained in the evidence bundle; PyTorch is not a runtime/training dependency.

Temporary write/validation workflows are removed. The four retained workflows
and README pin merged Meganeura, and README comparison images stay intact.
The earlier timed-out Contents write had actually landed; its commit was
preserved and reconciled using the authoritative Git ref, without a force push.

## Decision

Do not promote softmax or repeat a selector-normalization sweep. A correct mixer
is established; the reset network's feature/logit scaling and optimization still
need controlled fitting, and the candidate hull itself limits detail. The next
quality-first decoder must retain raw spatial/temporal evidence rather than only
choose among already filtered images. Build the real recorded-input comparator
in parallel; the declared DLSS 4 RR goal remains unmeasured and unmet.
