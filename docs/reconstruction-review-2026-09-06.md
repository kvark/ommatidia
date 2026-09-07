# Reconstruction review — 2026-09-06

## Decision

Keep the goal of a useful portable alternative to DLSS Ray Reconstruction, but stop treating the present quality ceiling as evidence that another small residual or capacity sweep is the right next experiment. The immediate problem is a combination of estimator mismatch, radiometric mistakes, gates that cannot inspect their candidates, and crop-dependent feature normalization. Correct and isolate those contracts before scaling the network.

This review started from `af2fa5d75bb5ba8172010d5f3e08a2952f80b591`. The implementation in this PR is a new, opt-in reconstruction path, not a newly trained checkpoint. No measured quality gain, production timing gain, or DLSS parity is claimed. Existing published weights retain their original interpretation.

The existing project has valuable infrastructure: independent sparse path capture, high-resolution surface guides, exact motion and projection jitter, phase/lobe experiments, causal evaluation, a portable training/runtime stack, checkpoint versioning, and GPU image tests. Preserve these. The failure is not simply that temporal accumulation or G-buffers were forgotten.

## Findings

### 1. Sparse input and reference represented different transport

At the reviewed revision, `ommatidia-data/src/render.rs` used three bounces for sparse paths and eight for references. The test explicitly required the reference to be deeper. Increasing sample count at depth three does not recover the expectation of the missing higher-order paths.

That is a legitimate *transport-regression* research task, but not a clean denoising benchmark. It is especially inappropriate to assume a normalized positive kernel can always solve it: a convex estimator can only select within its available candidate set. Geometry-dependent weights do not guarantee an unbiased estimate either. Remodulation changes the exact convexity statement but does not restore missing evidence.

**Implemented:** independent sparse paths now default to the reference's configured depth; `--input-bounces 3` explicitly requests the historical truncated-input ablation. New captures write `.transport.json` with input/reference depths and whether they match. A copied reference's depth is unknown, not optimistically marked matched. ReSTIR/SVGF controls keep their historical contract and do not accept the independent-path depth override.

**Still required:** regenerate matched-depth training and evaluation captures. The sidecar is provenance, not a complete proof of matched integrators or an enforced trainer admission policy. Check reference convergence, sample independence, ray offsets, exposure and material conventions too. Deeper input rays have a cost; include it in end-to-end comparisons. Existing `.omd` files have not been repaired or relabelled.

### 2. Linear kernels still fed nonlinear temporal averaging

The recent `linear_kernel` change correctly averages spatial taps before compression. However, the reviewed `model::gather`, CPU reconstruction and unpack shader still mix the deterministic guide and warped output in compressed colour. History reprojection also interpolates compressed pixels, and history is stored compressed in f16.

For `c(x)=x/(1+x)`, the arithmetic mean of radiances 0 and 4 is 2. Averaging their compressed values and decoding gives 2/3. This is not an aesthetic difference: the representation itself changes the estimator. The same issue recurs at fractional motion, not only at explicit mix gates.

**Implemented:** versioned `fusion::Mode::{Legacy, Linear, CandidateAware}`. The new modes interpolate and mix linear, possibly demodulated radiance; compression is used for network features and loss. The existing history texture stores linear demodulated f16 in the new modes. CPU teacher/evaluation replay quantizes that linear history to match native storage. Missing fields select Legacy, including sidecars that already have `linear_kernel: true`.

This removes these particular nonlinear averaging errors, not all bias. The rational loss transform, the established inverse-transform ceiling, learned sample selection, clamping, finite precision and reference noise remain separate issues. An explicit exposure/HDR contract should replace the fixed ceiling in a later version, with tests rather than silent reinterpretation of weights.

### 3. The history gate did not see the output it judged

The backbone consumes `cond`; the guide and actual warped previous reconstruction are supplied later to reconstruction. Thus the old gate may infer generic trust from geometry and accumulated evidence, but cannot directly compare the actual candidate images. Valid primary-surface reprojection does not establish stable appearance: a moving reflection, changing light or shadow can invalidate colour while the visible surface remains the same.

**Implemented:** the candidate-aware head predicts four signed coefficients for each gate and output subpixel. In the existing unpack pass, those coefficients act on

```
[1, mean((C-G)^2), valid*mean((C-H)^2), valid*mean((G-H)^2)]
```

where C is the actual learned spatial gather, G the deterministic guide and H the warped prior output, encoded for bounded numerical range. The sigmoid gates then blend *linear* candidates. Invalid history hard-closes the history gate; its placeholder zero is not treated as black evidence. The same contract exists in the differentiable image graph and CPU reference. Calibration adjusts signed gate intercepts, not arbitrary coefficient magnitudes.

This is deliberately a small, testable intervention. It does not add a second network pass, another history texture, learned recurrent features, temporal attention, or lobe-specific flow. It does widen the head, so unchanged dispatch counts are not a claim of unchanged latency.

A detached `confidence_target` primitive computes the constrained least-squares history fraction for a pair of linear candidates. Invalid, non-finite or indistinguishable candidates produce no label. **An auxiliary confidence-loss trainer is not wired in this PR.** The candidate path currently learns through reconstruction loss; a label utility is not a trained confidence objective. The old positive-odds-only gate telemetry is suppressed for signed heads rather than reporting coefficients as probabilities; actual-candidate usage telemetry remains work to do.

### 4. Spatial GroupNorm changes the operator between crop and full frame

`Builder::group_norm` passes `H*W` to Meganeura. The pinned Meganeura GroupNorm shader reduces across every spatial value in a channel group. Consequently, the same interior patch can receive a different prediction when the rest of the image changes, even beyond the convolutional receptive field. A 64-pixel training crop and a full-frame runtime are not simply two extents of the same local operator.

This is a concrete behavior and a plausible contributor to unstable illumination, not proof that it explains the observed quality gap.

**Implemented:** an independent `--backbone local` option: no spatial normalization, fixed 0.1 scaling on residual branches, direct objective only. Legacy GroupNorm remains the default. A nonzero-head test compares an aligned crop with the full-frame interior and includes the old GroupNorm path as a positive control. Multi-level crops also need aligned stride phase and adequate halo; removing normalization does not eliminate those requirements.

### 5. The negative probes do not rule out recurrence or useful attention

The [causal rollout result](results/causal-rollout-2026-09-04.md) is useful negative evidence: replaying four frames with detached predecessor outputs did not improve the selected model. It is not full backpropagation through time. Equal frame exposure also changes optimizer-update count when four losses are accumulated into one update. Report both budgets rather than concluding that recurrence in general has been falsified.

Likewise, the failed larger residual and spatial bottleneck-attention probes do not test a model that transports learned temporal features or selects among multiscale lobe estimates. They justify rejecting those particular checkpoints, not permanently closing all larger-network or attention designs. Preserve the results; remove their use as universal architectural claims.

### 6. The spatial support and state are still too narrow for the whole problem

A wide receptive field in the weight predictor does not enlarge the radiance support of its final small kernel. The deterministic guide can offer a broader candidate, but a binary choice between a small gather and one guide is not a general multiscale reconstructor. Low-frequency lighting variance needs broad evidence without washing out geometry and high-frequency detail.

The existing [split-lobe experiment](results/split-radiance-atrous-2026-08-23.md) already demonstrates useful multiscale filtering. The next architecture should carry that evidence into *native moving-frame reconstruction*, rather than only add another RGB correction after it. Diffuse illumination and specular radiance need separate confidence, moments, temporal horizons and support. Primary motion is not reflected-scene motion. Roughness and specular hit distance/motion should be used when the renderer can supply them; missing optional signals need explicit contracts.

There is also a reprojection validation question: `Surface::matches` compares current and previous encoded depth without an expected previous-view depth. Camera translation changes depth even for the same surface. Test this with known transforms and thin/far geometry before changing tolerances; the audit does not establish it as the measured dominant defect.

## Target architecture

Build a compact, estimator-aware recurrent reconstructor, not a diffusion image generator and not a large residual attached to a mostly finished image.

1. Preserve sparse radiance, sampling phase and surface data. Use matched transport for denoising experiments. Separate diffuse illumination, specular radiance and emission where capture supports it.
2. Maintain explicit per-lobe radiance/moments, age and validity. Reproject with per-tap surface validation and cut/reset semantics; treat appearance reactivity separately from geometric validity.
3. Form a small multiscale set of current-frame and temporal candidates. Let a local low-resolution network predict filtering/mixing decisions using the actual candidate differences and uncertainty. Keep a reliable deterministic candidate as a fallback, not as a compulsory information bottleneck.
4. Reconstruct at output subpixel positions with high-resolution geometry and exact albedo remodulation. Preserve direct emission. Store linear history; compress only features/loss according to a versioned exposure contract.
5. Train on causal state with explicit confidence/reactivity targets and masked, reference-change-aware temporal losses. Introduce short differentiable unrolls only after the simple contracts and optimizer are validated.

A learned low-resolution feature state and temporal/windowed attention remain viable later experiments. They require their own motion/occlusion and memory-bandwidth design. They should earn their cost by beating the candidate/multiscale control, not be added because a paper or product uses the word transformer.

## Experiment order

### A. Fix the measurement before selecting another checkpoint

Freeze scene-family-disjoint training, selector and untouched audit sets. Regenerate matched-depth paths, preserve motion/jitter metadata, and verify reference convergence with independent reference samples. Use full-frame causal evaluation, not only random crops. Keep first frames, cuts, disocclusions, moving reflections and changing lighting as named subsets; use sequences long enough to expose mature state and drift.

A ReSTIR+SVGF image remains a useful *pipeline* comparison, but changes both sampling and denoising relative to independent paths. For a denoiser comparison, SVGF must receive the same noisy frames, transport, guides and motion. Include all ray, guide, reconstruction and presentation costs. Compare DLSS **Ray Reconstruction**, not frame-generated frame rates, on supported hardware with correctly matched inputs and quality mode.

### B. Isolate the new contracts

Use identical matched captures and comparable schedules for these arms:

| Arm | Fusion | Backbone | Question |
|---|---|---|---|
| A | legacy | group-norm | matched-data baseline |
| B | linear | group-norm | radiometry/history-storage effect |
| C | candidate | group-norm | candidate visibility effect |
| D | candidate | local | crop-normalization effect |

For temporal arms retain `--prediction kernel --reconstruction-base sample --demodulate --history-frames 4 --temporal-features variance --previous-output --guide-mix`. Add `--fusion linear`, or `--fusion candidate`, and independently `--backbone local`. These are *new training configurations*, never sidecar edits to old weights. Keep `--rollout-training` controlled across arms; compare equal-frame exposure and equal optimizer updates separately.

Do not use the old depth-mismatched `.omd` files to interpret the matched-denoising result. Do not select a history-odds multiplier on the untouched audit set.

### C. Add state and supervision only after the bounded ablations

Wire identifiable confidence labels and candidate-aware usage telemetry into training/evaluation. Then bring per-lobe multiscale candidates and moments into native recurrence, with longer camera/object/light trajectories. Compare a small learned feature state against radiance-only state. Expand capacity or introduce temporal attention only when this control demonstrates an evidence/representation bottleneck.

### Promotion gates

A checkpoint must improve full-frame quality without buying it through energy loss, ghosting or broad blur. Report PSNR/SSIM in a named transform, linear energy error, low-frequency error, detail retention and reference-change-aware temporal error, including subset and per-sequence distributions. Inspect moving sequences at normal playback speed. Use repeated training seeds for finalists and confidence intervals over independent sequences/families, not individual correlated pixels.

The first external milestone is a convincing quality/latency result against matched-input SVGF. DLSS RR is the subsequent target, with an explicit quality and frame-time envelope. There is currently no defensible promise of parity on arbitrary games or equal latency across different GPU architectures.

The existing temporal baseline is already reported at 15.77 ms median on RX 7900 XT, excluding ray tracing and some host work, with 142.4 MiB recurrent history. Its fresh real-mesh audit loses 0.53 dB to its own guide. That is the wrong quality/performance point to scale blindly. See the [recorded probe](results/dlss-4-5-probes-2026-09-04.md), not these figures as measurements of this PR.

## Cleanup and compatibility

The README now leads with the actual research status. Historical result tables and comparison images live in [results-overview.md](results-overview.md); the former diffusion-first design lives in [design-legacy.md](design-legacy.md). The active [design.md](design.md) states the current checkpoint and dataflow contracts. Earlier roadmaps point here instead of giving contradictory next steps.

No published checkpoint, dataset or comparison image is deleted. No legacy sampler or serialized enum variant is removed before a new quality winner exists. Compatibility code is not automatically dead code. Future removal should be gated on a known supported-checkpoint matrix and an explicit version boundary.

## Validation scope

The patch adds mathematical fusion and fractional-reprojection tests, sidecar/config compatibility checks, matched-depth capture tests, CPU/image-graph/WGSL recurrence and reset parity, a nonzero-head crop/full-frame normalization test, and a candidate/local backward-optimizer smoke test. Exact execution results are recorded in the PR. A passing smoke test is not a converged training run or a production-GPU performance result.

No fresh quality training, full-sequence benchmark, new checkpoint publication or hardware DLSS comparison has been performed as part of this code review.

## Primary external references

- [SVGF, Schied et al., 2017](https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced): temporal accumulation, luminance variance and hierarchical filtering are a useful deterministic baseline, not a reason to discard neural reconstruction.
- [NVIDIA Real-time Denoising](https://github.com/NVIDIA-RTX/NRD): production input contracts and lobe-specific reconstruction provide useful engineering comparisons.
- [NVIDIA's DLSS 4 technical report](https://research.nvidia.com/labs/adlr/DLSS4/): the published design emphasizes long-range spatiotemporal evidence and GPU/network co-design. It does not establish that copying a generic attention block reproduces its quality or runtime.
- [Vulkan DLSS RR integration sample](https://github.com/nvpro-samples/vk_denoise_dlssrr): a practical primary reference for an eventual correctly matched RR comparison.
