# Quality roadmap

The objective is not merely to beat Open Image Denoise on one average metric.
Ommatidium should look at least as clean, retain its present edge and dark-region
advantages, and remain temporally stable on scenes that were not used to make
architecture decisions.

## Where the gap is

On the fixed six-scene spatial suite, Ommatidium currently averages 29.255 dB
PSNR versus OIDN High's 28.047 dB and retains 77% of reference detail versus
49%. OIDN nevertheless wins SSIM, 0.8731 to 0.8409, and looks smoother.
Ommatidium's low-frequency PSNR lead is only 0.36 dB and its mean-luminance
ratio is 0.935 rather than OIDN's 0.995. Ordinary PSNR rewards the sharp local
result while underweighting exactly the broad mottling and energy loss that
remain obvious in the images.

The temporal split-radiance experiments identify a concrete next target. On a
32-scene audit, choosing among six existing geometry-aware filter scales per
radiance lobe and 8×8 block raises the fixed estimator from 31.11 to 32.82 dB,
SSIM from 0.9349 to 0.9441, and low-frequency PSNR from 35.33 to 38.68 dB,
without losing mean energy. That reference-derived oracle is not deployable,
but it shows that better scale selection can remove substantially more broad
noise without a larger backbone or new general-purpose Meganeura operations.

The September DLSS 4/5 audit changes the immediate order without changing the
goal. Exact moving/jittered captures, reset-frame sampling, a linear-radiance
kernel, and a validity-aware mix with the deterministic guide now form one
native-compatible recurrent checkpoint. Its first 8k control improves
per-frame PSNR, reset quality, broad temporal fluctuation, and mean energy on
both the selector and an untouched audit. It still loses about 0.4 dB of
fine-grained temporal error to its guide and visibly smooths tight glossy
structure. This is progress, not a release-quality endpoint.

After exact-motion retraining that fine-temporal deficit narrows to 0.17 dB,
but a stricter real-mesh audit reverses the spatial result: on eight held-out
ABO families the network loses 0.53 dB to its own guide. Balancing 24 disjoint
ABO families into a short continuation does not improve that holdout and hurts
temporal quality on both domains. The rollout corpus must include real geometry
from the start; a procedural selector plus post-hoc adaptation is not a release
gate.

The next gates are consequently:

1. measure each reference set against an independently rendered path range and
   stop treating residual reference grain as model error;
2. train through rolled-out state and add explicit reactive/disocclusion
   evidence before reintroducing a temporal loss—the one-step detached teacher
   currently rewards a stable dark answer;
3. publish quality against the same total frame time spent on additional paths,
   not only against one fixed 1-spp input; and
4. only then compare a coarse current-query/history-key attention block with a
   gated convolutional state at matched complete-frame latency and memory.

The reasoning and rejected controls are in
[`dlss-4-5-lessons.md`](dlss-4-5-lessons.md) and the
[`controlled result`](results/dlss-4-5-probes-2026-09-04.md).

The first half of gate 2 is now measured rather than assumed. At equal frame
exposure, feeding the detached teacher causally through complete four-frame
sequences is worse than one-pair training on every spatial and temporal metric,
and a 0.1 temporal term makes it worse again. Both checkpoints are rejected;
the reusable rollout path remains for the reactive/confidence objective that
the result shows is missing. See the
[`causal rollout control`](results/causal-rollout-2026-09-04.md).
Training on procedural and real meshes together from initialization improves
SSIM and fine temporal error, but still trades away procedural low-frequency
fidelity and energy even after history calibration. It is another useful
Pareto point, not the clear visual win required for promotion.

## Ordered experiments

1. **Close the scale-selection investigation.** A pooled CPU selector recovered
   only 26–29% of the held-out PSNR and low-frequency oracle gaps. A spatial
   U-Net trained through the composed image then scored 31.01 dB versus 31.10
   dB for the fixed estimator; low-frequency supervision and smoother block
   decisions did not reverse the result. Both implementations were removed.
   The [`pooled`](results/learned-lobe-selector-2026-08-24.md) and
   [`spatial`](results/spatial-lobe-mixture-2026-08-24.md) reports rule out
   investing native runtime complexity in filter-choice prediction with the
   present data and features.

2. **Broaden the clean corpus and predict clean lobes directly.** Add
   scene-held-out captures spanning hard and soft shadows, small emissives,
   interiors, indirect fill, HDR highlights,
   glossy and rough materials, textured geometry, thin silhouettes, and
   animated occlusion. Keep independent high-sample references and measure
   their own convergence. Procedural variants of the same scene family must
   not cross the train/validation boundary. A first 24-scene direct-lobe probe
   improved error, detail, and temporal metrics on two fresh seed families but
   lost SSIM on the untouched audit; the
   [`result`](results/direct-lobe-residual-2026-08-24.md) was removed rather
   than promoted.
   The generator now loads a glTF catalog (ABO objects, HSSD interiors, Aria
   DTC scans) into the existing procedural room, or as an interior, and records
   asset ids in a sidecar so a hold-out cannot leak. Fetch, capture, and the
   `--eval-data` trainer switch are in [`catalog.md`](catalog.md). A B8
   residual trained on 24 held-out ABO objects
   ([`catalog-b8-split`](results/catalog-b8-split-2026-08-26.md)) peaked at
   +0.20 dB / 0.9156 SSIM versus the estimator's 32.91 dB / 0.9507, then
   overfit. Balanced procedural/ABO training plus an output-detail-preserving
   low-resolution head now clears every metric on a fresh full-frame audit
   when calibrated to a conservative 20% correction: +0.31 dB PSNR,
   +0.0019 SSIM, and +0.46 dB low-frequency PSNR
   ([`result`](results/balanced-low-resolution-2026-08-27.md)). This is a valid
   step, not the visual endpoint; it remains offline until the gain is obvious
   in full frames. The generator now also composes furnished HSSD scenes and
   rejects cross-split mesh reuse. Retain the fixed estimator as the stable
   input and baseline.

3. **Train reconstruction and history together.** Feed new sparse samples,
   reprojected history, validity/disocclusion information, and the learned
   lobe estimate into a sequence-trained residual. Test camera motion, object
   motion, exposure changes, cuts, and newly revealed surfaces. No
   reference-derived selector labels may enter the deployed recurrent state.

4. **Revisit the backbone only if the estimator saturates.** Compare the
   compact U-Net against a compute-matched windowed-attention hybrid after the
   stronger input exists. The previous global-bottleneck attention experiment
   improved only 0.03 dB and a shifted-window model was slower at matched
   quality, so a large transformer is not the default next bet. Any new
   architecture must win at matched frame time and code complexity, not only
   parameter count.

5. **Promote only after fixed visual gates pass.** Run the curated suite plus a
   hidden scene set at identical output extents and fixed display exposure.
   Require Ommatidium to match or beat OIDN High on SSIM and a perceptual
   display-space diagnostic such as FLIP, beat it clearly on low-frequency
   error, preserve the present detail and relative-MSE advantages, keep every
   scene's energy near unity, and avoid a worst-scene regression. For motion,
   report temporal block fluctuation, reprojection error on valid pixels, and
   disocclusion recovery time. Always publish representative full frames and
   crops beside the aggregate numbers.

Only after an experiment crosses its offline gate should its runtime cost be
profiled and optimized. Existing filtered intermediates should be reused; new
shader groups or Meganeura operations need a demonstrated quality or frame-time
benefit before becoming product code.

Independent 16,384-spp references differ at 48.66 dB pairwise, implying a
51.67 dB one-reference noise floor; target convergence is not the current
30–33 dB bottleneck. See the
[`reference-noise audit`](results/reference-noise-2026-08-27.md).
