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

## Ordered experiments

1. **Learn the measured scale-selection opportunity.** Train a small head to
   produce bounded, convex mixtures of the six already-computed à-trous scales,
   independently for diffuse and specular radiance. Supervise clean lobes and
   compare against oracle distillation. Prove the feature set in the CPU
   evaluator before adding a GPU path. The first gate is recovering at least
   half of the held-out oracle gap: 31.97 dB PSNR, 0.9395 SSIM, and 37.0 dB
   low-frequency PSNR on that same audit, with no detail regression and a
   per-scene luminance ratio between 0.98 and 1.02.

2. **Broaden the clean corpus.** Add scene-held-out captures spanning hard and
   soft shadows, small emissives, interiors, indirect fill, HDR highlights,
   glossy and rough materials, textured geometry, thin silhouettes, and
   animated occlusion. Keep independent high-sample references and measure
   their own convergence. Procedural variants of the same scene family must
   not cross the train/validation boundary.

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
