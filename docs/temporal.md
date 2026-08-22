# Temporal reconstruction

The published default checkpoint is spatial: every output is a function of the
current frame only. Training, evaluation, and the Rust shared-context runtime
now also have an experimental recurrent path. The runtime owns reprojection,
validity, and ping-ponged history resources rather than asking the renderer to
manage model-private state. A renderer already paid for motion vectors, and a
previous reconstructed frame contains roughly three quarters of the output
samples that a 2× spatial model is otherwise asked to invent again.

The first path-tracing experiment exposed a weak nearest-neighbor base rather
than a transformer deficit. On 76 crops from a separate 128-scene 4-spp
validation set, bilinear scores 26.46 dB / 0.5864 SSIM. Low-resolution
depth/normal/albedo guidance reaches 34.08 dB / 0.9473; supplying the exact
output-resolution primary surfaces plus a held-out guide tuning raises the
current 128-crop release score to 34.72 dB / 0.9574 before the learned
correction runs. The spatial network reaches 34.74 dB / 0.9575.

This makes geometry-aware spatial reconstruction credible, but it also makes
the remaining limitation unambiguous: a larger static network cannot recover
samples that were never observed. History is the next quality source for lower
ray budgets, sub-pixel evidence across motion, and 1-spp stability. The
superseded nearest-base result remains recorded in
[`results/path-trace-spatial-b24-2026-08-12.md`](results/path-trace-spatial-b24-2026-08-12.md).

The matched 1-spp control makes this an experimental result rather than an
intuition. Its HR guide reaches 31.74 dB / 0.9215 SSIM, while a b8 model trained
from scratch on the same 1-spp distribution adds less than 0.005 dB. The next
model must receive additional samples through valid history; changing the
static block family cannot recover evidence absent from the frame.

A common 128-crop static-history oracle quantifies the available signal. With
the tuned deterministic guide, one accumulated sample scores 32.17 dB /
0.9327 SSIM, two score 33.76 / 0.9488, four score 34.72 / 0.9574, eight score
35.50 / 0.9636, and sixteen score 35.89 / 0.9663. Thus four perfectly aligned
1-spp frames buy +2.55 dB over one frame before motion and rejection losses.
This is an upper bound rather than a temporal-model result, but it is over one
hundred times the gain measured from the static learned residual.

The dataset container records a fixed sequence length in its reserved
header and the generator can emit independent frames with `--sequence-frames
N`. Geometry and the base camera stay fixed while Blade's stochastic frame
index advances the sparse paths; `--camera-motion F` optionally translates the
camera in world X each frame and renders a matching canonical target. Sequence
records include Blade's current-to-previous motion in pixels, decoded from its
compact G-buffer representation. A static four-frame GPU smoke test produced
different radiance with byte-identical geometry/targets and zero motion; the
moving-camera test produced nonzero motion after the sequence boundary.

Legacy files read as length one. The trainer now splits only at sequence
boundaries, draws frames 2–N for temporal batches, and skips reset frames when
scoring. Spatial checkpoints still reject sequence files unless temporal
history is selected. The first frame of each sequence is an explicit history
reset. Curved camera motion and independent object motion are now present;
exposure changes, animation, and reactive masks remain future data work.

The first moving-camera oracle uses 32 four-frame sequences, one independent
path per input pixel, 256 accumulated canonical frames per target, and a 0.05
world-unit camera translation per frame. It caps history at four samples and
compares motion-only reprojection with a depth/normal/albedo consistency gate:

| reconstruction over frames 2–4 | MSE | PSNR | SSIM |
|---|---:|---:|---:|
| single-frame tuned HR guide | 0.000763 | 31.17 dB | 0.9119 |
| motion-only history | 0.001233 | 29.09 dB | 0.9238 |
| surface-rejected history | **0.000584** | **32.33 dB** | **0.9301** |

Motion alone ghosts: it loses 2.08 dB even while SSIM rises, a concrete example
of why SSIM cannot be the only release metric. Rejecting only 2.7% of pixels
turns that into a 1.16 dB and 0.0182 SSIM gain over the current single-frame
base. This establishes reprojection validity as a first-class, observable
stage before learned fusion. The reproducible CPU experiment is
`cargo run --release -p ommatidia-train --bin temporal-oracle -- DATA.omd`;
it adds no Meganeura graph operation or shipping shader.

A 14-point gate sweep (normal cosine 0.8–0.95, squared albedo delta
0.01–0.16, and encoded-depth delta 0.0025–0.04) stays within 0.01 dB of the
selected result while accepting 96.5–97.3% of pixels. The gain is therefore
not a fragile threshold fit on this set: categorical sky/surface changes and
surface discontinuities identify most bad reprojections. The oracle accepts
the three thresholds as optional arguments so the same claim can be retested
once object motion and harder materials are present.

The first learned gate keeps reprojection outside the network. Surface-rejected
colour replaces the noisy colour plane used by deterministic reconstruction;
the U-Net additionally receives current RGB, normalized accumulated count, and
the exact guided RGB base. These are seven ordinary input channels, so the
experiment adds only stem weights and no Meganeura operation or shader group.

The original 12-channel sub-pixel residual stayed at +0.00 dB after 1,000
steps: once the guide has pooled valid samples, those residuals are dominated
by high-resolution Monte Carlo outcomes absent from the input. A better target
predicts three low-resolution RGB corrections over the safe guided base, then
uses the existing output-resolution geometry gather. A canonical-low oracle
shows 36.11 dB / 0.9602 SSIM versus 32.33 / 0.9301 for rejected history, so the
target has headroom without asking the network to invent sub-pixel samples.

On 512 training sequences and an independent 32-sequence set, b8 reaches
32.28 dB / 0.9292 versus 32.26 / 0.9292 for the temporal HR guide. B16 reaches
32.30 / 0.9294, but raises model arithmetic from 12.7 to 47.2 GFLOP per 1080p
frame and parameters from 73.7k to 290k. This is useful architecture evidence,
not a release candidate: most quality still comes from valid history, and
width has sharply diminishing returns. The complete experiment is recorded in
[`results/temporal-low-color-2026-08-14.md`](results/temporal-low-color-2026-08-14.md).

The next curved-motion gate is recorded in
[`results/temporal-motion-gates-2026-08-15.md`](results/temporal-motion-gates-2026-08-15.md).
It adds a motion-compensated delta metric and a curved-motion training
distribution. The same b8 model then gains 1.17 dB spatially and 0.50 dB
temporally on an independent set; one history-deviation channel raises those
to 1.27 and 0.57 dB for 72 parameters and no dispatch. Eight-frame
accumulation is radiometrically tied, while a supervised blend gate is much
worse. The next larger experiment therefore needs paired consecutive outputs
and a temporal loss; changing the backbone is still unsupported by evidence.

The follow-up
[`object-motion gate`](results/temporal-object-motion-2026-08-15.md) splits a
random sphere and box into independently transformed Blade objects and adds a
moving-pixel temporal score. The camera-trained model already improves that
region by 0.63 dB. Mixed camera/object training plus a corrected 4,000-step
Adam decay raises object-only quality from 34.46 to 34.52 dB and camera-only
quality from 34.19 to 34.21 dB at the same 73.7k-parameter cost. An explicit
velocity feature was worse and was removed. Object motion is therefore now a
release gate; paired temporal loss remains the next justified model change.

The paired loss is in
[`monolithic-kernel-2026-08-15.md`](results/monolithic-kernel-2026-08-15.md).
Weight 1 is the standing temporal setting. Giving the teacher its own
occlusion test, rather than inheriting the sample-history mask, is in
[`teacher-reprojection-2026-08-19.md`](results/teacher-reprojection-2026-08-19.md):
the measurement changes (moving pixels go from −2.24 dB to +0.11 dB on
the same checkpoint) and training against the new target does not.

A second history tap with no surface gate is in
[`unrejected-history-tap-2026-08-19.md`](results/unrejected-history-tap-2026-08-19.md).
It does not help; the flag stays off.

Feeding the previous *reconstruction* as history, which is what a
temporal upscaler actually reuses, is in
[`previous-output-2026-08-19.md`](results/previous-output-2026-08-19.md).
That is the first change that moved temporal stability ( +0.03 dB to
+0.53 dB on the external set). Cutting the same model to b8, in
[`previous-output-b8-2026-08-19.md`](results/previous-output-b8-2026-08-19.md),
is 9.1 ms and 3 dB worse — recurrence feeds the grain back. Stay at
b16. Mixing previous-output after the gather, in
[`previous-output-mix-2026-08-20.md`](results/previous-output-mix-2026-08-20.md),
takes those four taps off the head: 14.66 ms at matched quality.

The matched 1-spp gate and validity correction are in
[`temporal-validity-1spp-2026-08-22.md`](results/temporal-validity-1spp-2026-08-22.md).
Rejected previous-output pixels had been stored as zero without giving the mix
gate their validity, so disocclusions could become black history. Explicit
per-sub-pixel validity fixes that contract. An r3/b16 model trained on 512
four-frame sequences then reaches 27.62 dB versus 26.04 dB for deterministic
accumulation on 63 unseen sequences. Its motion-compensated temporal error is
+4.14 dB better and its 16x16-block temporal error is +9.54 dB better, about a
ninefold reduction in coherent fluctuation. Quality improves through frames
2–4 rather than drifting. This is the first result that resolves the visible
low-frequency failure; it is an experimental quality checkpoint, not the
published default.

The Rust runtime implements the corresponding contract:

1. `FrameInputs::with_motion` takes current-to-previous motion in input-pixel
   units; `with_blade_motion` decodes Blade's compact convention;
2. `Upscaler::reset_history` invalidates recurrence for cuts or any break in
   frame continuity;
3. the upscaler owns ping-ponged low-resolution accumulation, reconstructed
   output, and output-resolution surface history;
4. both accumulation and previous-output reuse validate depth, normal, and
   albedo, and explicit validity hard-closes the learned history gate.

The first live Blade check ran the trained b16/r3 checkpoint on static and
moving four-frame sequences. On the static sequence, display-space 16x16-block
temporal MSE was 0.000045 for native Ommatidium versus 0.000934 for bilinear
1-spp input and 0.000010 for the finite-sample canonical target. At 960x540 →
1920x1080 on an RX 7900 XT, the temporal checkpoint measures 19.82 ms median
end to end: 0.25 ms pack, 17.63 ms model, and 2.33 ms unpack. Recurrent state
occupies 130.5 MiB. A minimally perturbative trace assigns 79.1% of model GPU
time to convolution, 14.2% to pointwise work, 5.0% to normalization, and 1.7%
to data movement. Reprojection is therefore not the primary speed limit;
network width and the 49-tap head are.

Camera jitter/exposure metadata, a reactive mask, and an optional observable
confidence/debug target remain API work. The C ABI still cannot run inference
on externally borrowed Vulkan handles; that separate integration boundary is
tracked in [`integration.md`](integration.md).

Training now uses short sequences with object and camera motion, disocclusions,
independent sparse paths, individual-frame radiometry, and a reprojected
temporal loss. Evaluation reports compressed-space PSNR, SSIM, relative error,
detail, low-frequency PSNR, motion-compensated temporal error, a block-averaged
low-frequency temporal error, and sequence age. Exposure changes, animation,
sub-pixel jitter, and visual regression clips remain before a release gate.

NVIDIA's public material does not disclose a single DLSS training or acceptance
metric to copy. Its [DLSS 2.0 overview](https://developer.nvidia.com/blog/dlss-2-0-ai-rendering)
does emphasize temporal feedback and frame-to-frame stability, while NVIDIA's
[temporal denoising research](https://research.nvidia.com/publication/2020-05_neural-temporal-adaptive-sampling-and-denoising)
co-trains on consecutive frames and calls out disocclusions and moving specular
highlights. PSNR therefore remains the radiometric anchor, SSIM catches lost
local structure, and sequence tests must add motion-compensated temporal error
plus visual regression clips. No one scalar should be a release gate.

NVIDIA's public [Streamline DLSS integration contract](https://github.com/NVIDIAGameWorks/Streamline/blob/main/docs/ProgrammingGuideDLSS.md)
is a useful minimum bar: input
color, output color, depth, motion vectors, exposure, jitter, reset state, and
camera constants are all explicit per-frame data. Ommatidium should use a
similar typed contract without copying Streamline's plugin architecture or
vendor-specific resource wrappers.

This is a checkpoint-format change. Spatial v0.1 weights stay loadable by the
spatial runtime; a temporal checkpoint must declare its history inputs and
motion convention in its sidecar rather than having the loader guess.
