# Temporal reconstruction plan

The current checkpoint is spatial: every output is a function of the current
frame only. That is a useful bring-up target, but it leaves both quality and
performance on the table. A renderer already paid for motion vectors, and a
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

The dataset container now records a fixed sequence length in its reserved
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
reset; object motion, disocclusions, and randomized trajectories remain the
next data expansion.

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

The runtime contract needs four additions:

1. current-to-previous motion vectors with an explicit pixel/normalized scale
   and direction convention;
2. camera jitter and a reset flag for cuts, resolution changes, and invalid
   history;
3. two library-owned output textures, so frame N can read N-1 while writing N;
4. depth/normal history validation and an observable confidence/debug target.

Training must move from independent scenes to short sequences. Each sequence
needs object and camera motion, disocclusions, exposure changes, animation, and
sub-pixel jitter. Losses should cover individual-frame radiometry and
reprojected temporal stability; static PSNR/SSIM alone can reward a blurry but
stable model. The initial gate will therefore report compressed-space PSNR,
SSIM, and a temporal error measured after ground-truth reprojection, with
disoccluded pixels excluded.

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
