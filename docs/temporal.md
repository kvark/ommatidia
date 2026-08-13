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

The first temporal model should keep reprojection outside the learned network.
The GPU pack stage will sample the previous high-resolution output at
`current_pixel + motion`, reject history using depth and normal disagreement,
and space-to-depth the accepted RGB plus one confidence channel. The U-Net then
receives current sparse color/G-buffer and reprojected history, all at input
resolution. This preserves the current low-resolution execution shape and adds
cost mainly to the stem convolution rather than every layer.

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
