# Architecture decision: restore first, attend where evidence moves

This note answers a narrower question than “are transformers better than
U-Nets?” Ommatidia has to reconstruct linear HDR radiance from sparse paths,
use motion history, upscale 2×, run on non-vendor-specific GPU kernels, and fit
inside a frame budget. Architecture names are secondary to those constraints.

## Decision

Keep the multi-scale U-shaped image path and make temporal evidence the next
quality milestone. Test attention as a **coarse-resolution fusion mechanism**,
not as a wholesale replacement for the convolutional encoder, decoder, or
sub-pixel output head.

The first controlled candidate replaced only the two 1/4-resolution
bottleneck residual blocks with 8×8 shifted-window attention and a
convolutional feed-forward path. It has 630,768 parameters and an estimated
104.5 GFLOP at 1920×1080 output, versus 649,200 parameters and 104.1 GFLOP for
the convolutional baseline. It used Meganeura's existing differentiable
attention implementation after a reversible NCHW-to-window permutation; it
did not add an image-specific attention kernel or shader group. The pack and
merge entry points extended Meganeura's transpose module for the experiment.

This candidate is an experiment, not the new default. It must beat the
convolutional checkpoint on the same bytes, crop order, seed, optimizer
schedule, held-out scenes, PSNR, and SSIM. Frame time and memory are veto
metrics. A static quality tie keeps the convolutional checkpoint and does not
justify retaining the attention-specific graph surface for a possible future
temporal experiment.

### Controlled result (2026-08-13)

Both arms trained for 20,000 steps on the byte-identical 2,400-scene 4-spp
dataset with batch 8, 64×64 crops, seed 0, and a cosine learning-rate schedule
from `3e-4` to `1e-5`. Final scoring uses 76 crops from a separate 128-scene,
seed-10000 set.

| backbone | parameters | MSE | PSNR | SSIM |
|---|---:|---:|---:|---:|
| convolutional | 649,200 | 0.004303 | 23.66 dB | 0.4595 |
| hybrid 8×8 window attention | 630,768 | 0.004306 | 23.66 dB | 0.4596 |

That is a quality tie, not a transformer win. Training throughput was 38.4
steps/s for convolution and 35.6 steps/s for the hybrid. An interleaved
model-only RX 7900 XT benchmark at 960×540 input measured 19.877 ms median
(19.774–19.929) for convolution and 21.360 ms (21.280–21.431) for the hybrid:
7.5% slower, with 104 dispatches instead of 84. It does not include texture
packing or unpacking.

The convolutional U-Net therefore remains the spatial path. The experimental
window option was removed again: retaining roughly 440 lines of Meganeura
graph, autodiff, compiler, shader, and test surface for a slower quality tie is
the wrong complexity trade. If temporal fusion later demonstrates an attention
win, its dataflow should justify the smallest generic primitive needed then.
We should not spend model or kernel complexity on static transformer scaling
before that evidence exists.

### DLSS 4/5 follow-up (2026-09-04)

The official DLSS 4 report strengthens the case for long-range
**spatiotemporal sample aggregation**, but still does not publish a topology
that can be reproduced independently of its fused FP8/Tensor-Core execution.
A cheaper control added a fourth U-Net level using only existing operations.
At matched data and schedule it used 3.8× the parameters and 21% more
arithmetic, yet lost 0.19 dB PSNR, 0.0127 SSIM, and 0.22 dB low-frequency
PSNR. More spatial reach is not the missing evidence.

Widening the new recurrent gather reaches the same answer. At a matched
4,000-step schedule, b24 gains only 0.03 dB and 0.0029 SSIM over b16 while
nearly doubling arithmetic (148.6 versus 77.3 GFLOP) and worsening temporal
error. B8 retains the static score but learns almost no output recurrence and
loses 0.86 dB temporal quality to its guide. B16 is therefore the smallest
tested useful recurrent width; neither more depth nor more width justifies
additional spatial capacity.

The first exact-motion real-mesh audit also prevents declaring this shape
finished: b16 loses 0.53 dB to its deterministic guide on eight held-out ABO
families, and a balanced continuation on 24 disjoint families does not improve
it. The next comparison must train real and procedural sequences together from
reset with rolled-out state; another post-hoc content fine-tune is not an
architecture experiment.

The DLSS 5 report addresses generative appearance rather than physical
reconstruction. Its transferable choices are causal one-frame streaming,
bounded carried state, pixel-space output, and explicit renderer-attribute
consistency—not its 154M-parameter transformer or perceptual objective. The
current probe therefore keeps the compact U-Net, exact surface-validated
motion, one previous output, and a learned mix between a linear physical
sample gather and a deterministic guide. It adds no Meganeura operation or
shader group. See the full source reading and controlled failures in
[`dlss-4-5-lessons.md`](dlss-4-5-lessons.md) and
[`results/dlss-4-5-probes-2026-09-04.md`](results/dlss-4-5-probes-2026-09-04.md).

This does not rule out attention. It makes the next attention test narrower:
only after reactive/disocclusion evidence and rollout-trained state work should
a coarse current-query/history-key block compete against gated convolution at
matched complete-frame time. A model-family change is not a substitute for a
correct estimator, reset distribution, or motion capture.

## Why not DINOv3

[DINOv3](https://arxiv.org/abs/2508.10104) is strong evidence that large
self-supervised models learn reusable dense semantic features. It is not a
single transformer topology: the released family includes ViT variants and
ConvNeXt distillates. Its gains therefore cannot be assigned to attention,
and they are not evidence that its encoder is a good radiance reconstructor:

- its result combines architecture with training on roughly 1.7 billion
  images and models up to billions of parameters;
- its reported tasks reward semantic invariance (classification, detection,
  segmentation, depth, and tracking), while Ommatidia must preserve exact
  sub-pixel geometry, stochastic sample energy, material response, and HDR
  intensity;
- patchification throws away the pixel-local path/G-buffer alignment that the
  renderer gives us for free; and
- even the small DINO family is in the wrong latency and memory class for a
  portable per-frame post-process.

DINOv3 may later be useful as a frozen perceptual evaluator or teacher on
display-referred images. Its weights and topology should not be imported into
the shipping network.

## What the transformer evidence does say

NVIDIA reports that DLSS 4 moved Ray Reconstruction and Super Resolution from
a convolutional backbone to a custom spatiotemporal transformer, improving
detail, disocclusions, stability, and generalization across sampling patterns.
That is directly relevant evidence. It is also explicitly a hardware/software
co-design using custom fused kernels, on-chip dataflow, tensor cores, and FP8;
the public report does not specify a reproducible model topology. See
[NVIDIA's DLSS 4 technical report](https://research.nvidia.com/labs/adlr/DLSS4/).

Published restoration work points to hybrids rather than a full-resolution
ViT:

- [SwinIR](https://arxiv.org/abs/2108.10257) retains a reconstruction path and
  confines attention to shifted local windows.
- [Restormer](https://arxiv.org/abs/2111.09881) starts from the fact that
  pixel-wise global attention is quadratic and infeasible at high resolution,
  then uses an efficient attention/conv-FFN hierarchy.
- [NAFNet](https://arxiv.org/abs/2204.04676) is an important counterexample to
  “newer block wins”: a simple gated convolutional restoration network beats
  more elaborate designs on several benchmarks at much lower compute.

The lesson is to measure a compute-matched local-attention block and a simple
gated-convolution block. A model-family label is not an ablation.

## Temporal evidence matters more than the spatial block

The original nearest-base one-frame model gained only 0.22 dB at 1 spp and
0.08 dB on the four-sample spatial set. Low-resolution renderer guidance fixes
most of the static denoising failure at 34.08 dB / 0.9473 SSIM; exact
output-resolution primary surfaces and a fixed-cost guide tuning lift the
deterministic 5×5 reconstruction to 34.72 dB / 0.9574 on the current 128-crop
release score. The learned b8 residual reaches 34.74 dB / 0.9575. History
is therefore the missing evidence for lower sample counts, sub-pixel detail
across motion, and temporal stability—not a reason to enlarge the static
receptive field. Two relevant production research results agree:

A matched lower-ray-budget gate reaches the same conclusion. At 1 spp,
bilinear scores 22.46 dB / 0.4293 SSIM and the 5×5 HR guide reaches 31.74 dB /
0.9215. A b8 network trained specifically on those 1-spp records adds less than
0.005 dB after 2,000 steps. More stochastic error does not make a static
residual more learnable; it makes time-domain evidence more valuable.

- [Neural Temporal Adaptive Sampling and Denoising](https://research.nvidia.com/publication/2020-05_neural-temporal-adaptive-sampling-and-denoising)
  reports that reprojected temporal feedback raises the effective sample count
  and improves both fidelity and stability.
- [Temporally Stable Real-Time Joint Neural Denoising and Supersampling](https://www.intel.com/content/www/us/en/developer/articles/technical/temporally-stable-denoising-and-supersampling.html)
  shares a low-precision feature extractor with filtering stages and consumes
  low-resolution temporal inputs rather than paying for two independent
  denoising and upscaling networks.

The target topology is consequently:

1. Pack current sparse radiance, depth, normals, material planes, motion,
   jitter, exposure, and validity at input resolution.
2. Reproject the previous learned feature state with host-provided motion;
   reject it using depth/normal/material consistency and explicit
   disocclusion/reactive masks.
3. Encode current and valid historical evidence through the existing
   multi-scale low-resolution path.
4. Fuse history at coarse levels. Start with gated convolution; compare local
   current-query/history-key attention only after the sequence data path is
   correct.
5. Decode once and use the existing sub-pixel head for 2× output. Preserve a
   compact 1/4-input-resolution feature state rather than a full-resolution
   activation pyramid.

The first moving-camera oracle validates steps 1–2 before a network is added.
On 32 four-frame sequences, motion-only accumulation ghosts and falls from
31.17 to 29.09 dB despite improving SSIM. Depth/normal/albedo rejection drops
only 2.7% of history samples and instead reaches 32.33 dB / 0.9301 SSIM. The
next model should therefore receive an explicit validity/confidence channel;
asking attention to discover reprojection failures implicitly would discard a
measured 1.16 dB gain and make failures harder to inspect.

The first sequence-aware model keeps that deterministic result as its zero
output and appends seven ordinary channels: current RGB, accepted-history
confidence, and guided RGB. Changing the target from 12 unpredictable
high-resolution sub-pixel residuals to three low-resolution radiance residuals
produces a small repeatable external gain (32.26 → 32.28 dB at b8). Widening to
b16 reaches 32.30 dB but costs 3.7× the arithmetic. This rejects width as the
next lever and keeps local attention conditional on a harder motion dataset and
temporal-stability loss demonstrating a need for it.

At 960×540 input, one 96-channel FP16 bottleneck state is about 6.2 MB. That is
small enough to be explicit in the native resource contract and large enough
that blindly retaining many frames is not free. Start with one recurrent state
and current-frame features; add more history only when a sequence ablation
justifies it.

## Evaluation gate

Static PSNR remains the radiometric metric, but it cannot be the only release
gate: the present 4-spp input and ReSTIR+SVGF have similar PSNR while their
SSIM differs dramatically. Every architecture run reports linear/compressed
MSE and PSNR, SSIM, parameter count, sustained GPU latency, p90, peak working
memory, and dispatch count. The temporal stage additionally needs warped
error on valid history, disocclusion error, flicker/temporal-SSIM, and camera
cut recovery. Add
[CGVQM](https://www.intel.com/content/www/us/en/developer/articles/technical/cgvqm-d-computer-graphics-video-quality.html)
as a perceptual sequence metric once those captures exist: its graphics-video
dataset specifically includes neural supersampling, path tracing, denoising,
and frame interpolation artifacts, and reports that conventional
full-reference metrics correlate poorly with viewers on that domain. It
supplements rather than replaces radiometric error and scene-specific failure
maps. Sequence-level metrics must be computed before any claim of DLSS-like
quality.

The decisive experiment is not “U-Net versus transformer” on static crops. It
is a 2×2 ablation on identical motion sequences: convolutional versus local
attention fusion, each without and with valid reprojected history. That tells
us whether attention contributes beyond the evidence supplied by time.
