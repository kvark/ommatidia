# Lessons from the DLSS 4 and DLSS 5 reports

This note separates two systems that solve different problems. NVIDIA's
[DLSS 4 report](https://research.nvidia.com/labs/adlr/DLSS4/) describes
reconstruction: joint ray denoising, anti-aliasing, and super-resolution toward
a renderer-defined reference. The
[DLSS 5 report](https://research.nvidia.com/labs/adlr/DLSS5/)
describes renderer-conditioned generation: changing displayed appearance
toward a photographic prior, not estimating the converged renderer output.
Ommatidium is presently a reconstruction system. A useful technique may cross
that boundary, but its objective must not do so accidentally.

## What the reports establish

### Reconstruction is spatial and temporal together

DLSS 4 explicitly replaces the traditional sequence “denoise low-resolution
lighting, then upscale” with one high-resolution reconstruction. Its stated
reason is important: a low-resolution denoiser irreversibly limits shading
detail before the upscaler sees it. The model has to handle sparse paths,
aliasing, motion, disocclusion, and different white-noise, blue-noise, QMC, and
ReSTIR sampling correlations as one problem.

Its transformer result is meaningful but not a portable architecture recipe.
NVIDIA reports four times the arithmetic and twice the parameters of its CNN in
a similar frame budget by co-designing the topology with vertically fused
kernels, on-chip intermediates, tensor cores, and FP8. The public report does
not disclose the topology. Ommatidium's previous compute-matched
shifted-window probe tied its convolutional baseline and ran 7.5% slower on an
RX 7900 XT. Repeating a model-family label without the stronger temporal input
or the corresponding kernel implementation would not reproduce DLSS 4.

### Causal state, not frame chunks

DLSS 5 accepts one rendered frame, motion, carried state, and controls, and
returns one deterministic frame without future input. The report rejects
chunked video inference because its latency and working set grow with the
chunk. This supports Ommatidium's explicit resettable, ping-ponged history
contract and argues for one bounded learned state at a coarse resolution. It
does not support buffering a transformer window of future or jointly processed
frames.

### Renderer grounding can be training-only

DLSS 5 ships an RGB-plus-motion interface but supervises albedo, normals, and
lighting attributes during training. That is a clever way to constrain a
generative model without making every G-buffer a deployment requirement. It
does not imply that Ommatidium should discard its runtime G-buffer: exact
output-resolution surfaces currently provide most of its reconstruction gain
and make disocclusion rejection observable. It does suggest two practices:

- treat renderer attributes as named consistency objectives instead of hoping
  one RGB metric preserves every property; and
- ablate whether each runtime plane remains necessary after a stronger model
  exists, rather than making the interface permanent by assumption.

### Pixel-space, one-step inference is the safe real-time direction

DLSS 5 uses a one-step deterministic pixel-space diffusion model. Pixel space
avoids the edge, text, and fine-detail drift of a lossy learned autoencoder;
one step avoids iterative latency and stochastic output. Its task is still
generative and its 154-million-parameter transformer is reported at about
8 ms and 731 MB for 3840x2160 on an RTX 5090, using mostly FP8 and Blackwell
TMA.

Ommatidium already measured the relevant reconstruction ablation: direct
regression beat its diffusion objective by a wide margin, and more diffusion
steps made the result worse. Strong renderer conditioning leaves one physical
answer rather than a useful distribution to sample. We should retain direct,
single-pass, pixel-space reconstruction and not revive iterative diffusion
because DLSS 5 uses the word.

### Tone and structure are different failure modes

DLSS 5 exposes independently learned tone and structure strengths, described
as low- and high-spatial-frequency behavior, plus per-pixel masks. This maps
closely to Ommatidium's current failure: the split-radiance estimator preserves
edges and material detail, while its broad illumination varies. A correction
trained against every residual texel becomes overconfident and loses SSIM; a
post-hoc 20% correction is safer but exposes a training mismatch.

The first direct experiment from this report was therefore a training-only
band-limited target for the existing three-channel correction. It asked the
network to learn broad illumination error while leaving all output-resolution
structure to the fixed split-radiance estimator. It changed neither the
inference graph nor Meganeura's shader groups.

The controlled result rejected it. On the 480-crop, mesh-disjoint ABO selector,
a 4x4 target at its structure-safe 12.5% correction reached 33.00 dB / 0.9509
SSIM / 37.24 dB low-frequency PSNR. The existing ordinary target at its frozen
20% correction reaches 33.12 / 0.9513 / 37.35. On 240 furnished-interior crops,
the band-limited target reached 37.23 / 0.9606 / 41.76, while the ordinary one
reached 37.25 / 0.9604 / 41.81 and also won relative, worst-crop, and temporal
error. An 8x8 target was worse again. The SSIM-only interior difference is too
small to outweigh every radiometric and temporal loss, so the target option was
removed rather than becoming permanent training surface. Separating tone from
structure remains useful; a box-filtered label was simply not the mechanism.

### Metrics have jobs, not authority

DLSS 5 deliberately does not call LPIPS or DINOv2 an image-quality score. It
uses them for content/identity alignment, renderer-attribute metrics for
structure alignment, and a blinded pairwise study for photographic realism.
It also compares against spending the same total time on four times as many
rays. That division is more useful than selecting a fashionable scalar.

For physical reconstruction, Ommatidium should continue to anchor on
renderer-reference PSNR and relative error; use SSIM, detail retention,
low-frequency error, edge/attribute consistency, and temporal reprojection to
name particular failures; add a perceptual display-space metric and blinded
comparisons for visual ranking; and publish matched-total-frame-time ray
baselines. DINO distance would measure whether content moved, not whether a
denoiser is correct.

### Missing input evidence cannot be architected away

DLSS 5 explicitly calls out input quality and train/deployment distribution
shift as limitations: missing or corrupted structure may be preserved, and
different shading conventions, material distributions, scene statistics, or
artifact patterns can degrade the result. This warning applies even more
strongly to a physical reconstructor. It supports Ommatidium's exact
output-surface option, split-safe real-scene catalogs, independent high-sample
targets, and sampler diversity. It also explains why a larger spatial model is
unlikely to fix low-frequency light transport that its current frame never
observed; causal history or more paths have to provide that evidence.

### Preserve the physical estimator before learning its confidence

The report-driven probe exposed a more basic issue than model family. A
predicted kernel was documented as a convex combination of measured radiance,
but training and runtime averaged `compress(radiance)` and decompressed the
result. The compression is concave, so Jensen's inequality makes every broad
filter systematically dark. New kernel checkpoints now average linear
radiance and compress only the result; old sidecars retain their historical
path for compatible inference. A two-tap 0/4-radiance unit case produces the
physical mean 2.0 on the new path and 0.667 on the old one.

Linear averaging alone exposes Monte Carlo outliers. The useful compact
translation of DLSS 5's tone/structure separation is therefore not a second
network or a filtered label: it is a learned mix between the physical sample
gather and the existing deterministic output-resolution guide. That adds four
gate channels to the existing head and uses existing pointwise operations and
the existing unpack shader. It does not add a Meganeura operation or shader
family.

### Motion has a capture contract as well as a texture format

Exact output-resolution vectors are only exact if the renderer still remembers
the preceding displayed frame. Blade prepares the same camera for every path
accumulation sample and ReSTIR settling iteration. Reading its G-buffer at the
end therefore finds both ping-ponged camera slots holding the current camera
and collapses camera motion to numerical zero. Ommatidium's capture harness now
snapshots primary surfaces after the first iteration, then continues
accumulating radiance or reservoirs. Fixed LavaPipe sequences cover motion,
projection jitter, multiple canonical samples, ReSTIR settling, and both
G-buffer resolutions.

Even correct geometric motion is not correct radiance motion for reflections,
shadows, particles, or newly visible lighting. This is exactly the side-input
failure DLSS 4 describes. Surface rejection and a learned validity-aware mix
remain necessary; an exact vector should not be mistaken for permission to
reuse every shading value.

## Concrete vNext architecture gate

The evidence now supports a narrower target than “replace the U-Net with a
transformer”:

1. retain the three-level b16 convolutional encoder/decoder and its positive
   7x7 per-phase kernel head;
2. keep reconstruction physical: average demodulated linear path samples,
   restore exact output albedo, and learn confidence against the deterministic
   HR guide;
3. retain the surface-validated previous output as the full-resolution causal
   state, then add at most one small f16 learned state at the bottleneck. At
   one-quarter input width and height, 16 channels cost about 2 MiB ping-ponged
   for a 960x540 input rather than another full-resolution feature pyramid;
4. update that state with a gated convolution first. A local
   current-query/history-key attention block may replace only this bottleneck
   update if it wins at matched complete-frame latency and memory; and
5. emit guide/history confidence together with the state update. Exact motion,
   surface validity, a renderer reactive/disocclusion signal, jitter, and
   exposure are evidence for those gates, not values to blend blindly.

Training must roll this state forward from a reset for an entire short
sequence. The current detached teacher reconstructs the immediately preceding
frame from a reset path; it does not reproduce its own deployment distribution
after several recurrent steps. Four-to-sixteen-frame rollouts, camera cuts,
exposure changes, animated occluders, reflection/shadow changes, and mixed
white/blue/QMC input correlations are therefore prerequisites for judging the
state block. Primary compressed/display error, linear energy, relative error,
surface attributes, low-frequency error, and motion-compensated temporal error
remain separate acceptance axes. LPIPS or DINO can flag content drift in a
future appearance mode, but cannot certify physical reconstruction.

## Ordered consequences for Ommatidium

1. Test long-range context cheaply by adding one coarse U-Net level before
   retaining any attention-specific graph surface. Compare at matched data and
   schedule, then veto it on frame time and memory if its quality gain is small.
2. Add exposure changes, cuts, animated occlusion, and multiple sampling
   patterns to sequence data. Exact output-resolution motion now makes
   projection-jittered moving captures valid, and reset frames are sampled at
   their natural sequence frequency; diversity of sampling correlation is
   still a stated DLSS 4 requirement and is underrepresented here.
3. Learn one bounded coarse recurrent feature state, with explicit validity and
   reset. Compare gated convolution against current-query/history-key local
   attention at matched frame time; do not replace the full U-Net with a ViT.
4. Profile complete frame time and memory, including packing, reprojection,
   model, unpacking, and any primary-surface pass. Compare the same budget spent
   on more paths.
5. Keep generative appearance enhancement as a separate future mode with
   explicit strength/masks and different training/evaluation. It must never be
   presented as a more physically accurate denoiser.
