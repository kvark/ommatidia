# Curated OIDN comparison — 2026-08-22

## Contract

Six fixed seed-10000 Blade scenes cover canopy and hard shadows, glossy contact
lighting, a small local emitter, textured gloss, and a dark interior. Each arm
uses the same independent 4-spp path trace at 128x128 and the exact same
4,096-spp, eight-bounce canonical record at 256x256. The checked-in artifacts
and per-scene CSV are in [`docs/comparison-suite`](../comparison-suite).

[Open Image Denoise 2.4.1](https://github.com/RenderKit/oidn/releases/tag/v2.4.1)
is invoked through its official `oidnDenoise` program with the `RT` HDR filter,
clean albedo and normal AOVs, and `cleanAux`. OIDN is a spatial denoiser, not an
upscaler. Its primary arms therefore denoise the 128x128 input and then use the
same texel-centre 2x bilinear reconstruction as the ordinary baseline. Running
OIDN after expanding the noisy input is retained as a negative control.

ReSTIR+SVGF is generated separately, verifies its canonical bytes and G-buffer
against the path-traced dataset, and is bilinearly expanded for the table. It
is never an Ommatidium input.

## Quality

Arithmetic mean over the six full frames:

| method | PSNR | SSIM | relMSE | detail | energy | low-frequency PSNR |
|---|---:|---:|---:|---:|---:|---:|
| bilinear 2x | 19.876 dB | 0.4101 | 3.2426 | 285% | 0.990 | 29.235 dB |
| **Ommatidium 2x** | **29.255 dB** | 0.8409 | **0.0359** | **77%** | 0.935 | **34.476 dB** |
| OIDN High, then bilinear 2x | 28.047 dB | **0.8731** | 2.2672 | 49% | 0.995 | 34.119 dB |
| OIDN Fast, then bilinear 2x | 27.744 dB | 0.8640 | 2.1883 | 50% | 0.996 | 33.452 dB |
| ReSTIR+SVGF control, bilinear 2x | 22.849 dB | 0.7395 | 1.9791 | 59% | 0.875 | 24.106 dB |

Ommatidium wins mean PSNR by 1.21 dB over OIDN High, retains substantially more
detail, and is orders of magnitude safer in dark regions by relative MSE. OIDN
wins SSIM because it produces a much smoother image while retaining only 49%
of canonical gradient energy. This is exactly why SSIM is not a sufficient
ranking here. Its fixed constants also make dark, low-variance blocks unusually
forgiving.

The screenshots expose the remaining failure more clearly than aggregate PSNR:
Ommatidium has broad mottling and carries only 93.5% of canonical mean
luminance. The new 16x16 block-average metric isolates that component. It beats
OIDN High by only 0.36 dB there, far less than its 1.21 dB ordinary-PSNR lead.
PSNR remains a useful regression score, but acceptance now needs PSNR, relMSE,
detail retention, linear energy, low-frequency error, and the fixed images.
NVIDIA's [FLIP](https://research.nvidia.com/publication/2020-07_FLIP) is a good
candidate for a later perceptual diagnostic; it should complement rather than
replace the radiometric metrics.

## Speed

All measurements below are from the same Radeon RX 7900 XT. They state their
contracts because OIDN is not doing Ommatidium's 2x reconstruction.

| path | measured work | time |
|---|---|---:|
| Ommatidium b16 | 960x540 input to 1920x1080 output, pack + model + unpack + submissions | 16.10 ms median, 16.37 ms p90 |
| OIDN High | resident half-precision 960x540 HDR + clean AOV denoise; bilinear 2x excluded | 6.64 ms |
| OIDN Fast | same input contract; bilinear 2x excluded | 2.00 ms |
| OIDN High negative control | resident 1920x1080 denoise | 27.00 ms |
| OIDN Fast negative control | resident 1920x1080 denoise | 6.96 ms |
| Blade ReSTIR+SVGF suite capture | eight settled 128x128 frames plus scene work and readback | 125 ms/scene, about 15.6 ms/frame |
| Blade canonical reference | 4-spp sparse input plus 4,096-spp 256x256 reference | 0.9 s/scene after setup |

OIDN timings come from its official `oidnBenchmark`, with device buffers,
in-place output, 20 warmups, and 40 measured runs. They exclude filter creation,
file I/O, the 2x reconstruction, and Vulkan/HIP interop. The ReSTIR and
canonical rows are offline generator throughput, not isolated steady-state
renderer latency, and should not be compared to post-process latency as though
they were the same operation.

The detailed Meganeura trace accounts for the Ommatidium result: the model is
14.79 ms of the frame, across 80 dispatches in 71 barrier groups. Convolution is
12.15 ms (82.4%), pointwise work 1.41 ms (9.5%), normalization 0.90 ms (6.1%),
and data movement 0.30 ms (2.0%). Instrumentation changed wall time by 3.2% and
GPU timestamps covered 96.8% of profiled time, so this trace is sufficiently
lightweight for attribution. The separate integration measurement was 0.06 ms
pack, 14.77 ms model, and 0.95 ms unpack.

## ReSTIR/SVGF energy fix

The old A-trous luminance test used only the centre pixel's variance and then
renormalized surviving neighbours. The gate was asymmetric: a noisy bright
pixel could accept a dark neighbour which rejected it in the reverse
direction, destroying energy over repeated passes.

Blade now uses the maximum variance of each pixel pair and applies a symmetric
pairwise delta, leaving rejected weight on the centre. The new HDR regression
measures filtered/raw ReSTIR energy at 0.9990. A separate transport-matched
regression measures ReSTIR/canonical direct energy at 1.0272 with no unexpected
black pixels. The suite's remaining 0.875 energy against the eight-bounce
canonical is therefore expected missing indirect transport: Blade's real-time
path estimates direct illumination, while the reference includes secondary
fill.

## First low-frequency experiments

Three controlled attempts were rejected rather than added to the runtime:

| experiment | PSNR | SSIM | relMSE | detail | energy | low-frequency PSNR |
|---|---:|---:|---:|---:|---:|---:|
| retained checkpoint | **29.255** | **0.8409** | **0.0359** | 77% | **0.935** | 34.476 |
| extra 8x8 block-average training loss | 28.797 | 0.8184 | 0.0429 | 82% | 0.929 | 34.294 |
| gather in linear rather than compressed radiance | 28.985 | 0.8202 | 0.0463 | **83%** | 0.924 | **34.641** |

A runtime-available global exposure match raised low-frequency PSNR to 34.790
and ordinary PSNR to 29.340, but worsened SSIM and relMSE; it fixes a scalar,
not the spatial mottling. A local low-frequency correction from the raw input
failed badly because sparse fireflies contaminate precisely the guide being
used to correct the result.

The conclusion is useful: the current frame does not contain a trustworthy
low-frequency estimate which a different static loss can reveal. OIDN obtains
smoothness by discarding detail; Ommatidium preserves more evidence and leaves
some of its uncertainty visible. The next credible quality step is the planned
motion-reprojected frame history, providing genuinely new samples, followed by
training and scoring on whole sequences. No failed experimental option or new
Meganeura operation remains in the production path.
