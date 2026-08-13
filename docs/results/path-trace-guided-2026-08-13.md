# Independent-path guided reconstruction, 2026-08-13

## Question

Can the spatial model replace Blade's denoiser when its source is an ordinary
sparse path trace, rather than ReSTIR or ReSTIR followed by SVGF?

The earlier checkpoint could not answer that question. It was trained on
ReSTIR output and predicted a residual over nearest-neighbour upsampling. This
run uses independent paths throughout and scores every deterministic
reconstruction base before attributing a gain to the network.

## Controlled setup

- Training: 2,400 procedurally generated scenes, four independent three-bounce
  paths per 128×128 input pixel.
- Target: 4,096 eight-bounce paths per 256×256 reference pixel.
- Validation: a separate 128-scene set generated with seed 10000; 76 fixed
  64×64 low-resolution crops are scored.
- Conditioning: sparse radiance, depth, world normal, diffuse albedo, specular
  F0, and roughness.
- Models: compute-matched direct residual U-Nets with three levels and one
  block per level. The b24 arm has 649,200 parameters and trained for 6,000
  steps; the b8 arm has 73,808 parameters and trained for 4,000 steps.
- Metrics: MSE and PSNR in compressed linear-radiance space, plus SSIM.

## Reconstruction-base result

| reconstruction | MSE | PSNR | SSIM |
|---|---:|---:|---:|
| nearest | 0.004385 | 23.58 dB | 0.4593 |
| texel-centre bilinear | 0.002261 | 26.46 dB | 0.5864 |
| depth/normal/albedo guide + bilinear | 0.000391 | 34.08 dB | 0.9473 |
| guide + learned b8 residual | 0.000387 | 34.12 dB | 0.9474 |
| guide + learned b24 residual | 0.000385 | **34.15 dB** | **0.9476** |

The decisive gain is renderer-aware reconstruction, not a larger model. The
guide filters once at input resolution with spatial, encoded-depth, normal,
and albedo weights, then the unpack stage performs exact texel-centre bilinear
upsampling. CPU training and the WGSL runtime implement the same operation; a
GPU reference test bounds their worst relative difference below 0.1%.

The b8 learned correction adds 0.04 dB over that strong base, and b24 adds
0.07 dB. On training-side validation b8 reached 33.10 dB at step 1,000, 33.11
at 2,000, and 33.12 at both 3,000 and 4,000. B24 reached 33.13 at step 2,000,
33.15 at 4,000, and 33.16 at 6,000. Both had plateaued; the 0.03 dB external
validation difference does not justify b24's 2.67× end-to-end latency.

## ReSTIR+SVGF control

The matched ReSTIR+SVGF dataset is scored only as a comparison input, never as
the source for Ommatidium training:

| reconstruction | MSE | PSNR | SSIM |
|---|---:|---:|---:|
| nearest | — | 23.61 dB | 0.8776 |
| bilinear | — | 23.86 dB | 0.8893 |
| guided | — | 23.96 dB | 0.8920 |

Applying the independent-path network to this control would be an
out-of-distribution test and is deliberately not reported as product quality.

## Runtime trace

On an idle Radeon RX 7900 XT, a 960×540 input reconstructed to 1920×1080 with
b8 in 7.76 ms median and 7.94 ms p90 over 40 timed frames after 20 warmups.
Isolated GPU stages measured 0.76 ms for pack (including the guide), 6.99 ms
for the model, and 0.12 ms for unpack. B24 measured 20.71 ms median and 20.99
ms p90, split into 0.79 ms pack, 19.45 ms model, and 0.12 ms unpack. Ray
tracing and display post-processing are outside those numbers.

The guide is therefore not the latency problem, and b8 is the selected spatial
checkpoint. The next major quality experiment should add valid reprojected
history rather than another static backbone family; a filter-only mode remains
a useful lower bound when applications prefer 0.76 ms to a 0.04 dB learned
gain.

## Energy interpretation

Blade's real-time ReSTIR estimator and the full canonical path tracer do not
currently estimate the same transport. Against a direct-only canonical
reference, no-reuse and pairwise-reuse HDR energy are 99.1% and 99.0%; the
remaining visibly dark faces appear when the canonical renderer is allowed
secondary bounces. They are indirect illumination that real-time mode never
traces, not energy lost by reservoir normalization. A fair direct-light
regression must set canonical `max_bounces` to zero; a full-path comparison
requires a separate GI estimator.
