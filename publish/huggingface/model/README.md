---
license: mit
library_name: meganeura
pipeline_tag: image-to-image
datasets:
- mad-bot/ommatidia
tags:
- ray-tracing
- denoising
- upscaling
- vulkan
- metal
---

# Ommatidium

Ommatidium is a portable neural denoiser and 2× upscaler for real-time ray and
path tracing. The current checkpoint accepts independent sparse-path radiance
and the renderer's depth, world normal, diffuse albedo, specular F0, and
roughness buffers. It does not require ReSTIR or an upstream denoiser.

- **Source and integration:** https://github.com/kvark/ommatidia
- **Training and validation data:** https://huggingface.co/datasets/mad-bot/ommatidia
- **Runtime:** Meganeura on the graphics context already owned by the host
- **Backends:** Vulkan and Metal; the published performance result is Vulkan

## Checkpoint

The release contains `model.safetensors`, `config.ron`, and a machine-readable
`manifest.json`. This is a direct, single-frame b8 U-Net with three resolution
levels, one residual block per level, 73,808 parameters, and a 2× output scale.
It predicts a small sub-pixel residual over a joint bilateral reconstruction
that uses output-resolution primary surfaces to place silhouettes exactly.

The low-resolution input groups are:

1. linear RGB radiance from independent sparse paths;
2. depth;
3. world-space normal;
4. diffuse albedo;
5. specular F0; and
6. roughness.

Use the Ommatidium loader rather than constructing tensors manually. It applies
the checkpoint's stored radiance transform, guided reconstruction, and residual
gain and keeps the complete path on the host's GPU.

This checkpoint additionally requires output-resolution depth, world normal,
and diffuse albedo. They guide a 5×5 gather in unpack and are not fed through
the network. A host with no output-resolution primary surfaces should pin
`v0.2.0`, whose low-resolution-only reconstruction contract remains supported.

## Results

On 128 non-overlapping crops from a separate 128-scene, seed-10000 validation set with 4-spp
128×128 independent path inputs and 4,096-spp 256×256 canonical references:

| Reconstruction | MSE | PSNR | SSIM |
|---|---:|---:|---:|
| nearest 2× | 0.004367 | 23.60 dB | 0.4466 |
| bilinear 2× | 0.002233 | 26.51 dB | 0.5776 |
| v0.3 HR-guided 5×5, no network | 0.000346 | 34.61 dB | 0.9543 |
| tuned HR-guided 5×5, no network | 0.000337 | 34.72 dB | 0.9574 |
| tuned HR-guided + b8 residual | **0.000335** | **34.74 dB** | **0.9575** |

The fixed-cost coefficient tuning adds 0.11 dB and 0.0031 SSIM over the v0.3
base; the matching b8 residual adds 0.02 dB after 2,000 steps. A
960×540 → 1920×1080 trace on a Radeon RX 7900 XT measured 8.83 ms median and
8.90 ms p90. Its isolated stages are 0.79 ms pack, 7.22 ms network, and
0.88 ms unpack. Ray tracing, the optional
output-resolution primary-surface pass, and display post-processing are
excluded.

ReSTIR+SVGF is retained in the dataset as a matched comparison control, not as
training input for this checkpoint. Blade's ReSTIR output looks darker than the
full canonical target because its real-time estimator covers direct environment
and first-hit emission while the canonical target includes indirect bounces;
against a transport-matched direct reference its measured HDR energy is 99%.

## Intended use and limitations

This is an early spatial research checkpoint intended for renderer integration.
It was trained on small procedural Blade scenes and may fail on authored
geometry, materials, lighting, resolutions, or renderers outside that narrow
distribution. It has no motion vectors or temporal history, cannot recover
disoccluded samples from prior frames, and should not be described as DLSS-like
quality yet.

Pin the `v0.3.1` Hub revision, or its exact commit, in applications. Do not
download mutable `main` for a shipped build.

The weights are released under the MIT license.
