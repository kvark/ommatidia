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

Ommatidium is a portable neural denoiser and 2× upscaler for real-time ray
tracing. This first checkpoint replaces Blade's variance-guided SVGF pass: it
accepts raw ReSTIR radiance and the renderer's depth, normal, diffuse albedo,
specular F0, and roughness buffers, then reconstructs a high-resolution frame
through [Meganeura](https://github.com/kvark/meganeura).

- **Source and integration:** https://github.com/kvark/ommatidia
- **Training and validation data:** https://huggingface.co/datasets/mad-bot/ommatidia
- **Runtime:** Meganeura on a graphics context shared with Blade
- **Backends:** Vulkan and Metal; the published performance result is Vulkan

## Checkpoint

The release contains `model.safetensors`, `config.ron`, and a machine-readable
`manifest.json`. The configuration is a direct, single-frame base-24 U-Net with
three resolution levels, one residual block per level, 649,200 parameters, and
a 2× output scale. It has no temporal history yet.

The input contract is six low-resolution plane groups in this order:

1. RGB radiance
2. depth
3. world-space normal
4. diffuse albedo
5. specular F0
6. roughness

Use the Ommatidium loader rather than constructing the tensor manually: it
applies the checkpoint's stored radiance compression and residual gain and
keeps the prediction on the shared GPU.

## Results

On the matched 128-scene validation capture:

| Reconstruction | Error |
|---|---:|
| Raw ReSTIR, nearest 2× | 0.005748 |
| Blade SVGF, nearest 2× | 0.004284 |
| Ommatidium from raw ReSTIR | **0.002876** |

That is 3.01 dB over raw nearest reconstruction and 1.73 dB over the Blade
filter it replaces. The canonical references are byte-identical between the
raw and SVGF captures.

At the actual rectangular deployment shape, repeated sustained benchmarks
measure **19.54–20.91 ms** for 960×540 input / 1920×1080 output on an idle
Radeon RX 7900 XT with RADV. This includes Ommatidium's texture pack, the 104.1
GFLOP Meganeura network, unpack, and queue submissions; the ray tracer and
display post-processing are not included. The benchmark also reports tail
latency; a recent 40-frame run measured a 29.95 ms p90.

## Intended use and limitations

This is an early research checkpoint intended for Blade integration and for
developing portable neural reconstruction runtimes. It was trained entirely
on procedural Blade scenes at 128×128 input and 256×256 reference resolution.
It may fail on geometry, materials, lighting distributions, resolutions, or
renderers outside that training distribution. It is spatial-only, so it cannot
recover information from prior frames or guarantee temporal stability.

Pin the `v0.1.0` Hub revision, or its exact commit, in applications. Do not
download mutable `main` for a shipped build.

## Revisions

- Ommatidium: `7f08f025a3150c355643af513bc2825d88441520`
- Meganeura upstream integration: `256b906`
- Blade upstream integration: `3a8895a`
- Dataset: `mad-bot/ommatidia@v0.1.0`

The weights are released under the MIT license.
