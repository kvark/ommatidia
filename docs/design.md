# One recurrent reconstruction model

The maintained model is a lobe-separated radiance-residual U-Net. There is one
graph builder, one trainer, and one GPU runtime. Config version 3 rejects the
retired selector versions. Earlier diffusion, kernel, field/relighting and C ABI
implementations are recoverable at Git commit `b838674` (also the local branch
`archive/experiments-2026-09-25`); they are not compatibility modes.

## Architecture

The GPU constructs five geometry-guided spatial scales and reprojects the
previous diffuse/specular estimates. Each lobe has its own validity, reactive
rejection, age and moments. History and final reconstruction remain linear HDR.

A three-level local convolutional U-Net sees the encoded spatial scales,
reprojected history, raw diffuse/specular samples, sample offsets, surface
normal/depth/albedo/roughness, variance, age and validity. Subpixel packing keeps
convolutions at input resolution. No image-wide normalization, diffusion,
transformer branch, generated texture, or ground-truth input.

The six-channel subpixel head predicts radiance residuals:
`max(guide + 0.1 * (guide + 1/exposure) * residual, 0)`.
A zero head exactly reproduces the fixed spatial/history guide. Unlike a convex
candidate selector, this decoder can recover detail outside its filtered
candidates' range. The default width is 16: 188,160 parameters at 2x scale.
Known material albedo and emission are composed only at the end:
`RGB = albedo * diffuse + specular + emission`.

Training uses the same recurrence, warming history with the current model's
own predictions. Two-frame BPTT differentiates through bilinear radiance
reprojection; geometry, rejection maps, ages and moments are detached.
The objective combines compressed displayed-RGB MSE, a small fixed-exposure
linear RGB term, coarse linear structure and valid-history temporal changes.
There are no selector labels or target-normalized brightness weights.
The trainer reuses native GPU preparation; the CPU only expands differentiable
history-gather maps and supplies the numerical reference implementation.
Warmup advances the native state without downloading unused displayed RGB.

## Why this scope

[REGEN](https://github.com/stefanos50/REGEN) uses paired supervision to train a
lightweight photorealism-enhancement model. Its appearance-translation objective
is different from recovering a particular path-traced scene.

[OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) documents a same-resolution
generative renderer, not DLSS super-resolution or ray reconstruction. Useful
lessons here are residual decoding, reprojected feedback, and independent
numerical parity checks. Copying its large transformer or style/noise inputs
would not establish reconstruction quality on our data.

Our choice is an engineering judgment based on the old selector's bounded
output and observed loss of detail, not a claim to reproduce either project.
Capacity changes must earn their place in this one implementation.

## Runtime

`ommatidia::transport::native::Native::new(context, config, low_extent)` accepts
the host's existing Blade context. Input `Ray` and `Surface` buffers contain only
renderer observations; `Target` is a separate training type.

Submit `record_prepare`, then the Meganeura session, then `record_resolve` in
queue order. Output is linear RGBA f32. Reset on camera cuts; recreate on extent
changes. `process` is the synchronous CPU-upload/readback evaluation helper,
not a frame-time benchmark. Old `Upscaler` and the old C ABI were removed.

## Correctness boundary

CPU/WGSL preparation parity covers raw features, 12-frame recurrence, invalid
history, resets and HDR. The actual two-frame training graph is checked against
Meganeura's f64 reference for its loss and every parameter gradient, including
fused/unfused lowerings. A training-and-reload test must reduce loss.

Blade is pinned to `fbb4f28`, Meganeura to `0dbfcc0`, and Naga to `323acfb`.
Sibling Cargo patches allow local development; run manifests record their actual
revisions and dirty diffs. The current Naga SPIR-V backend still triggers
`VUID-StandaloneSpirv-None-10684` for Workgroup array layout. Numerical tests
pass on the tested adapters, but validation is not clean. The run recorder
treats this as failure even when the child exits zero. This remains a release
blocker; optimized quality measurements do not waive it.
