# Reference quality

The training target is only ground truth to the extent that its residual Monte
Carlo noise is negligible. Ommatidium now treats reference sample count as a
measured quality setting, not a magic constant.

On the procedural seed-7 scene at 256×256, Blade's 4,096-spp path trace differs
from an independent 16,384-spp accumulation by compressed-space MSE
`0.00000613`, or **52.13 dB PSNR**. The old 1,024-spp setting scores 45.17 dB
against the same target. The 4,096-spp image is visually clean at native scale;
it is the new default for generated references. Long captures submit in bounded
chunks so transient acceleration structures cannot exhaust device memory.
Reference paths allow eight bounces with Russian roulette after the fourth;
the sparse input remains a practical three-bounce trace. This keeps a clean
target from also inheriting the real-time path-depth truncation.

This is a convergence check, not an independent correctness proof. The next
reference audit should export a fixed Cornell/material scene to
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html)
and
compare linear EXR output under matched camera, geometry, metallic-roughness
conversion, emitter radiance, path depth, and sample count. Mitsuba's basic
path integrator uses emitter/BSDF MIS and arbitrary-length paths, making it a
better comparison than a raster image or a second setting of Blade itself.

Reference acceptance should use two independent seeds. Their mutual error is
an estimate of the noise floor; model improvements smaller than that floor are
not evidence. Bright specular pixels also need an HDR-aware view in addition to
the project's bounded `x/(1+x)` PSNR and SSIM.

As a noise-floor calibration rather than a matched correctness test,
`scripts/mitsuba_convergence.py` renders Mitsuba 3.9.1's Cornell scene with its
eight-depth path integrator. At 128×128, seed 7, a 4,096-spp render is 58.65 dB
from the 16,384-spp render in the same bounded space; 1,024 spp is 53.04 dB.
Blade's procedural seed-7 scene is harder and measures 52.13 dB at 4,096 spp,
so the current target is in a production-tracer noise regime but still has a
higher residual floor. This comparison does not excuse a renderer bias: the
matched exported scene above remains required.
