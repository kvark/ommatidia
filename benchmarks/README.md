# Curated comparison suite

Six fixed seed-10000 scenes cover a canopy shadow, contact lighting, small
emitters, textured gloss, a dark interior, and a hard shadow boundary. Every
arm reads the same 4-spp independent path trace and the same 4,096-spp
canonical record. The ReSTIR+SVGF dataset copies those canonical bytes and
verifies its regenerated G-buffer against them.

The OIDN arms use the official Open Image Denoise `RT` filter with HDR color,
albedo, normal, `cleanAux`, and both high and fast quality modes. OIDN does not
upscale, so its primary arms denoise at input resolution and then use the same
texel-center-aligned 2x bilinear reconstruction as the ordinary baseline.
`oidn-output-high` instead expands the noisy color first and is retained as a
negative control; it must not be described as native OIDN super-resolution.

Download OIDN 2.4.1 from its official release, then run:

```sh
curl -L --create-dirs -o data/rich-4spp-validation-128.omd \
  https://huggingface.co/datasets/mad-bot/ommatidia/resolve/main/benchmarks/rich-4spp-validation-128.omd
curl -L --create-dirs -o runs/rich-kernel-b16-demod025.safetensors \
  https://huggingface.co/mad-bot/ommatidia/resolve/main/experiments/rich-kernel-b16-demod025/model.safetensors
curl -L --create-dirs -o runs/rich-kernel-b16-demod025.ron \
  https://huggingface.co/mad-bot/ommatidia/resolve/main/experiments/rich-kernel-b16-demod025/config.ron
```

Then run the suite:

```sh
benchmarks/run-curated.sh /path/to/oidnDenoise 0x744c runs/comparison-suite \
  runs/rich-kernel-b16-demod025
```

The runner writes linear-HDR-derived PNGs, provenance metadata, one CSV row per
scene and method, and `summary.csv` with the arithmetic mean over the six
scenes. Metrics are compressed-space PSNR, the project's legacy SSIM, relative
MSE, detail retention, mean linear luminance ratio, and 16x16 block-average
error for the visible low-frequency mottling. Lower is better for the two MSE
columns; ratios should be read against one.

For a device-resident OIDN timing without file I/O or filter construction, use
the official benchmark from the same package:

```sh
oidnBenchmark --device 0 --run 'RT\.hdr_calb_cnrm\..*' \
  --size 960 540 --quality high --type half --buffer device --inplace -n 40
```
