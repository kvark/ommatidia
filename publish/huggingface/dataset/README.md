---
license: cc
pretty_name: Ommatidium synthetic ray-tracing reconstruction data
size_categories:
- 1K<n<10K
task_categories:
- image-to-image
tags:
- synthetic
- ray-tracing
- denoising
- upscaling
viewer: false
---

# Ommatidium dataset

Synthetic renderer-native training and validation data for
[Ommatidium](https://github.com/kvark/ommatidia), generated with
[Blade](https://github.com/kvark/blade).

- **Checkpoint:** https://huggingface.co/mad-bot/ommatidia
- **Format and training code:** https://github.com/kvark/ommatidia
- **Generator:** the `ommatidia-data` workspace crate

## Contents

| File | Input source | Samples | Purpose |
|---|---|---:|---|
| `benchmarks/rich-4spp-validation-128.omd` | independent path trace, 4 spp + rich shadows/materials + HR primary surfaces | 128 | six-scene OIDN/ReSTIR comparison source |
| `data/blade-path-trace/4spp-hr-gbuffer-train.omd` | independent path trace, 4 spp + HR primary surfaces | 2,400 | v0.3 spatial training set |
| `benchmarks/path-trace-4spp-hr-gbuffer-validation.omd` | independent path trace, 4 spp + HR primary surfaces | 128 | v0.3 separate seed-10000 validation |
| `ablations/path-trace-4spp-static-train.omd` | independent path trace, 4 spp | 2,400 | v0.2 low-resolution-guide training set |
| `benchmarks/path-trace-4spp-static-validation.omd` | independent path trace, 4 spp | 128 | separate seed-10000 validation |
| `benchmarks/path-reference-svgf-validation.omd` | Blade ReSTIR+SVGF | 128 | comparison control with matched references |
| `data/blade-path-trace/1spp-train.omd` | independent path trace, 1 spp | 2,400 | lower-ray-budget experiment |
| `benchmarks/path-trace-1spp-validation.omd` | independent path trace, 1 spp | 128 | lower-ray-budget validation |
| `data/blade-restir/train.omd` | raw Blade ReSTIR | 2,400 | 2,040 train / 360 held out |
| `benchmarks/matched-raw.omd` | raw Blade ReSTIR | 128 | matched replacement evaluation |
| `benchmarks/matched-svgf.omd` | Blade variance-guided SVGF | 128 | matched Blade baseline |

Every sample contains low-resolution radiance, depth, world-space normals,
diffuse albedo, specular F0, and roughness, plus a high-resolution canonical
path-traced reference. The v0.3 files additionally contain output-resolution
depth, world normal, diffuse albedo, specular F0, and roughness from the same
primary-surface pass. The primary set uses 128×128 inputs and 256×256
references. Dataset-format version 2 records both plane sets and whether the
low-resolution source is an independent path trace, raw ReSTIR, or
SVGF-filtered.

Each benchmark family uses the same 128 procedural scenes and seed 10000; its
canonical record payloads are byte-identical. The path-first references use
4,096 spp, eight-bounce targets. The 1/4-spp and SVGF files differ only in the
low-resolution evidence paired with those records. The 4-spp files accumulate
four independent paths within one static frame. At quarter resolution that is
one traced path per output pixel, before accounting for the cheaper
low-resolution G-buffer. It is spatial, not motion-aware temporal, data.

## Loading

`.omd` is Ommatidium's little-endian, contiguous `f16` training format. It is
designed for the Rust trainer's direct batch reads rather than the Hugging Face
Dataset Viewer, which is disabled for this custom binary representation.

```sh
hf download mad-bot/ommatidia \
  data/blade-path-trace/4spp-hr-gbuffer-train.omd \
  --repo-type dataset --revision v0.3.0 --local-dir data/hf

cargo run --release -p ommatidia-train -- \
  --data data/hf/data/blade-path-trace/4spp-hr-gbuffer-train.omd \
  --reconstruction-base hr-guided --steps 2000 --batch 8 --out runs/ommatidia
```

See [`docs/design.md`](https://github.com/kvark/ommatidia/blob/main/docs/design.md#data-generation)
for the semantic contract and the source tree for the authoritative parser.
The curated shadow, local-light, and material comparison is documented in the
[August 2026 result](https://github.com/kvark/ommatidia/blob/main/docs/results/curated-oidn-2026-08-22.md).

## Generation provenance

- Ommatidium: `d416f7abdf01e761b7601d597dd0eacfec7a8157`
- Blade: `101c3abd283ddcff858671f3405f041e8d87b782`
- Meganeura: `c3159e830d84be920fc41ecf13b690973e8732a6`
- Primary generation seed: 0
- Matched validation seed: 10000

Future render estimators belong in this repository as separately identified
sources rather than changing the dataset's identity. Training recipes should
pin a Hub revision and enumerate the source names they consume.

The old nearest-base spatial experiment gained only 0.22 dB at 1 spp and 0.08
dB at 4 spp. Re-evaluating deterministic reconstruction exposed that the
baseline, rather than the backbone, was the limiting factor: low-resolution
guidance reaches 34.08 dB / 0.9473 SSIM, while output-resolution primary
surfaces raise the selected 5×5 reconstruction to 34.75 dB / 0.9545 before the
learned correction. See the
[full result](https://github.com/kvark/ommatidia/blob/main/docs/results/path-trace-guided-2026-08-13.md).

## Limitations

The repository contains small procedural scenes from one renderer and a narrow
resolution distribution. It should not be treated as representative of
production game content. The canonical targets reduce Monte Carlo noise but
are finite-sample path traces rather than analytic ground truth. No uploaded
path dataset contains motion vectors, animation, or disocclusions yet.
