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
| `ablations/path-trace-4spp-static-train.omd` | independent path trace, 4 spp | 2,400 | current spatial training set |
| `benchmarks/path-trace-4spp-static-validation.omd` | independent path trace, 4 spp | 128 | separate seed-10000 validation |
| `benchmarks/path-reference-svgf-validation.omd` | Blade ReSTIR+SVGF | 128 | comparison control with matched references |
| `data/blade-path-trace/1spp-train.omd` | independent path trace, 1 spp | 2,400 | lower-ray-budget experiment |
| `benchmarks/path-trace-1spp-validation.omd` | independent path trace, 1 spp | 128 | lower-ray-budget validation |
| `data/blade-restir/train.omd` | raw Blade ReSTIR | 2,400 | 2,040 train / 360 held out |
| `benchmarks/matched-raw.omd` | raw Blade ReSTIR | 128 | matched replacement evaluation |
| `benchmarks/matched-svgf.omd` | Blade variance-guided SVGF | 128 | matched Blade baseline |

Every sample contains low-resolution radiance, depth, world-space normals,
diffuse albedo, specular F0, and roughness, plus a high-resolution canonical
path-traced reference. The primary set uses 128×128 inputs and 256×256
references. Dataset-format version 2 records whether the low-resolution source
is an independent path trace, raw ReSTIR, or SVGF-filtered.

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
  ablations/path-trace-4spp-static-train.omd \
  --repo-type dataset --revision v0.2.0 --local-dir data/hf

cargo run --release -p ommatidia-train -- \
  --data data/hf/ablations/path-trace-4spp-static-train.omd \
  --steps 6000 --batch 8 --out runs/ommatidia
```

See [`docs/design.md`](https://github.com/kvark/ommatidia/blob/main/docs/design.md#data-generation)
for the semantic contract and the source tree for the authoritative parser.

## Generation provenance

- Ommatidium: `7f08f025a3150c355643af513bc2825d88441520`
- Blade upstream integration: `3a8895a`
- Meganeura upstream integration: `256b906`
- Primary generation seed: 1
- Matched validation seed: 10000

Future render estimators belong in this repository as separately identified
sources rather than changing the dataset's identity. Training recipes should
pin a Hub revision and enumerate the source names they consume.

The old nearest-base spatial experiment gained only 0.22 dB at 1 spp and 0.08
dB at 4 spp. Re-evaluating the deterministic reconstruction exposed that the
baseline, rather than the backbone, was the limiting factor: the current
depth/normal/albedo-guided reconstruction reaches 34.08 dB / 0.9473 SSIM on
the separate 4-spp validation set before the learned correction. See the
[full result](https://github.com/kvark/ommatidia/blob/main/docs/results/path-trace-guided-2026-08-13.md).

## Limitations

The repository contains small procedural scenes from one renderer and a narrow
resolution distribution. It should not be treated as representative of
production game content. The canonical targets reduce Monte Carlo noise but
are finite-sample path traces rather than analytic ground truth. No uploaded
path dataset contains motion vectors, animation, or disocclusions yet.
