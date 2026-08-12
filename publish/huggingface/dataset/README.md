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
| `data/blade-restir/train.omd` | raw Blade ReSTIR | 2,400 | 2,040 train / 360 held out |
| `benchmarks/matched-raw.omd` | raw Blade ReSTIR | 128 | matched replacement evaluation |
| `benchmarks/matched-svgf.omd` | Blade variance-guided SVGF | 128 | matched Blade baseline |
| `data/blade-path-trace/1spp-train.omd` | sparse path trace, 1 spp | 2,400 | path-first spatial training |
| `benchmarks/path-trace-1spp-validation.omd` | sparse path trace, 1 spp | 128 | independent path validation |
| `ablations/path-trace-4spp-static-train.omd` | sparse path trace, 4 static spp | 2,400 | favorable history-evidence proxy |
| `benchmarks/path-trace-4spp-static-validation.omd` | sparse path trace, 4 static spp | 128 | independent proxy validation |
| `benchmarks/path-reference-svgf-validation.omd` | Blade ReSTIR+SVGF | 128 | same new canonical references |

Every sample contains low-resolution radiance, depth, world-space normals,
diffuse albedo, specular F0, and roughness, plus a high-resolution canonical
path-traced reference. The primary set uses 128×128 inputs and 256×256
references. Dataset-format version 2 records whether the low-resolution source
is raw ReSTIR or SVGF-filtered.

Each benchmark family uses the same 128 procedural scenes and seed 10000; its
canonical record payloads are byte-identical. The path-first references use
4,096 spp, eight-bounce targets. The 1/4-spp and SVGF files differ only in the
low-resolution evidence paired with those records. The 4-spp files accumulate
four samples at a static camera; they are an ablation, not motion-aware temporal
training data.

## Loading

`.omd` is Ommatidium's little-endian, contiguous `f16` training format. It is
designed for the Rust trainer's direct batch reads rather than the Hugging Face
Dataset Viewer, which is disabled for this custom binary representation.

```sh
hf download mad-bot/ommatidia \
  data/blade-restir/train.omd \
  --repo-type dataset --revision v0.1.0 --local-dir data/hf

cargo run --release -p ommatidia-train -- \
  --data data/hf/data/blade-restir/train.omd \
  --steps 20000 --batch 8 --out runs/ommatidia
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

The first spatial path experiment is intentionally retained even though it is
a negative result. On the independent set, the 1-spp model gains only 0.22 dB
and the 4-spp model only 0.08 dB. See the
[full result](https://github.com/kvark/ommatidia/blob/agent/path-tracing-ci/docs/results/path-trace-spatial-b24-2026-08-12.md);
these files are the reproducible baseline for the temporal model, not a claim
that the current spatial checkpoint solves sparse-path reconstruction.

## Limitations

The repository contains small procedural scenes from one renderer and a narrow
resolution distribution. It should not be treated as representative of
production game content. The canonical targets reduce Monte Carlo noise but
are finite-sample path traces rather than analytic ground truth. No uploaded
path dataset contains motion vectors, animation, or disocclusions yet.
