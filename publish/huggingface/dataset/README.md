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

Every sample contains low-resolution radiance, depth, world-space normals,
diffuse albedo, specular F0, and roughness, plus a high-resolution canonical
path-traced reference. The primary set uses 128×128 inputs and 256×256
references. Dataset-format version 2 records whether the low-resolution source
is raw ReSTIR or SVGF-filtered.

The two benchmark captures use the same 128 procedural scenes and seed 10000;
their canonical record payloads are byte-identical. They differ only in whether
Blade's three-pass variance-guided filter was enabled for the low-resolution
input.

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

## Limitations

The first release contains small procedural scenes from one renderer and a
narrow resolution distribution. It should not be treated as representative of
production game content. The canonical targets reduce Monte Carlo noise but
are finite-sample path traces rather than analytic ground truth.
