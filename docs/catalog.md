# External scene catalog

The generator can place published glTF assets in its procedural rooms, or load
an authored interior as the whole scene. That is the data-scale path in the
[`quality roadmap`](quality-roadmap.md): new geometry and material families,
held out by asset identity, without growing the network.

The runtime does not vendor the meshes. `scripts/fetch-catalog.py` downloads a
small ABO subset into `data/catalog/`, which `.gitignore` already excludes.

## Sources

| source | what it is | format | license | fetch |
|---|---|---|---|---|
| **ABO** | 7,953 artist-designed household objects with PBR materials | `.glb`, Y-up, standing on y=0 | 3D models: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (attribute Amazon.com). The AWS registry listing for the broader collection says CC BY-NC 4.0; this path uses only `3dmodels/`. | public S3, no account |
| **HSSD** | 211 furnished interiors | Habitat/glTF scenes | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | gated Hugging Face (`hssd/hssd-hab`) |
| **DTC** | 2,400+ scanned objects, millimetre geometry, 4K PBR | `.glb` per object | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) with a no-sell restriction | Project Aria email signup, then `dtc_object_downloader` |

Do not commit those binaries. A published Ommatidium dataset or checkpoint that
includes HSSD or DTC renderings inherits those licenses; ABO 3D-model
renderings need Amazon.com attribution. The existing Hugging Face procedural
sets stay the redistributable default.

## Catalog file

A JSON list of entries. Paths are relative to the catalog file.

```json
{
  "entries": [
    {
      "id": "abo/B07H8V49M2",
      "source": "abo",
      "license": "CC-BY-4.0",
      "kind": "object",
      "family": "abo/B07H8V49M2",
      "split": "train",
      "path": "abo/B07H8V49M2.glb"
    }
  ]
}
```

`kind` is `object` (placed on the procedural ground) or `interior` (replaces
the room; the camera samples the axis-aligned interior). `split` is `train` or
`holdout`. The same `id` cannot appear in both. Family is the hold-out key:
today each asset is its own family so a chair in training is never the same
mesh as a chair in the audit. Loading rejects a family assigned to both splits
even if its entries use different ids.

An HSSD interior can additionally name its Habitat instance file and the
renderer-compatible model root:

```json
{
  "id": "hssd/102343992",
  "source": "hssd",
  "license": "CC-BY-NC-4.0",
  "kind": "interior",
  "family": "hssd/102343992",
  "split": "train",
  "path": "hssd/stages/102343992.glb",
  "scene": "hssd/scenes/102343992.scene_instance.json",
  "object_root": "hssd-uncompressed/objects"
}
```

The loader composes each instance's translation, quaternion, and non-uniform
scale under the stage placement. It reports object ids absent from the public
model subset instead of silently treating a stage shell as furnished. The
fetcher rejects a train/hold-out scene selection that reuses a furnishing
mesh across the split.

The generator writes `dataset.catalog.json` beside the `.omd`, naming the ids
used in each scene. Furnished interiors list both the scene and every loaded
object mesh, rather than hiding repeated furniture behind one stage id. That
sidecar is how a later run proves the audit set did not leak. ABO splits are
mesh-disjoint; HSSD splits are authored-scene-disjoint and their recorded
furnishing ids should also be checked when choosing a larger subset.

## Fetch ABO

```sh
python3 scripts/fetch-catalog.py --out data/catalog --train 24 --holdout 8
```

This downloads `3dmodels/metadata/3dmodels.csv.gz`, keeps compact standing
objects, hashes ids into train/holdout, and writes `data/catalog/catalog.json`.
`--dry-run` prints the selection without downloading GLBs.

HSSD: accept the terms for both Hugging Face repositories, authenticate `hf`,
then fetch a bounded furnished subset:

```sh
./scripts/fetch-hssd-scenes.sh data/catalog \
    train:102343992 holdout:102344022
```

Stages and scene-instance JSON come from `hssd/hssd-hab`. Objects come from
`hssd/hssd-models`: its ordinary glTF textures work in Blade, whereas the
compact Habitat GLBs require KTX2/BasisU support Blade does not yet expose.
The fetcher adds or replaces those ids in `data/catalog/catalog.json`; an id
without a prefix is training data only when it is the first positional id.
DTC assets still require the Project Aria form. All sources use this schema.

## Capture

Object scenes keep canopy, lights, textures, and gloss. Authored meshes supply
silhouettes and materials the procedural palette cannot. Interior scenes drop
the procedural room. `--reference-from` copies the expensive 16,384-spp target
onto a 16-frame 1-spp sequence; both runs of a pair must share `--seed`.

```sh
python3 scripts/fetch-catalog.py --out data/catalog --train 24 --holdout 8
./scripts/fetch-hssd-scenes.sh data/catalog \
    train:102343992 holdout:102344022

./scripts/capture-catalog.sh
```

`scripts/capture-catalog.sh` writes:

| file | contents |
|---|---|
| `data/catalog-abo-train-reference.omd` | 24 ABO objects, 16,384 spp |
| `data/catalog-abo-train.omd` | same scenes, 16× 1-spp jittered |
| `data/catalog-abo-holdout-reference.omd` | 8 held-out ABO objects |
| `data/catalog-abo-holdout.omd` | matching 16-frame inputs |
| `data/catalog-hssd-*-reference.omd` | HSSD furnished interiors, 16,384 spp |
| `data/catalog-hssd-*.omd` | matching 16-frame furnished inputs |

For the HSSD arm the capture script derives `data/catalog/furnished.json` and
admits only interior entries with a Habitat `scene`; old stage-only entries
cannot silently turn the furnished corpus back into empty shells.

The checked-in fixture at `ommatidia-data/tests/fixtures/catalog.json` points at
Blade's example glTFs and is what the unit tests load. It is not a quality set.

## Train against a disjoint audit

`--eval-data` scores a second file and uses every sequence in each `--data`
for fitting. Repeat `--data` to alternate whole optimizer batches equally
between corpora, rather than letting the largest file dominate. `--audit-data`
is scored once after the final checkpoint; periodic model selection never
reads it.

```sh
cargo run --release -p ommatidia-train -- \
    --data data/lobe-scale-oracle-1spp-16f-32.omd \
    --data data/catalog-abo-train.omd \
    --eval-data data/catalog-abo-holdout.omd \
    --audit-data data/fresh-procedural-audit.omd \
    --prediction low-color --reconstruction-base split-guided \
    --history-frames 16 --temporal-features phase-lobes \
    --steps 4000 --lr 3e-4 --lr-final 1e-5 \
    --eval-every 1000 --checkpoint-every 1000 --eval-crops 2048 \
    --eval-out runs/catalog-b8-split/eval \
    --out runs/catalog-b8-split
```

This low-color head predicts one correction per input pixel, bilinearly applies
it over the fixed output-resolution split-lobe estimate, and therefore cannot
redraw its high-frequency geometry. It is the constrained B8 experiment after
the unconstrained subpixel residual sharpened noise and lost SSIM.

Repeat B8 on that pair before B16. Only a model that improves SSIM and the
visible full-frame result on the hold-out earns a native runtime contract.
An audit named in an earlier result is no longer untouched; make a fresh seed
family for each promotion decision rather than recycling its filename.
