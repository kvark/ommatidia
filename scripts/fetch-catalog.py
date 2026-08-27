#!/usr/bin/env python3
"""Download a small ABO glTF subset and write an Ommatidium catalog.

HSSD and DTC are gated; this script only fetches the public ABO 3D-model
bucket. See docs/catalog.md for how to add those sources to the same JSON.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

BUCKET = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com"
METADATA = f"{BUCKET}/3dmodels/metadata/3dmodels.csv.gz"
ORIGINAL = f"{BUCKET}/3dmodels/original"


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ommatidia-catalog/1"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def load_rows() -> list[dict[str, str]]:
    raw = gzip.decompress(download(METADATA))
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


def extent(row: dict[str, str]) -> float:
    return max(float(row["extent_x"]), float(row["extent_y"]), float(row["extent_z"]))


def compact(row: dict[str, str]) -> bool:
    vertices = int(row["vertices"])
    longest = extent(row)
    height = int(row["image_height_max"] or 0)
    return (
        1500 <= vertices <= 20000
        and 0.35 <= longest <= 2.5
        and height <= 2048
        and float(row["extent_y"]) >= 0.15
    )


def bucket(model_id: str) -> str:
    digest = hashlib.sha256(model_id.encode("utf-8")).digest()
    return "holdout" if digest[0] < 64 else "train"


def select(rows: list[dict[str, str]], train: int, holdout: int) -> list[dict[str, str]]:
    ranked = sorted((row for row in rows if compact(row)), key=lambda row: row["3dmodel_id"])
    chosen_train: list[dict[str, str]] = []
    chosen_holdout: list[dict[str, str]] = []
    for row in ranked:
        target = bucket(row["3dmodel_id"])
        if target == "train" and len(chosen_train) < train:
            chosen_train.append(row)
        elif target == "holdout" and len(chosen_holdout) < holdout:
            chosen_holdout.append(row)
        if len(chosen_train) >= train and len(chosen_holdout) >= holdout:
            break
    if len(chosen_train) < train or len(chosen_holdout) < holdout:
        raise SystemExit(
            f"only found {len(chosen_train)} train and {len(chosen_holdout)} holdout "
            f"objects that pass the compactness filter"
        )
    return chosen_train + chosen_holdout


def write_catalog(out: Path, rows: list[dict[str, str]]) -> None:
    entries = []
    for row in rows:
        model_id = row["3dmodel_id"]
        split = bucket(model_id)
        rel = Path("abo") / f"{model_id}.glb"
        entries.append(
            {
                "id": f"abo/{model_id}",
                "source": "abo",
                "license": "CC-BY-4.0",
                "kind": "object",
                "family": f"abo/{model_id}",
                "split": split,
                "path": str(rel),
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    catalog = out / "catalog.json"
    catalog.write_text(json.dumps({"entries": entries}, indent=2) + "\n")
    print(f"wrote {catalog} ({len(entries)} entries)")


def fetch_models(out: Path, rows: list[dict[str, str]]) -> None:
    dest = out / "abo"
    dest.mkdir(parents=True, exist_ok=True)
    for row in rows:
        model_id = row["3dmodel_id"]
        target = dest / f"{model_id}.glb"
        if target.exists() and target.stat().st_size > 0:
            print(f"keep {target}")
            continue
        url = f"{ORIGINAL}/{row['path']}"
        print(f"get  {url}")
        target.write_bytes(download(url))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/catalog"))
    parser.add_argument("--train", type=int, default=24)
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write catalog.json from metadata only, do not download GLBs",
    )
    args = parser.parse_args()
    print("loading ABO 3d model metadata", file=sys.stderr)
    rows = select(load_rows(), args.train, args.holdout)
    write_catalog(args.out, rows)
    for row in rows:
        print(
            f"{bucket(row['3dmodel_id']):7} {row['3dmodel_id']}  "
            f"v={row['vertices']}  y={float(row['extent_y']):.2f}  {row['path']}"
        )
    if args.dry_run:
        return
    fetch_models(args.out, rows)


if __name__ == "__main__":
    main()
