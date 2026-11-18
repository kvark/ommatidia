#!/usr/bin/env python3
"""Separate input-noise sensitivity from repeatable RGB error on matched evaluations."""
import argparse
import copy
import json
from pathlib import Path
import runpy

import numpy as np

from evaluation_metrics import read_frames, same_protocol, sha256


def compressed(path, extent):
    rgb = np.fromfile(path, dtype="<f4").astype(np.float64)
    if rgb.size != extent[0] * extent[1] * 3 or not np.isfinite(rgb).all() or (rgb < 0).any():
        raise ValueError(f"invalid radiance in {path}")
    return rgb / (1 + rgb)


def measure(first, second):
    qa, _, rows = read_frames(first / "frames.csv")
    qb, _, _ = read_frames(second / "frames.csv")
    if qa["checkpoint"] != qb["checkpoint"] or not qa["save_linear"] or not qb["save_linear"]:
        raise ValueError("expected two linear evaluations of the same checkpoint")
    captures = [copy.deepcopy(q["capture"]) for q in (qa, qb)]
    offsets = [[capture.pop("input_sample_offset") for capture in c["captures"]] for c in captures]
    if captures[0] != captures[1] or any(a == b for a, b in zip(*offsets)):
        raise ValueError("expected the same captures at different path offsets")
    same_protocol(qa, {**qb, "capture": qa["capture"]})
    diagnostics = json.loads((first / "diagnostics.json").read_text())
    for row, diag in zip(rows, diagnostics, strict=True):
        sequence, frame = int(row["sequence"]), int(row["frame"])
        if (sequence, frame) != (diag["sequence"], diag["frame"]):
            raise ValueError("lobe metrics differ in frame order")
        stem = f"{sequence:03}-{frame:03}"
        ref = f"{stem}-reference.rgbf32"
        row["reference_sha256"] = sha256(first / ref)
        if row["reference_sha256"] != sha256(second / ref):
            raise ValueError(f"different targets at {stem}")
        a, b = [compressed(p / f"{stem}-learned.rgbf32", qa["extent"]) for p in (first, second)]
        y = compressed(first / ref, qa["extent"])
        row["learned_input_noise_mse"] = 0.5 * np.mean((a - b) ** 2)
        row["learned_repeatable_mse"] = np.mean((a - y) * (b - y))
        row["learned_pair_mse"] = 0.5 * (np.mean((a - y) ** 2) + np.mean((b - y) ** 2))
        for key, value in diag["learned"].items():
            row[f"learned_{key}"] = value
    return (qa, qb), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", nargs=2, type=Path, required=True)
    parser.add_argument("--after", nargs=2, type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    before, a = measure(*args.before)
    after, b = measure(*args.after)
    for left, right in zip(before, after):
        same_protocol(left, right)
    compare = runpy.run_path(str(Path(__file__).with_name("bootstrap-evaluation.py")))["compare"]
    report = compare(a, b, metrics=("input_noise_mse", "repeatable_mse", "pair_mse",
                                    "diffuse_illumination_mse", "diffuse_radiance_mse",
                                    "specular_radiance_mse"))
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
