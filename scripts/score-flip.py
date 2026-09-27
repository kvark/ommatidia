#!/usr/bin/env python3
"""Enrich a --save-linear evaluation with official LDR-FLIP at fixed 67 PPD.

Writes a NEW output directory, never modifies the source evaluation. Radiance is
mapped by x/(1+x), then sRGB, without PNG quantization or exposure fitting.
"""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path

import flip_evaluator as flip
import numpy as np

from evaluation_metrics import means, read_frames, sha256


def display(image):
    compressed = image / (1.0 + image)
    return np.ascontiguousarray(np.clip(np.where(
        compressed <= 0.0031308, 12.92 * compressed,
        1.055 * np.power(compressed, 1.0 / 2.4) - 0.055), 0.0, 1.0), dtype=np.float32)


def read_image(path, extent):
    image = np.fromfile(path, dtype="<f4")
    if path.stat().st_size != extent[0] * extent[1] * 3 * 4:
        raise ValueError(f"wrong image size: {path}")
    if not np.isfinite(image).all() or (image < 0).any():
        raise ValueError(f"nonfinite or negative radiance: {path}")
    return image.reshape(extent[1], extent[0], 3)


def ldr_flip(reference, image):
    _, mean, parameters = flip.evaluate(reference, image, "LDR", inputsRGB=True,
        applyMagma=False, computeMeanError=True, parameters={"ppd": 67.0})
    if not np.isfinite(mean) or not 0 <= mean <= 1 or parameters["ppd"] != 67.0:
        raise ValueError("invalid FLIP result or display parameters")
    return float(mean)


def score(directory, output):
    quality, fields, rows = read_frames(directory / "frames.csv")
    if not quality["save_linear"]:
        raise ValueError("requires --save-linear evaluation outputs")
    if "learned_flip" in fields:
        raise ValueError("evaluation already contains FLIP; use original raw outputs")
    # Archived v3 reports retain their baseline; v4 has only one live model.
    roles = {}
    if "baseline_psnr" in fields:
        roles["baseline"] = (directory, "base")
    roles["learned"] = (directory, "learned")
    if quality["control_run"] is not None:
        roles["control"] = (Path(quality["control_run"]), "learned")
    added = ["reference_sha256"] + [f"{role}_flip" for role in roles]
    if "control" in roles:
        added.append("delta_flip")
    output.mkdir(parents=True, exist_ok=False)
    for index, row in enumerate(rows):
        prefix = f'{int(row["sequence"]):03}-{int(row["frame"]):03}'
        reference_path = directory / f"{prefix}-reference.rgbf32"
        row["reference_sha256"] = sha256(reference_path)
        reference = display(read_image(reference_path, quality["extent"]))
        for role, (location, image_name) in roles.items():
            if role == "control" and sha256(location / reference_path.name) != row["reference_sha256"]:
                raise ValueError(f"control reference changed: {prefix}")
            prediction = display(read_image(location / f"{prefix}-{image_name}.rgbf32", quality["extent"]))
            row[f"{role}_flip"] = str(ldr_flip(reference, prediction))
        if "control" in roles:
            row["delta_flip"] = str(float(row["learned_flip"]) - float(row["control_flip"]))
        if (index + 1) % 64 == 0:
            print(f"FLIP: {index + 1}/{len(rows)} frames", flush=True)
    with (output / "frames.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + added)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "implementation": "NVlabs/flip official Python CPU implementation",
        "flip_evaluator_version": importlib.metadata.version("flip-evaluator"),
        "numpy_version": np.__version__, "ppd": 67.0,
        "transform": "scene-linear x/(1+x), then sRGB; no quantization",
        "source_evaluation": str(directory.resolve()),
        "source_frames_sha256": sha256(directory / "frames.csv"),
        "source_quality_sha256": sha256(directory / "quality.json"),
        "roles": {role: means(rows, f"{role}_flip") for role in roles},
    }
    quality["flip"] = report
    # This directory holds scores, not another copy of the raw images.
    quality["save_linear"] = False
    for role in roles:
        quality[role]["flip"] = report["roles"][role]["all"]["mean"]
        for group in ("cold", "early", "settling", "warm"):
            if quality["buckets"][group][role] is not None:
                quality["buckets"][group][role]["flip"] = report["roles"][role][group]["mean"]
    (output / "quality.json").write_text(json.dumps(quality, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="evaluation OUTPUT directory, not recording directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    score(args.run, args.out)


if __name__ == "__main__":
    main()
