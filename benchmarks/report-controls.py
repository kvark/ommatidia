#!/usr/bin/env python3
"""Summarize fixed-budget controls without selecting checkpoints or claiming promotion."""
import argparse
import hashlib
import json
from pathlib import Path

FIELD = ("field-midpoint", "field-stratified", "field-fine", "field-wide")
TRANSPORT = ("transport-control", "transport-no-compressed", "transport-no-confidence", "transport-no-temporal")


def read(root: Path, arm: str) -> tuple[dict, Path]:
    directory = root / arm
    if not directory.is_dir():
        directory = root / ("control-" + arm)
    report = json.loads((directory / "quality.json").read_text())
    if not (directory / "model.safetensors").is_file():
        raise ValueError(f"{arm}: missing serialized checkpoint")
    return report, directory


def capture_hashes(directory: Path, track: str) -> dict[str, str]:
    result = {}
    for line in (directory / "recipe.txt").read_text().splitlines():
        bits = line.split(maxsplit=1)
        if len(bits) == 2 and len(bits[0]) == 64 and all(c in "0123456789abcdef" for c in bits[0]):
            result[Path(bits[1]).name] = bits[0]
    names = ("field.omd", "field.scene.json", "field.transport.json") if track == "field" else (
        "train.omd", "train.transport.json", "validation.omd", "validation.transport.json")
    if any(name not in result for name in names):
        raise ValueError(f"{directory}: missing {track} capture hashes")
    return {name: result[name] for name in names}


def summarize(root: Path) -> dict:
    result = {"role": "construction controls; not promotion or an unseen-scene audit", "field": [], "transport": []}
    fingerprints = {"field": [], "transport": []}
    for arm in FIELD:
        q, directory = read(root, arm)
        fingerprints["field"].append(capture_hashes(directory, "field"))
        if len(q["scores"]) != 1:
            raise ValueError("field control expects one construction scene")
        s = q["scores"][0]
        d = s["diagnostics"]
        result["field"].append({
            "arm": arm, "steps": q["steps"], "seed": q["seed"], "samples": q["samples"],
            "sampling": q["sampling"], "parameters": q["parameter_count"],
            "image_rays_seen": q["image_rays_seen"], "ray_queries_seen": q["ray_queries_seen"],
            "fit_psnr": d["fitting_camera"]["compressed_psnr"],
            "validation_psnr": s["learned"]["compressed_psnr"],
            "linear_mse": s["learned"]["linear_mse"],
            "depth_mae": d["geometry_held"]["hit_depth_mae_world"],
            "hit_accuracy": d["geometry_held"]["hit_miss_accuracy"],
            "context_controls": d["context_controls"],
            "checkpoint_sha256": hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest(),
        })
    baseline = None
    for arm in TRANSPORT:
        q, directory = read(root, arm)
        fingerprints["transport"].append(capture_hashes(directory, "transport"))
        if baseline is None:
            baseline = q["baseline"]
        elif q["baseline"] != baseline:
            raise ValueError("transport controls received different deterministic baselines")
        t = json.loads((directory / "training.json").read_text())
        result["transport"].append({"arm": arm, "steps": t["steps"], "seed": t["seed"],
            "loss_weights": t["loss_weights"], **q["learned"],
            "checkpoint_sha256": hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest()})
    for track, values in fingerprints.items():
        if any(f != values[0] for f in values):
            raise ValueError(f"{track} arms did not use identical capture bytes")
    result["capture_hashes"] = {track: values[0] for track, values in fingerprints.items()}
    result["deterministic"] = baseline
    return result


def markdown(report: dict) -> str:
    text = ["# Controlled reconstruction study", "", report["role"], "",
        "PSNR uses x/(1+x). Field depth is in world units. Energy is scene-linear.", "",
        "| Field | Samples | Parameters | Fit PSNR | Validation PSNR | Linear MSE | Depth MAE |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in report["field"]:
        text.append(f'| {r["arm"]} | {r["samples"]} | {r["parameters"]} | {r["fit_psnr"]:.4f} | {r["validation_psnr"]:.4f} | {r["linear_mse"]:.6f} | {r["depth_mae"]:.6f} |')
    text += ["", "| Transport | PSNR | SSIM | Low-frequency PSNR | Energy | Detail | Temporal MSE |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in [{"arm": "deterministic", **report["deterministic"]}, *report["transport"]]:
        text.append(f'| {r["arm"]} | {r["psnr"]:.4f} | {r["ssim"]:.5f} | {r["low_frequency_psnr"]:.4f} | {r["energy_ratio"]:.5f} | {r["detail_ratio"]:.5f} | {r["temporal_mse"]:.8f} |')
    text += ["", "Images and per-sequence tails must be inspected separately. Lower training loss is not a promotion gate.",
        "The finer field changes both quadrature resolution and surface-target bin width; NLL across sample counts is not directly comparable.", ""]
    return "\n".join(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing all eight result directories")
    args = parser.parse_args()
    report = summarize(args.root)
    (args.root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    text = markdown(report)
    (args.root / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
