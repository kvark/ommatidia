#!/usr/bin/env python3
"""Reproduce Phase 4: preflight, capture 8x4 scenes/case, then verify the 200-scene corpus.

Run each subcommand through record-run.py. Capture additionally records each batch.
No model inference, crop selection, new asset downloads, or evaluation is performed.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess


ROOT = Path("runs/v4-phase4")
OLD = Path("runs/quality-week-2026-09-26")
CASES = ("static", "camera", "objects", "lights", "catalog")
POOL = Path("runs/quality-2026-09-26/data/opaque-train-catalog.json")
BUILD = OLD / "confirmation-capture-build/manifest.json"
PROTECTED = [Path(p) for p in (
    "docs/dev-crops.json", "docs/quality-benchmark.json",
    "docs/quality-confirmation.json", "docs/results/selection.json",
    "docs/results/results.json",
)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def seeds(base, count):
    return [base ^ ((i * 0x9E3779B9) & ((1 << 64) - 1)) for i in range(count)]


def option(command, name):
    return command[command.index(name) + 1]


def replace(command, name, value):
    result = command.copy()
    result[result.index(name) + 1] = str(value)
    return result


def commands():
    jobs = []
    for batch in range(8):
        for case_index, case in enumerate(CASES):
            template = OLD / f"capture-train-extra-{case}/manifest.json"
            source = read(template)
            require(source["status"] == "complete", f"incomplete template: {template}")
            command = source["command"]
            require(option(command, "--samples") == "4", "template batch changed")
            base = 1_010_001 + case_index * 10_000 + batch * 1_000
            name = f"capture-{case}-{batch:02}"
            out = ROOT / "data" / f"train-{case}-{batch:02}.omd"
            changed = replace(replace(command, "--seed", base), "--out", out)
            # Keep even the archived executable; only seed/output are replaced.
            require(replace(replace(changed, "--seed", option(command, "--seed")),
                            "--out", option(command, "--out")) == command,
                    "capture settings changed")
            executable = next(i for i in source["inputs"]
                              if i["path"].endswith("/confirmation-runtime/ommatidia-data"))
            require(sha(command[0]) == executable["sha256"], "archived executable changed")
            jobs.append({"name": name, "case": case, "batch": batch,
                         "template": identity(template), "command": changed,
                         "scene_seeds": seeds(base, 4), "output": str(out),
                         "environment": source["environment"]})
    return jobs


def seed_evidence(value):
    """Include nested published ancestry and recorded capture commands, not train RNGs."""
    found = set()
    if isinstance(value, dict):
        if "scene_seeds" in value:
            found.update(value["scene_seeds"])
        command = value.get("command", [])
        if isinstance(command, list) and all(isinstance(s, str) for s in command):
            if any(Path(s).name == "ommatidia-data" for s in command) and "--samples" in command:
                base = int(option(command, "--seed")) if "--seed" in command else 0
                found.update(seeds(base, int(option(command, "--samples"))))
        for child in value.values():
            found.update(seed_evidence(child))
    elif isinstance(value, list):
        for child in value:
            found.update(seed_evidence(child))
    return found


def header(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(64)
    require(len(raw) == 64 and raw[:8] == b"OMMATIDA", f"bad OMD: {path}")
    version, scale, width, height, lr, hr, count, source, length = struct.unpack_from("<9I", raw, 8)
    require(version == 2 and scale == 2 and width == height == 128 and source == 3
            and length == 64 and count == 256, f"unexpected layout: {path}")
    channels = [3, 1, 3, 3, 3, 1, 2, 2, 3, 3, 3]
    require(lr < 2048 and hr < 2048, "unknown planes")
    n = width * height
    record_values = n * sum(c for i, c in enumerate(channels) if lr & (1 << i))
    record_values += n * scale * scale * sum(c for i, c in enumerate(channels) if hr & (1 << i))
    require(Path(path).stat().st_size == 64 + count * record_values * 2, f"truncated/extra data: {path}")
    return {"version": version, "scale": scale, "width": width, "height": height,
            "lr_planes": lr, "hr_planes": hr, "records": count, "source": source,
            "sequence_length": length, "record_values": record_values}


def visibility(value, allowed):
    scenes = value["scenes"]
    require(len(scenes) == 4, "catalog must contain four scenes")
    minimum = 1.0
    families = set()
    for index, scene in enumerate(scenes):
        require(scene["index"] == index, "catalog index/order changed")
        require(scene["kind"] == "object" and scene["sources"] == ["abo"], "unexpected assets")
        require(len(scene["ids"]) == len(scene["families"]) == 1, "expected one catalog object")
        require(set(scene["families"]) <= allowed, "non-training catalog family")
        coverage = scene["visible_fraction"]
        require(len(coverage) == 64, "incomplete visibility trajectory")
        require(all(math.isfinite(v) and 0.01 <= v <= 1 for v in coverage), "invisible catalog frame")
        require(1 <= scene["camera_attempts"] <= 8, "invalid camera retry count")
        minimum = min(minimum, *coverage)
        families.update(scene["families"])
    return {"frames": 256, "minimum": minimum, "families": sorted(families)}


def original_paths():
    cmd = read(OLD / "cold-coverage-training/manifest.json")["command"]
    return [Path(cmd[i + 1]) for i, arg in enumerate(cmd) if arg == "--data"]


def protected_paths():
    paths = set(PROTECTED)
    for role in ("dev", "audit", "confirm"):
        for case in CASES:
            data = OLD / f"data/{role}-{case}.omd"
            paths.update([data, data.with_suffix(".transport.json")])
            if case == "catalog":
                paths.add(data.with_suffix(".catalog.json"))
    return sorted(paths)


def preflight():
    jobs = commands()
    proposed = [seed for j in jobs for seed in j["scene_seeds"]]
    require(len(proposed) == len(set(proposed)) == 160, "new seeds overlap")
    evidence = []
    # Includes every transport sidecar, manifest snapshot, nested result and published ancestry.
    for root in (Path("runs"), Path("data"), Path("docs")):
        for path in sorted(root.rglob("*.json")):
            if path.is_relative_to(ROOT):
                continue
            try:
                value = read(path)
            except (ValueError, UnicodeError):
                continue  # Historical interrupted run manifests can be empty.
            existing = seed_evidence(value)
            if not existing:
                continue
            overlap = existing.intersection(proposed)
            require(not overlap, f"seed collision {sorted(overlap)}: {path}")
            evidence.append({**identity(path), "scene_seeds": sorted(existing)})
    require(evidence, "no existing provenance found")
    pool = read(POOL)["entries"]
    families = {entry["family"] for entry in pool}
    require(len(families) == len(pool) == 22, "training pool changed")
    require(all(entry["split"] == "train" and entry["source"] == "abo"
                and entry["kind"] == "object" for entry in pool), "new asset source/split")
    assets = [identity((POOL.parent / entry["path"]).resolve()) for entry in pool]
    for role in ("dev", "audit", "confirm"):
        used = set(read(OLD / f"data/{role}-catalog.transport.json")["family_ids"])
        require(not used.intersection(families), f"{role} family leakage")
    old_paths = original_paths()
    require(len(old_paths) == 10, "expected original 40-scene corpus")
    old = []
    for path in old_paths:
        provenance = read(path.with_suffix(".transport.json"))
        require(len(provenance["scene_seeds"]) == 4, "wrong original scene count")
        old.append({**identity(path), "header": header(path),
                    "provenance": identity(path.with_suffix(".transport.json"))})
    required_bytes = sum(Path(option(j["command"], "--out")).stat().st_size
                         if Path(j["output"]).exists() else
                         Path(option(read(j["template"]["path"])["command"], "--out")).stat().st_size
                         for j in jobs)
    require(not any(Path(j["output"]).exists() for j in jobs), "capture output already exists")
    require(shutil.disk_usage(ROOT).free > required_bytes + 8 * 1024**3, "insufficient disk headroom")
    report = {"schema": 1, "jobs": jobs, "seed_evidence": evidence,
              "unique_prior_seeds": len({s for e in evidence for s in e["scene_seeds"]}),
              "training_pool": identity(POOL), "training_families": sorted(families),
              "assets": assets, "build": identity(BUILD),
              "executable": identity(jobs[0]["command"][0]),
              "original": old, "protected": [identity(p) for p in protected_paths()],
              "new_scenes": 160, "new_frames": 10240, "expected_bytes": required_bytes}
    write_new(ROOT / "preflight.json", report)
    print(f"PASS: 160 unique proposed scenes disjoint from {report['unique_prior_seeds']} "
          f"recorded prior seeds ({len(evidence)} provenance sources); {required_bytes} bytes needed")


def check_identity(item):
    require(identity(item["path"]) == item, f"protected input changed: {item['path']}")


def same_f32(a, b):
    # Struct sidecars serialize f32 directly; serde_json::Value serializes its
    # exact f64 promotion. Compare the underlying f32, not decimal spellings.
    return math.isfinite(a) and math.isfinite(b) and struct.pack("<f", a) == struct.pack("<f", b)


def capture():
    report = read(ROOT / "preflight.json")
    for item in [report[k] for k in ("training_pool", "build", "executable")] + report["assets"]:
        check_identity(item)
    (ROOT / "data").mkdir(exist_ok=True)
    for job in report["jobs"]:
        check_identity(job["template"])
        manifest = ROOT / job["name"] / "manifest.json"
        if manifest.exists():
            prior = read(manifest)
            require(prior["status"] == "complete" and prior["command"] == job["command"],
                    f"failed/interrupted capture needs inspection, not overwrite: {manifest}")
            header(job["output"])
            print(f"Already complete: {job['name']}", flush=True)
            continue
        require(not Path(job["output"]).exists(), "refusing to overwrite capture")
        require(shutil.disk_usage(ROOT).free > 10 * 1024**3, "disk reserve exhausted")
        inputs = [Path(__file__), ROOT / "preflight.json", Path(job["template"]["path"]),
                  Path(report["executable"]["path"]), BUILD, POOL]
        command = ["python3", "scripts/record-run.py", str(ROOT / job["name"])]
        for path in inputs:
            command.extend(["--input", str(path)])
        command.extend(["--", *job["command"]])
        print(f"Starting {job['name']}: {job['scene_seeds']}", flush=True)
        subprocess.run(command, env={**os.environ, **job["environment"]}, check=True)
        header(job["output"])
        if job["case"] == "catalog":
            visibility(read(Path(job["output"]).with_suffix(".catalog.json")),
                       set(report["training_families"]))


def finite_data(path, info):
    import numpy as np

    values = np.memmap(path, dtype="<f2", mode="r", offset=64,
                       shape=(info["records"], info["record_values"]))
    channels = [3, 1, 3, 3, 3, 1, 2, 2, 3, 3, 3]
    n = info["width"] * info["height"]
    lr_values = n * sum(c for i, c in enumerate(channels) if info["lr_planes"] & (1 << i))
    hr_n = n * info["scale"]**2
    # ALL_PLANES starts with RGB, diffuse, specular and emission; all four
    # must be present before treating the first twelve channels as radiance.
    radiance_mask = 1 | (1 << 8) | (1 << 9) | (1 << 10)
    require(info["hr_planes"] & radiance_mask == radiance_mask, "missing split radiance")
    peak = 0.0
    for index in range(info["records"]):
        require(np.isfinite(values[index]).all(), f"nonfinite record {path}:{index}")
        radiance = values[index, lr_values:lr_values + 12 * hr_n]
        require((radiance >= 0).all(), f"negative reference radiance {path}:{index}")
        require(radiance[:3 * hr_n].sum(dtype=np.float64) > 1e-6,
                f"entirely black reference {path}:{index}")
        peak = max(peak, float(radiance.max()))
    del values
    return {"finite_records": info["records"], "nonnegative_nonblack_references": True,
            "peak_reference_radiance": peak}


def verify():
    report = read(ROOT / "preflight.json")
    for item in report["protected"]:
        check_identity(item)
    for item in [report[k] for k in ("training_pool", "build", "executable")] + report["assets"]:
        check_identity(item)
    corpus = []
    all_seeds = set()
    for index, path in enumerate(original_paths() + [Path(j["output"]) for j in report["jobs"]]):
        info = header(path)
        provenance_path = path.with_suffix(".transport.json")
        provenance = read(provenance_path)
        require(provenance["records"] == 256 and len(provenance["scene_seeds"]) == 4,
                "provenance count mismatch")
        require(provenance["scene_seeds"] == seeds(provenance["capture_seed"], 4), "seed schedule changed")
        require(not all_seeds.intersection(provenance["scene_seeds"]), "corpus seed overlap")
        all_seeds.update(provenance["scene_seeds"])
        for key, expected in {"canonical_frames": 256, "input_frames": 1,
                              "input_max_bounces": 8, "reference_max_bounces": 8,
                              "matching_path_depth": True, "input_estimator": "independent-paths",
                              "reference_from": None, "input_sample_offset": 0,
                              "reference_sample_offset": 0}.items():
            require(provenance[key] == expected, f"wrong {key}: {path}")
        require(provenance["device"]["driver_info"] == "Mesa 26.2.3-1ubuntu1", "wrong capture driver")
        numerics = finite_data(path, info)
        item = {**identity(path), "header": info, "scene_seeds": provenance["scene_seeds"],
                "provenance": identity(provenance_path), "numerics": numerics}
        if index < 10:
            original = report["original"][index]
            require(all(item[k] == v for k, v in original.items()), "original corpus changed")
            case = CASES[index % 5]
        else:
            job = report["jobs"][index - 10]
            case = job["case"]
            run_path = ROOT / job["name"] / "manifest.json"
            run = read(run_path)
            require(run["status"] == "complete" and run["exit_code"] == run["validation_errors"] == 0,
                    f"failed capture: {run_path}")
            require(run["command"] == job["command"] and item["scene_seeds"] == job["scene_seeds"],
                    "capture differs from preflight")
            require(all(run["environment"].get(k) == v for k, v in job["environment"].items()),
                    "capture environment differs from template")
            template_path = Path(option(read(job["template"]["path"])["command"], "--out"))
            require(info == header(template_path), "capture layout differs from template")
            item["capture_run"] = identity(run_path)
        item["case"] = case
        if case == "catalog":
            catalog_path = path.with_suffix(".catalog.json")
            item["catalog"] = identity(catalog_path)
            item["visibility"] = visibility(read(catalog_path), set(report["training_families"]))
            require(set(provenance["family_ids"]) == set(item["visibility"]["families"]), "family metadata mismatch")
            require(same_f32(provenance["minimum_catalog_visible_fraction"], item["visibility"]["minimum"]),
                    "visibility metadata mismatch")
        corpus.append(item)
        print(f"Verified {index + 1}/50: {path}", flush=True)
    require(len(all_seeds) == 200, "expected 200 unique scenes")
    counts = Counter(item["case"] for item in corpus)
    require(counts == Counter({case: 10 for case in CASES}), "unbalanced case membership")
    result = {"schema": 1, "preflight": identity(ROOT / "preflight.json"), "scenes": 200,
              "frames": 12800, "sequences_per_case": 40, "datasets": corpus,
              "bytes": sum(item["bytes"] for item in corpus),
              "catalog_minimum_visible_fraction": min(item["visibility"]["minimum"]
                                                       for item in corpus if "visibility" in item),
              "protected": report["protected"], "pass": True}
    write_new(ROOT / "corpus.json", result)
    print(f"PASS: 200 scenes, 12800 frames; {result['bytes']} bytes; all frames finite; "
          "full catalog trajectories visible; development and audits unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "capture", "verify"))
    args = parser.parse_args()
    globals()[args.action]()
