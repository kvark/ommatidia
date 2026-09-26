#!/usr/bin/env python3
"""Score predeclared crops from two recorded --eval-only --save-linear runs.

Requires only Python's standard library. Images are row-major, little-endian
scene-linear RGB f32, not decoded PNGs. The benchmark and ordered datasets must
be hashed inputs to both runs, so crops cannot be chosen after seeing a result.
"""

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import sys


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_benchmark(benchmark):
    if benchmark["schema"] != 1:
        raise ValueError("unsupported benchmark schema")
    width, height = benchmark["extent"]
    length = benchmark["sequence_length"]
    if min(width, height, length) < 2 or not benchmark["datasets"] or not benchmark["regions"]:
        raise ValueError("empty or invalid benchmark")
    sequences = sum(case["sequences"] for case in benchmark["datasets"])
    names = set()
    for region in benchmark["regions"]:
        if region["name"] in names or region["kind"] not in ("smooth", "edge", "texture"):
            raise ValueError("duplicate region name or unknown region kind")
        names.add(region["name"])
        x, y, w, h = region["rect"]
        if min(x, y) < 0 or min(w, h) < 2 or x + w > width or y + h > height:
            raise ValueError("invalid crop bounds")
        frames = region["frames"]
        if not 0 <= region["sequence"] < sequences or not frames or len(set(frames)) != len(frames):
            raise ValueError("invalid sequence or duplicate/empty frame selection")
        if any(not isinstance(frame, int) or not 0 <= frame < length for frame in frames):
            raise ValueError("frame outside benchmark sequence")


def verify_run(directory, benchmark_path, benchmark):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or manifest.get("validation_errors", 0):
        raise ValueError(f"run did not complete cleanly: {directory}")
    command = manifest["command"]
    if "--eval-only" not in command or "--save-linear" not in command:
        raise ValueError("expected a recorded full-precision evaluation")
    cwd = Path(manifest["cwd"])
    inputs = {str(Path(item["path"]).resolve()): item["sha256"] for item in manifest["inputs"]}
    if inputs.get(str(benchmark_path.resolve())) != sha256(benchmark_path):
        raise ValueError("benchmark was not locked as an input to this evaluation")
    datasets = [(cwd / command[i + 1]).resolve() for i, arg in enumerate(command) if arg == "--eval-data"]
    expected = [(cwd / case["path"]).resolve() for case in benchmark["datasets"]]
    if datasets != expected:
        raise ValueError("evaluation dataset order differs from benchmark")
    for path, case in zip(datasets, benchmark["datasets"]):
        if inputs.get(str(path)) != case["sha256"] or sha256(path) != case["sha256"]:
            raise ValueError(f"dataset hash mismatch: {path}")
    output = cwd / command[command.index("--out") + 1]
    return output, sha256(manifest_path)


def read_image(path, extent):
    values = array("f")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != extent[0] * extent[1] * 3:
        raise ValueError(f"wrong image size: {path}")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"invalid radiance: {path}")
    return values


def crop_error(prediction, reference, extent, rect):
    x, y, width, height = rect
    residual = []
    for row in range(y, y + height):
        for column in range(x, x + width):
            i = (row * extent[0] + column) * 3
            for a, b in zip(prediction[i:i + 3], reference[i:i + 3]):
                residual.append(a / (1 + a) - b / (1 + b))
    squared = math.fsum(value * value for value in residual)
    gradient = 0.0
    for row in range(height):
        for column in range(width):
            for channel in range(3):
                i = (row * width + column) * 3 + channel
                if column + 1 < width:
                    gradient += (residual[i + 3] - residual[i]) ** 2
                if row + 1 < height:
                    gradient += (residual[i + width * 3] - residual[i]) ** 2
    return {"squared_sum": squared, "values": len(residual), "gradient_squared_sum": gradient,
            "gradient_values": 3 * ((width - 1) * height + (height - 1) * width)}


def summarize(rows):
    result = {}
    for model in ("before", "after"):
        result[model] = {
            "mse": sum(row[model]["squared_sum"] for row in rows) / sum(row[model]["values"] for row in rows),
            "gradient_mse": sum(row[model]["gradient_squared_sum"] for row in rows)
                            / sum(row[model]["gradient_values"] for row in rows),
        }
    result["ratio"] = {metric: result["after"][metric] / result["before"][metric]
                       if result["before"][metric] else None for metric in ("mse", "gradient_mse")}
    return result


def score(benchmark, before, after):
    rows = []
    case_by_sequence = [case["name"] for case in benchmark["datasets"] for _ in range(case["sequences"])]
    frame_regions = {}
    for region in benchmark["regions"]:
        for frame in region["frames"]:
            frame_regions.setdefault((region["sequence"], frame), []).append(region)
    for (sequence, frame), regions in sorted(frame_regions.items()):
        prefix = f"{sequence:03}-{frame:03}"
        reference_path = before / f"{prefix}-reference.rgbf32"
        if sha256(reference_path) != sha256(after / reference_path.name):
            raise ValueError(f"reference differs between runs: {prefix}")
        reference = read_image(reference_path, benchmark["extent"])
        images = {name: read_image(directory / f"{prefix}-learned.rgbf32", benchmark["extent"])
                  for name, directory in (("before", before), ("after", after))}
        for region in regions:
            row = {"region": region["name"], "kind": region["kind"], "case": case_by_sequence[sequence],
                   "sequence": sequence, "frame": frame, "rect": region["rect"]}
            row.update({name: crop_error(image, reference, benchmark["extent"], region["rect"])
                        for name, image in images.items()})
            rows.append(row)
    groups = {}
    for field in ("region", "case", "kind"):
        groups[field] = {value: summarize([row for row in rows if row[field] == value])
                         for value in sorted({row[field] for row in rows})}
    return {"metric_space": "fixed x/(1+x), before sRGB or quantization",
            "aggregation": "RGB-value weighted; gradient-neighbor weighted",
            "scope": "spatial crops only; no automatic temporal or visual pass", "groups": groups, "frames": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--before-run", type=Path, required=True)
    parser.add_argument("--after-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text())
    validate_benchmark(benchmark)
    before, before_hash = verify_run(args.before_run, args.benchmark, benchmark)
    after, after_hash = verify_run(args.after_run, args.benchmark, benchmark)
    result = score(benchmark, before, after)
    result["benchmark_sha256"] = sha256(args.benchmark)
    result["run_manifest_sha256"] = {"before": before_hash, "after": after_hash}
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["groups"]["kind"], indent=2))


if __name__ == "__main__":
    main()
