#!/usr/bin/env python3
"""GPU smoke: training metrics-only, serialized reload, periodic cuts and saved control.

Use tiny disjoint, two-frame captures, e.g. those made by the CI LavaPipe job.
Wrap this command with record-run.py and record the binary and both captures.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    base = [str(args.binary), "--eval-data", str(args.eval_data)]
    train = args.out / "train"
    settings = ["--data", str(args.train_data), "--steps", "2", "--channels", "4",
                "--unroll", "2", "--batch", "2", "--crop", "16", "--eval-every", "1"]
    subprocess.run(base + settings + ["--out", str(train)], check=True)
    interrupted = args.out / "interrupted"
    subprocess.run(base + settings + ["--out", str(interrupted), "--stop-after", "1"], check=True)
    resumed = args.out / "resumed"
    subprocess.run(base + settings + ["--out", str(resumed), "--checkpoint",
                   str(interrupted / "model.safetensors")], check=True)
    assert (resumed / "frames.csv").read_bytes() == (train / "frames.csv").read_bytes(), "interrupted resume changed metrics"
    resumed_metadata = json.loads((resumed / "training.json").read_text())
    assert resumed_metadata["optimizer_resumed"] and resumed_metadata["start_step"] == 1
    checkpoint = train / "model.safetensors"
    base += ["--eval-only", "--checkpoint", str(checkpoint), "--save-linear",
             "--alpha-frame", "0:0", "--alpha-frame", "0:1"]
    causal = args.out / "causal"
    subprocess.run(base + ["--out", str(causal)], check=True)
    assert not list(train.rglob("*.png")), "training evaluation wrote PNGs"
    assert (train / "dev-1/frames.csv").is_file(), "periodic evaluation did not run"
    assert (train / "frames.csv").read_bytes() == (causal / "frames.csv").read_bytes(), "reload metrics changed"
    quality = json.loads((causal / "quality.json").read_text())
    assert quality["alpha_frames"] == [[0, 0], [0, 1]]
    for frame in json.loads((causal / "diagnostics.json").read_text()):
        for lobe in ("diffuse", "specular"):
            gate = frame[f"alpha_{lobe}"]
            path = causal / gate["raw"]
            values = [v for (v,) in struct.iter_unpack("<f", path.read_bytes())]
            assert len(values) == math.prod(quality["extent"])
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
            histogram = [0] * 20
            for value in values:
                product, = struct.unpack("<f", struct.pack("<f", value * 20))
                histogram[min(int(product), 19)] += 1
            assert histogram == gate["summary"]["histogram"]
            if frame["frame"] == 0:
                assert all(v == 0 for v in values), "cold alpha must be zero"
            png = path.with_suffix(".png").read_bytes()
            assert png[:8] == b"\x89PNG\r\n\x1a\n"
            width, height, depth, color = struct.unpack(">IIBB", png[16:26])
            assert [width, height] == quality["extent"] and (depth, color) == (8, 0)
    compared = args.out / "compared"
    subprocess.run(base + ["--out", str(compared), "--control-run", str(causal), "--no-images"], check=True)
    assert not list(compared.glob("*.png"))
    shared = 0
    for source in causal.glob("*-reference.rgbf32"):
        destination = compared / source.name
        assert source.read_bytes() == destination.read_bytes(), "shared reference changed"
        shared += destination.samefile(source)
    assert json.loads((compared / "quality.json").read_text())["shared_reference_frames"] == shared
    with (compared / "frames.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if key.startswith("delta_") and value:
                assert float(value) == 0.0, (key, value)
    for interval in (1, 16):
        output = args.out / f"reset-{interval}"
        subprocess.run(base + ["--out", str(output), "--reset-every", str(interval), "--no-images"], check=True)
        quality = json.loads((output / "quality.json").read_text())
        assert quality["buckets"]["warm"]["learned"] is None
        if interval == 1:
            assert quality["learned"]["temporal_mse"] is None
            assert quality["learned"]["resets"] == quality["frames"]
            assert quality["buckets"]["early"]["learned"] is None
        else:
            # Smoke captures have only two frames: reset-16 must be exactly causal.
            assert quality["sequence_length"] == 2
            assert (output / "frames.csv").read_bytes() == (causal / "frames.csv").read_bytes()
    rejected = subprocess.run(base + ["--out", str(args.out / "mismatched-control"),
        "--reset-every", "1", "--control-run", str(causal)], capture_output=True, text=True)
    assert rejected.returncode != 0 and "control run reset_every differs" in rejected.stderr, rejected.stderr
    print("PASS: metrics-only periodic/final training evaluation, interrupted optimizer/cursor resume, exact reload, alpha maps/histograms, shared reference identity, zero control deltas, reset-1/16, mismatched-control rejection")


if __name__ == "__main__":
    main()
