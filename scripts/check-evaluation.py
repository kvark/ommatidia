#!/usr/bin/env python3
"""GPU smoke: training metrics-only, serialized reload, periodic cuts and saved control.

Use tiny disjoint, two-frame captures, e.g. those made by the CI LavaPipe job.
Wrap this command with record-run.py and record the binary and both captures.
"""
import argparse
import csv
import json
from pathlib import Path
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
    subprocess.run(base + ["--data", str(args.train_data), "--out", str(train),
                          "--steps", "2", "--channels", "4", "--eval-every", "1"], check=True)
    checkpoint = train / "model.safetensors"
    base += ["--eval-only", "--checkpoint", str(checkpoint), "--save-linear"]
    causal = args.out / "causal"
    subprocess.run(base + ["--out", str(causal)], check=True)
    assert not list(train.rglob("*.png")), "training evaluation wrote PNGs"
    assert (train / "dev-1/frames.csv").is_file(), "periodic evaluation did not run"
    assert (train / "frames.csv").read_bytes() == (causal / "frames.csv").read_bytes(), "reload metrics changed"
    compared = args.out / "compared"
    subprocess.run(base + ["--out", str(compared), "--control-run", str(causal), "--no-images"], check=True)
    assert not list(compared.glob("*.png"))
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
    print("PASS: metrics-only periodic/final training evaluation, exact reload, zero control deltas, reset-1/16, mismatched-control rejection")


if __name__ == "__main__":
    main()
