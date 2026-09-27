#!/usr/bin/env python3
"""Paired frame-mean differences with 1,000 whole-sequence bootstrap resamples.

Inputs are score-flip.py's enriched frames.csv (reference hashes are required).
Positive differences mean AFTER minus BEFORE, irrespective of metric direction.
"""
import argparse
import json
import math
from pathlib import Path
import random

from evaluation_metrics import BUCKETS, bucket, metric, read_frames, same_protocol, sha256

METRICS = ("psnr", "flip", "temporal_mse", "energy_ratio")


def percentile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lo = math.floor(index)
    hi = math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def compare(before, after, before_prefix="learned", after_prefix="learned", seed=31, resamples=1000):
    if not before or len(before) != len(after):
        raise ValueError("empty or differently sized frame lists")
    keys = ("sequence", "frame", "frames_since_reset", "reference_sha256")
    for left, right in zip(before, after):
        if any(left[key] != right[key] for key in keys):
            raise ValueError("frame identity, reset age or reference hash differs")
        digest = left["reference_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid reference digest")
    sequences = sorted({int(row["sequence"]) for row in before})
    rng = random.Random(seed)
    draws = [rng.choices(sequences, k=len(sequences)) for _ in range(resamples)]
    result = {}
    for group in BUCKETS:
        result[group] = {}
        for name in METRICS:
            # Whole-sequence totals retain frame weighting, even with missing temporal pairs.
            clusters = {sequence: [0.0, 0.0, 0] for sequence in sequences}
            for left, right in zip(before, after):
                if group != "all" and bucket(int(left["frames_since_reset"])) != group:
                    continue
                a, b = metric(left, f"{before_prefix}_{name}"), metric(right, f"{after_prefix}_{name}")
                if (a is None) != (b is None):
                    raise ValueError(f"unpaired missing {name}")
                if a is not None:
                    cluster = clusters[int(left["sequence"])]
                    cluster[0] += a
                    cluster[1] += b
                    cluster[2] += 1
            count = sum(c[2] for c in clusters.values())
            entry = {"frames": count, "sequences": sum(c[2] > 0 for c in clusters.values()),
                     "before": None, "after": None, "difference": None,
                     "ci95": None, "valid_resamples": 0}
            if count:
                entry["before"] = math.fsum(c[0] for c in clusters.values()) / count
                entry["after"] = math.fsum(c[1] for c in clusters.values()) / count
                entry["difference"] = entry["after"] - entry["before"]
                samples = []
                for draw in draws:
                    size = sum(clusters[s][2] for s in draw)
                    if size:
                        samples.append(math.fsum(clusters[s][1] - clusters[s][0] for s in draw) / size)
                entry["valid_resamples"] = len(samples)
                if samples:
                    entry["ci95"] = [percentile(samples, 0.025), percentile(samples, 0.975)]
            result[group][name] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before-prefix", default="learned", choices=("learned", "baseline", "control"))
    parser.add_argument("--after-prefix", default="learned", choices=("learned", "baseline", "control"))
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    before_quality, _, before = read_frames(args.before)
    after_quality, _, after = read_frames(args.after)
    same_protocol(before_quality, after_quality)
    report = {
        "schema": 1, "unit": "whole sequences, paired resampling with replacement",
        "aggregation": "frame-weighted means; undefined pairs excluded, never zero-filled",
        "direction": "after minus before; PSNR higher is better, FLIP/temporal MSE lower is better; energy ratio target is 1",
        "interval": "percentile 95%, linear interpolation", "resamples": 1000, "seed": args.seed,
        "before": {"csv": str(args.before), "sha256": sha256(args.before), "prefix": args.before_prefix},
        "after": {"csv": str(args.after), "sha256": sha256(args.after), "prefix": args.after_prefix},
        "groups": compare(before, after, args.before_prefix, args.after_prefix, args.seed),
    }
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["groups"], indent=2))


if __name__ == "__main__":
    main()
