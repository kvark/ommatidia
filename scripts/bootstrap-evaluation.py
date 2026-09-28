#!/usr/bin/env python3
"""Paired frame means, differences and ratios with whole-sequence bootstraps.

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


def compare(before, after, before_prefix="learned", after_prefix="learned", seed=31, resamples=1000,
            metrics=METRICS):
    if not before or len(before) != len(after):
        raise ValueError("empty or differently sized frame lists")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
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
        for name in metrics:
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
                     "ci95": None, "valid_resamples": 0,
                     "before_ci95": None, "after_ci95": None,
                     "ratio": None, "ratio_ci95": None, "ratio_valid_resamples": 0}
            if count:
                entry["before"] = math.fsum(c[0] for c in clusters.values()) / count
                entry["after"] = math.fsum(c[1] for c in clusters.values()) / count
                entry["difference"] = entry["after"] - entry["before"]
                if entry["before"] > 0:
                    entry["ratio"] = entry["after"] / entry["before"]
                samples = []
                before_samples, after_samples, ratios = [], [], []
                for draw in draws:
                    size = sum(clusters[s][2] for s in draw)
                    if size:
                        samples.append(math.fsum(clusters[s][1] - clusters[s][0] for s in draw) / size)
                        a = math.fsum(clusters[s][0] for s in draw) / size
                        b = math.fsum(clusters[s][1] for s in draw) / size
                        before_samples.append(a)
                        after_samples.append(b)
                        if a > 0:
                            ratios.append(b / a)
                entry["valid_resamples"] = len(samples)
                entry["ratio_valid_resamples"] = len(ratios)
                for key, values in (("ci95", samples), ("before_ci95", before_samples),
                                    ("after_ci95", after_samples), ("ratio_ci95", ratios)):
                    if values:
                        entry[key] = [percentile(values, 0.025), percentile(values, 0.975)]
            result[group][name] = entry
    return result


def final_frames(before, after, sequences, sequence_length,
                 before_prefix="learned", after_prefix="learned", seed=31):
    """Report fixed sequences' final frames individually as well as pooled.

    Individual points prevent a good lighting sequence from hiding a regressing
    one. Only the pooled report has a sequence-bootstrap confidence interval.
    """
    if not sequences or len(sequences) != len(set(sequences)):
        raise ValueError("final-frame sequences must be nonempty and unique")
    if sequence_length <= 0 or any(sequence < 0 for sequence in sequences):
        raise ValueError("invalid final-frame selection")
    selected = []
    for rows in (before, after):
        chosen = [row for row in rows if int(row["sequence"]) in sequences
                  and int(row["frame"]) == sequence_length - 1]
        if sorted(int(row["sequence"]) for row in chosen) != sorted(sequences):
            raise ValueError("missing or duplicate selected final frame")
        selected.append(chosen)
    metrics = (*METRICS, "linear_mse")
    pooled = compare(*selected, before_prefix, after_prefix, seed, metrics=metrics)["all"]
    points = []
    for left, right in zip(*selected):
        values = {}
        for name in metrics:
            a = metric(left, f"{before_prefix}_{name}")
            b = metric(right, f"{after_prefix}_{name}")
            values[name] = {"before": a, "after": b,
                            "difference": b - a if a is not None else None,
                            "ratio": b / a if a is not None and a > 0 else None}
        points.append({"sequence": int(left["sequence"]), "frame": int(left["frame"]),
                       "metrics": values})
    return {"sequences": sorted(sequences), "frames": points, "pooled": pooled,
            "scope": "selected final frames only; individual points have no independent frame uncertainty"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before-prefix", default="learned", choices=("learned", "baseline", "control"))
    parser.add_argument("--after-prefix", default="learned", choices=("learned", "baseline", "control"))
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--final-sequence", type=int, action="append", default=[],
                        help="also report this sequence's last frame (repeatable; select before scoring)")
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
        "ratio_definition": "ratio of paired frame-weighted means, after/before; undefined for nonpositive denominators",
        "before": {"csv": str(args.before), "sha256": sha256(args.before), "prefix": args.before_prefix},
        "after": {"csv": str(args.after), "sha256": sha256(args.after), "prefix": args.after_prefix},
        "groups": compare(before, after, args.before_prefix, args.after_prefix, args.seed),
    }
    if args.final_sequence:
        report["final_frames"] = final_frames(before, after, args.final_sequence,
            before_quality["sequence_length"], args.before_prefix, args.after_prefix, args.seed)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["groups"], indent=2))


if __name__ == "__main__":
    main()
