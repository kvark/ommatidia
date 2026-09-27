"""Shared, model-independent CSV contract for evaluation postprocessing."""
import csv
import hashlib
import json
import math
from pathlib import Path

BUCKETS = ("all", "cold", "early", "settling", "warm")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bucket(age):
    return "cold" if age == 0 else "early" if age < 8 else "settling" if age < 16 else "warm"


def read_frames(path):
    path = Path(path)
    quality = json.loads(path.with_name("quality.json").read_text())
    if quality.get("schema") != 2:
        raise ValueError("requires evaluation schema 2")
    length, count = quality["sequence_length"], quality["frames"]
    interval = quality["reset_every"]
    if (length <= 0 or count <= 0 or count % length
            or interval is not None and (not isinstance(interval, int) or interval <= 0)):
        raise ValueError("invalid frame count or reset interval")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError("missing or duplicate CSV columns")
        rows = list(reader)
    if len(rows) != count:
        raise ValueError("incomplete/extra evaluation frames")
    for index, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise ValueError("CSV row length differs from header")
        sequence, frame = divmod(index, length)
        age = frame if interval is None else frame % interval
        if tuple(int(row[key]) for key in ("sequence", "frame", "frames_since_reset")) != (sequence, frame, age):
            raise ValueError("frame order, count or reset ages differ from protocol")
    return quality, fields, rows


def metric(row, name):
    value = row[name]
    if value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}")
    return result


def same_protocol(before, after):
    for key in ("capture", "extent", "frames", "sequence_length", "reset_every"):
        if before[key] != after[key]:
            raise ValueError(f"comparison {key} differs")


def means(rows, column):
    result = {}
    for group in BUCKETS:
        values = [metric(row, column) for row in rows
                  if group == "all" or bucket(int(row["frames_since_reset"])) == group]
        values = [value for value in values if value is not None]
        result[group] = {"frames": len(values), "mean": math.fsum(values) / len(values) if values else None}
    return result
