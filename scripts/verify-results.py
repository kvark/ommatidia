#!/usr/bin/env python3
"""Check that README evidence matches the recorded, full-sequence evaluation."""
import csv
import hashlib
import json
import math
import re
import struct
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "docs/results/results.json").read_text())
assert manifest["model"]["version"] == 3
assert manifest["model"]["parameters"] == 188160
for entry in manifest["images"]:
    path = root / entry["path"]
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"], path
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    assert list(struct.unpack(">II", data[16:24])) == entry["size"], path
    assert entry["sequence"] == 0 and entry["frame"] == 7
for name, audit in manifest["audits"].items():
    with (root / audit["frames_csv"]).open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == audit["quality"]["learned"]["frames"]
    assert [(int(r["sequence"]), int(r["frame"])) for r in rows] == [
        (s, f) for s in range(audit["sequences"]) for f in range(audit["sequence_length"])
    ], name
    for model in ["baseline", "learned"]:
        mean = sum(float(r[f"{model}_psnr"]) for r in rows) / len(rows)
        assert math.isclose(mean, audit["quality"][model]["psnr"], abs_tol=2e-5), name
    assert audit["quality"]["speed_claim"] is False
readme = (root / "README.md").read_text()
shown = re.findall(r'src="([^"]+)"', readme)
assert set(shown) == {e["path"] for e in manifest["images"]}
for path in shown:
    assert (root / path).is_file(), path
    assert any(e["path"] == path for e in manifest["images"]), path
print("README images, hashes, dimensions and full-sequence PSNR agree.")
