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
assert manifest["previous_model"]["version"] == manifest["model"]["version"]
assert manifest["previous_model"]["parameters"] == manifest["model"]["parameters"]
selection = manifest["selection"]
best = max(selection["development"], key=lambda row: row["quality"]["learned"]["psnr"])
assert best["update"] == manifest["model"]["selected_update"]
assert best["checkpoint"] == manifest["model"]["checkpoint"]
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
    previous = sum(float(r["previous_psnr"]) for r in rows) / len(rows)
    assert math.isclose(previous, audit["previous_quality"]["learned"]["psnr"], abs_tol=2e-5), name
    assert sum(float(r["learned_psnr"]) > float(r["previous_psnr"]) for r in rows) == audit["improved_frames_over_previous"]
    assert audit["quality"]["capture"] == audit["previous_quality"]["capture"]
    assert selection["selected_at"] < manifest["commands"][f"evaluate-audit-{name}"]["started"]
    assert audit["quality"]["speed_claim"] is False
for name in ["train-abo-visible", "dev-abo", "audit-abo"]:
    dataset = manifest["datasets"][name]
    coverage = [v for s in dataset["catalog"]["scenes"] for v in s["visible_fraction"]]
    assert len(coverage) == dataset["capture"]["records"], name
    assert all(0.01 <= v <= 1.0 for v in coverage), name
    assert math.isclose(min(coverage), dataset["capture"]["minimum_catalog_visible_fraction"], abs_tol=1e-8), name
for a, b in [("train", "dev"), ("train", "audit"), ("dev", "audit"),
             ("warm_start_training", "dev"), ("warm_start_training", "audit")]:
    for field in ["seeds", "families"]:
        assert not set(manifest["datasets"][a+"_membership"][field]) & set(manifest["datasets"][b+"_membership"][field])
readme = (root / "README.md").read_text()
shown = re.findall(r'src="([^"]+)"', readme)
assert len(shown) == len(manifest["images"]) == 6
assert set(shown) == {e["path"] for e in manifest["images"]}
for audit in manifest["audits"].values():
    previous = audit["previous_quality"]["learned"]
    current = audit["quality"]["learned"]
    assert f'{previous["psnr"]:.2f} → **{current["psnr"]:.2f} dB**' in readme
    assert f'{previous["ssim"]:.3f} → {current["ssim"]:.3f}' in readme
    assert f'−{100 * (1 - current["temporal_mse"] / previous["temporal_mse"]):.1f}%' in readme
for path in shown:
    assert (root / path).is_file(), path
    assert any(e["path"] == path for e in manifest["images"]), path

quality = json.loads((root / "docs/quality-iteration.json").read_text())
panel = root / "docs/quality-iteration.png"
assert hashlib.sha256(panel.read_bytes()).hexdigest() == quality["panel_sha256"]
transfer = quality["historical_development"]
assert transfer["frames"] == 640 and transfer["sequences"] == 10
control = json.loads((root / "docs/phase1-results.json").read_text())
trial = json.loads((root / "docs/crop-context.json").read_text())
assert trial["frames_per_protocol"] == 640 and trial["sequences"] == 10
assert trial["crop_frame_selections"] == 40
assert hashlib.sha256((root / trial["benchmark"]).read_bytes()).hexdigest() == trial["benchmark_sha256"]
panel = "docs/crop-context.png"
assert hashlib.sha256((root / panel).read_bytes()).hexdigest() == trial["panel_sha256"]
assert f"]({panel})" in readme

metrics = (("psnr", 2), ("flip", 5), ("temporal_mse", 6))
v3_row = ["Corrected v3"]
for key, decimals in metrics:
    v3, f1 = transfer["causal"]["all"][key]
    assert v3 == control["protocols"]["causal"]["metrics"]["all"][key]["after"]
    baseline = trial["experiments"]["context"]["seeds"]["1"]["causal"]["metrics_by_age"]["all"][key][0]
    assert math.isclose(f1, baseline, rel_tol=1e-8)
    v3_row.append(f"{v3:.{decimals}f}")
assert "| " + " | ".join(v3_row) + " |" in readme

for label, arm, index in (
    ("F1 · 8 scenes · 32×32", "context", 0),
    ("F1 · 8 scenes · 48×48", "context", 1),
    ("F1 · 16 scenes · 32×32", "breadth", 1),
):
    row = [label]
    seeds = trial["experiments"][arm]["seeds"]
    for key, decimals in metrics:
        values = [seeds[i]["causal"]["metrics_by_age"]["all"][key][index] for i in ("1", "2")]
        row.append(" / ".join(f"{v:.{decimals}f}" for v in values))
    assert "| " + " | ".join(row) + " |" in readme

for seed in ("1", "2"):
    context, breadth = (trial["experiments"][arm]["seeds"][seed] for arm in ("context", "breadth"))
    assert context["control_weights_sha256"] == breadth["control_weights_sha256"]
    assert context["supervised_pixel_gradients"] == breadth["supervised_pixel_gradients"]
    for result in (context, breadth):
        causal = result["causal"]["metrics_by_age"]
        reset = result["reset16"]["metrics_by_age"]
        passed = (causal["all"]["flip"][2] < 0 and causal["warm"]["psnr"][3][0] > 0
                  and min(causal["cold"]["psnr"][2], reset["cold"]["psnr"][2]) >= -0.1)
        assert result["passes_primary"] == passed
        assert reset["warm"]["psnr"] == [None, None, None, None]
print("Published v3 evidence, paired F1 development results and comparison pixels agree.")
