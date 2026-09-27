#!/usr/bin/env python3
"""CPU-only evaluation contract and clustered-bootstrap tests (standard library)."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from evaluation_metrics import read_frames, same_protocol

spec = importlib.util.spec_from_file_location("bootstrap", Path(__file__).with_name("bootstrap-evaluation.py"))
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def rows(length=32, shifts=(1, 3)):
    result = []
    for sequence, shift in enumerate(shifts):
        for frame in range(length):
            row = {"sequence": str(sequence), "frame": str(frame), "frames_since_reset": str(frame),
                   "reference_sha256": "a" * 64}
            for name in bootstrap.METRICS:
                row[f"baseline_{name}"] = "10"
                row[f"learned_{name}"] = str(10 + shift)
            if frame == 0:
                row["baseline_temporal_mse"] = row["learned_temporal_mse"] = ""
            result.append(row)
    return result


class EvaluationTests(unittest.TestCase):
    def test_cluster_resampling_and_cold_null(self):
        values = rows()
        result = bootstrap.compare(values, values, "baseline", "learned")
        for group in ("all", "cold", "early", "settling", "warm"):
            self.assertEqual(result[group]["psnr"]["difference"], 2)
            # Two correlated sequence clusters give [1, 3], not the tight
            # interval incorrectly obtained by treating 64 frames as independent.
            self.assertEqual(result[group]["psnr"]["ci95"], [1, 3])
            self.assertEqual(result[group]["psnr"]["valid_resamples"], 1000)
        self.assertIsNone(result["cold"]["temporal_mse"]["difference"])
        self.assertIsNone(result["cold"]["temporal_mse"]["ci95"])
        self.assertEqual(result["all"]["temporal_mse"]["frames"], 62)
        self.assertEqual(result, bootstrap.compare(values, values, "baseline", "learned"))

    def test_constant_shift_and_missing_warm(self):
        values = rows(16, (2, 2, 2))
        result = bootstrap.compare(values, values, "baseline", "learned")
        self.assertEqual(result["early"]["flip"]["ci95"], [2, 2])
        self.assertIsNone(result["warm"]["flip"]["difference"])
        self.assertEqual(result["warm"]["flip"]["frames"], 0)

    def test_frame_weighting_with_unequal_temporal_counts(self):
        values = rows(4)
        values[1]["baseline_temporal_mse"] = values[1]["learned_temporal_mse"] = ""
        result = bootstrap.compare(values, values, "baseline", "learned")
        self.assertAlmostEqual(result["all"]["temporal_mse"]["difference"], (2 * 1 + 3 * 3) / 5)

    def test_unpaired_missing_nonfinite_and_mismatched_frames_fail(self):
        values = rows(2)
        for key, value in (("sequence", "99"), ("reference_sha256", "b" * 64),
                           ("frames_since_reset", "99"), ("learned_temporal_mse", "1"),
                           ("learned_psnr", "nan")):
            changed = copy.deepcopy(values)
            changed[0][key] = value
            with self.assertRaises(ValueError):
                bootstrap.compare(values, changed)

    def test_csv_protocol_validation(self):
        values = rows(4)
        quality = {"schema": 2, "sequence_length": 4, "frames": 8, "reset_every": None,
                   "capture": {"captures": [1, 2]}, "extent": [16, 16]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "frames.csv"
            path.with_name("quality.json").write_text(json.dumps(quality))
            def write(data):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                    writer.writeheader()
                    writer.writerows(data)
            write(values)
            self.assertEqual(read_frames(path)[2], values)
            for bad in (values[:-1], values[::-1], values + [values[0]]):
                write(bad)
                with self.assertRaises(ValueError):
                    read_frames(path)
        different = copy.deepcopy(quality)
        different["capture"]["captures"].reverse()
        with self.assertRaises(ValueError):
            same_protocol(quality, different)


if __name__ == "__main__":
    unittest.main()
