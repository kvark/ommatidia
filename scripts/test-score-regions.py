#!/usr/bin/env python3
"""Small independent fixtures for the full-precision crop scorer."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("score_regions", Path(__file__).with_name("score-regions.py"))
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


class RegionTests(unittest.TestCase):
    def benchmark(self):
        return {"schema": 1, "extent": [4, 4], "sequence_length": 2,
                "datasets": [{"name": "static", "path": "data.omd", "sha256": "placeholder", "sequences": 1}],
                "regions": [{"name": "wall", "kind": "smooth", "sequence": 0, "frames": [0, 1], "rect": [1, 1, 2, 2]}]}

    def test_crop_uses_only_declared_pixels_and_compresses_once(self):
        prediction = [100.0] * 48
        for row in range(1, 3):
            for column in range(1, 3):
                i = (row * 4 + column) * 3
                prediction[i:i + 3] = [1.0] * 3
        result = scorer.crop_error(prediction, [0.0] * 48, [4, 4], [1, 1, 2, 2])
        self.assertEqual(result, {"squared_sum": 3.0, "values": 12,
                                  "gradient_squared_sum": 0.0, "gradient_values": 12})

    def test_gradient_penalizes_lost_edges_and_spurious_noise(self):
        edge = [0.0] * 3 + [1.0] * 3 + [0.0] * 3 + [1.0] * 3
        self.assertEqual(scorer.crop_error(edge, edge, [2, 2], [0, 0, 2, 2])["gradient_squared_sum"], 0)
        for prediction, reference in [([0.5] * 12, edge), (edge, [0.5] * 12)]:
            self.assertAlmostEqual(scorer.crop_error(prediction, reference, [2, 2], [0, 0, 2, 2])["gradient_squared_sum"], 1.5)

    def test_invalid_regions_fail(self):
        for key, value in [("rect", [3, 3, 2, 2]), ("sequence", 1), ("frames", []), ("frames", [0, 0]), ("frames", [2]), ("kind", "selected-after-evaluation")]:
            benchmark = self.benchmark()
            benchmark["regions"][0][key] = value
            with self.assertRaises(ValueError):
                scorer.validate_benchmark(benchmark)

    def test_run_requires_predeclared_benchmark_and_exact_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data.omd"
            data.write_bytes(b"dataset")
            benchmark = self.benchmark()
            benchmark["datasets"][0]["sha256"] = scorer.sha256(data)
            path = root / "benchmark.json"
            path.write_text(json.dumps(benchmark))
            manifest = {"cwd": str(root), "status": "complete", "validation_errors": 0,
                        "command": ["transport", "--eval-only", "--save-linear", "--eval-data", "data.omd", "--out", "images"],
                        "inputs": [{"path": str(p), "sha256": scorer.sha256(p)} for p in [data, path]]}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            self.assertEqual(scorer.verify_run(root, path, benchmark)[0], root / "images")
            for field, value in [("status", "failed"), ("validation_errors", 1), ("inputs", manifest["inputs"][:1])]:
                bad = copy.deepcopy(manifest)
                bad[field] = value
                manifest_path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    scorer.verify_run(root, path, benchmark)
            manifest_path.write_text(json.dumps(manifest))
            data.write_bytes(b"different data")
            with self.assertRaises(ValueError):
                scorer.verify_run(root, path, benchmark)

    def test_full_precision_pair_and_reference_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before, after = root / "before", root / "after"
            before.mkdir()
            after.mkdir()
            for frame in range(2):
                prefix = f"000-{frame:03}"
                for directory, value in [(before, 1.0), (after, 1.0 / 3.0)]:
                    (directory / f"{prefix}-learned.rgbf32").write_bytes(struct.pack("<48f", *([value] * 48)))
                    (directory / f"{prefix}-reference.rgbf32").write_bytes(bytes(48 * 4))
            result = scorer.score(self.benchmark(), before, after)
            self.assertAlmostEqual(result["groups"]["kind"]["smooth"]["ratio"]["mse"], 0.25)
            self.assertIsNone(result["groups"]["kind"]["smooth"]["ratio"]["gradient_mse"])
            (after / "000-001-reference.rgbf32").write_bytes(struct.pack("<48f", *([0.01] * 48)))
            with self.assertRaises(ValueError):
                scorer.score(self.benchmark(), before, after)


if __name__ == "__main__":
    unittest.main()
