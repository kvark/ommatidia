#!/usr/bin/env python3
"""Small independent fixtures for the full-precision crop scorer."""
import copy
import hashlib
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
    def test_crop_bootstrap_uses_sequences_not_independent_frames(self):
        rows = []
        for sequence, ratio in enumerate((1, 3)):
            for frame in range(32):
                rows.append({"sequence": sequence, "frames_since_reset": frame, "kind": "smooth",
                             "before": {"squared_sum": 12, "values": 12, "gradient_squared_sum": 0, "gradient_values": 12},
                             "after": {"squared_sum": 12 * ratio, "values": 12, "gradient_squared_sum": 0, "gradient_values": 12}})
        result = scorer.bootstrap_regions(rows, 2)
        for age in ("all", "cold", "early", "settling", "warm"):
            entry = result["groups"][age]["smooth"]["metrics"]
            self.assertEqual(entry["mse"]["ratio"], 2)
            self.assertEqual(entry["mse"]["ratio_ci95"], [1, 3])
            self.assertEqual(entry["mse"]["difference_ci95"], [0, 2])
            self.assertIsNone(entry["gradient_mse"]["ratio_ci95"])
            self.assertEqual(entry["gradient_mse"]["difference_ci95"], [0, 0])
            self.assertIsNone(result["groups"][age]["edge"]["metrics"]["mse"]["ratio"])
        self.assertEqual(result, scorer.bootstrap_regions(rows, 2))

    def test_crop_bootstrap_area_weighting_missing_strata_and_bad_pairs(self):
        rows = [{"sequence": i, "frames_since_reset": 0, "kind": "smooth",
                 "before": {"squared_sum": n, "values": n, "gradient_squared_sum": n, "gradient_values": n},
                 "after": {"squared_sum": n * ratio, "values": n, "gradient_squared_sum": n * ratio, "gradient_values": n}}
                for i, n, ratio in [(0, 1, 1), (1, 3, 3)]]
        result = scorer.bootstrap_regions(rows, 3)["groups"]
        self.assertEqual(result["cold"]["smooth"]["metrics"]["mse"]["ratio"], 2.5)
        self.assertIsNone(result["warm"]["smooth"]["metrics"]["mse"]["ratio_ci95"])
        for key, value in [("values", 2), ("squared_sum", float("nan")), ("gradient_squared_sum", -1)]:
            bad = copy.deepcopy(rows)
            bad[0]["after"][key] = value
            with self.assertRaises(ValueError):
                scorer.bootstrap_regions(bad, 3)

    def test_development_crops_are_valid_and_identify_all_captures(self):
        root = Path(__file__).resolve().parents[1]
        benchmark = json.loads((root / "docs/dev-crops.json").read_text())
        scorer.validate_benchmark(benchmark)
        self.assertEqual(sum(d["sequences"] for d in benchmark["datasets"]), 10)
        self.assertEqual(len(benchmark["regions"]), 16)
        self.assertEqual(sum(len(r["frames"]) for r in benchmark["regions"]), 40)
        self.assertEqual(benchmark["early_reference_extension"]["frames"], [3])
        self.assertTrue(all(3 in r["frames"] for r in benchmark["regions"]))
        self.assertEqual({r["kind"] for r in benchmark["regions"]}, {"smooth", "edge", "texture"})
        for capture in benchmark["datasets"]:
            self.assertEqual(len(bytes.fromhex(capture["sha256"])), 32)
            self.assertTrue(Path(capture["path"]).name.startswith("dev-"))

    def test_development_crops_match_the_available_archived_definitions(self):
        root = Path(__file__).resolve().parents[1]
        benchmark = json.loads((root / "docs/dev-crops.json").read_text())
        reports = [root / path for path in benchmark["source_reports"]]
        if not all(path.is_file() for path in reports):
            self.skipTest("historical reports are workstation artifacts")
        def expand(regions):
            return {(r["name"], r["kind"], r["sequence"], frame, tuple(r["rect"]))
                    for r in regions for frame in r["frames"]}
        expected = set()
        for path in reports:
            self.assertEqual(scorer.sha256(path), benchmark["source_report_sha256"][str(path.relative_to(root))])
            source = json.loads(path.read_text())["benchmark"]
            expected |= expand(source["regions"])
            self.assertEqual([(Path(d["path"]).resolve(), d["sequences"]) for d in source["datasets"]],
                             [((root / d["path"]).resolve(), d["sequences"]) for d in benchmark["datasets"]])
        self.assertEqual({entry for entry in expand(benchmark["regions"]) if entry[3] != 3}, expected)
        for name, digest in benchmark["early_reference_extension"]["reference_png_sha256"].items():
            if (root / name).is_file():
                self.assertEqual(scorer.sha256(root / name), digest)

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
            self.assertIsNotNone(result["groups"]["age"]["cold"]["smooth"])
            self.assertIsNotNone(result["groups"]["age"]["early"]["smooth"])
            self.assertIsNone(result["groups"]["age"]["warm"]["smooth"])
            for frame in range(2):
                (before / f"000-{frame:03}-base.rgbf32").write_bytes(struct.pack("<48f", *([1.0] * 48)))
            cut = scorer.score(self.benchmark(), before, after, before_role="base", reset_every=1)
            self.assertIsNone(cut["groups"]["age"]["early"]["smooth"])
            self.assertAlmostEqual(cut["groups"]["age"]["cold"]["smooth"]["ratio"]["mse"], 0.25)
            (after / "000-001-reference.rgbf32").write_bytes(struct.pack("<48f", *([0.01] * 48)))
            with self.assertRaises(ValueError):
                scorer.score(self.benchmark(), before, after)

    def frozen_candidate(self, root):
        data = root / "data.omd"
        data.write_bytes(b"dataset")
        benchmark = self.benchmark()
        benchmark["datasets"][0]["sha256"] = scorer.sha256(data)
        benchmark_path = root / "benchmark.json"
        benchmark_path.write_text(json.dumps(benchmark))
        checkpoint, config, executable = [root / name for name in ["model.safetensors", "model.transport.ron", "transport"]]
        for path in [checkpoint, config, executable]:
            path.write_bytes(path.name.encode())
        build_path = root / "build.json"
        build = {"status": "complete", "exit_code": 0, "validation_errors": 0,
                 "finished": "2026-09-26T12:00:00+00:00", "repositories": {str(root): {
                     "head": "source-revision", "status": "", "tracked_diff_sha256": hashlib.sha256(b"").hexdigest()}}}
        build_path.write_text(json.dumps(build))
        selection = {"schema": 1, "selected_at": "2026-09-26T12:01:00+00:00",
                     "checkpoint": str(checkpoint), "checkpoint_sha256": scorer.sha256(checkpoint),
                     "config": str(config), "config_sha256": scorer.sha256(config),
                     "inference_executable_sha256": scorer.sha256(executable),
                     "inference_source_commit": "source-revision", "build_manifest": str(build_path),
                     "build_manifest_sha256": scorer.sha256(build_path), "benchmark_sha256": scorer.sha256(benchmark_path)}
        selection_path = root / "selection.json"
        selection_path.write_text(json.dumps(selection))
        output = root / "images"
        output.mkdir()
        (output / "quality.json").write_text(json.dumps({"history_mode": "causal", "learned": {"frames": 2, "resets": 1}}))
        manifest = {"cwd": str(root), "status": "complete", "exit_code": 0, "validation_errors": 0,
                    "started": "2026-09-26T12:02:00+00:00",
                    "command": [str(executable), "--eval-only", "--save-linear", "--eval-data", str(data),
                                "--checkpoint", str(checkpoint), "--out", str(output)],
                    "inputs": [{"path": str(path), "sha256": scorer.sha256(path)} for path in
                               [data, benchmark_path, checkpoint, config, executable, build_path, selection_path]]}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return benchmark_path, benchmark, selection_path, selection, manifest

    def test_frozen_selection_matches_evaluated_artifacts_and_chronology(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark_path, benchmark, selection_path, selection, manifest = self.frozen_candidate(root)
            scorer.verify_run(root, benchmark_path, benchmark)
            result = scorer.verify_selection(root, selection_path, benchmark_path, benchmark)
            self.assertEqual(result["checkpoint_sha256"], selection["checkpoint_sha256"])
            self.assertIn("not a quality", result["scope"])
            # Both predeclared audit hashes may belong to one frozen candidate.
            confirmation = root / "confirmation.json"
            confirmation.write_text(json.dumps({**benchmark, "purpose": "confirmation"}))
            selection["confirmation_benchmark_sha256"] = scorer.sha256(confirmation)
            selection_path.write_text(json.dumps(selection))
            manifest["inputs"][-1]["sha256"] = scorer.sha256(selection_path)
            manifest["inputs"].append({"path": str(confirmation), "sha256": scorer.sha256(confirmation)})
            (root / "manifest.json").write_text(json.dumps(manifest))
            scorer.verify_selection(root, selection_path, confirmation, benchmark)

    def test_frozen_selection_rejects_changed_or_unrecorded_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark_path, benchmark, selection_path, _, manifest = self.frozen_candidate(root)
            for item in manifest["inputs"][1:]:
                path = Path(item["path"])
                original = path.read_bytes()
                with self.subTest(changed=path.name):
                    path.write_bytes(original + b" ")
                    with self.assertRaises(ValueError):
                        scorer.verify_selection(root, selection_path, benchmark_path, benchmark)
                    path.write_bytes(original)
                with self.subTest(unrecorded=path.name):
                    bad = copy.deepcopy(manifest)
                    bad["inputs"].remove(item)
                    (root / "manifest.json").write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):
                        scorer.verify_selection(root, selection_path, benchmark_path, benchmark)
                    (root / "manifest.json").write_text(json.dumps(manifest))

    def test_frozen_selection_rejects_bad_dates_source_or_evaluation(self):
        mutations = ["late", "before-build", "naive-time", "wrong-source", "dirty-build",
                     "wrong-checkpoint", "duplicate-checkpoint", "reset-command", "periodic-reset-command", "reset-report", "incomplete-report"]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                benchmark_path, benchmark, selection_path, selection, manifest = self.frozen_candidate(root)
                if mutation in ["late", "before-build", "naive-time"]:
                    selection["selected_at"] = {"late": "2026-09-26T12:03:00+00:00",
                                                "before-build": "2026-09-26T11:59:00+00:00",
                                                "naive-time": "2026-09-26T12:01:00"}[mutation]
                elif mutation == "wrong-source":
                    selection["inference_source_commit"] = "different-revision"
                elif mutation == "dirty-build":
                    build_path = Path(selection["build_manifest"])
                    build = json.loads(build_path.read_text())
                    build["repositories"][str(root)]["status"] = " M source.rs"
                    build_path.write_text(json.dumps(build))
                    selection["build_manifest_sha256"] = scorer.sha256(build_path)
                    next(item for item in manifest["inputs"] if item["path"] == str(build_path))["sha256"] = scorer.sha256(build_path)
                elif mutation == "wrong-checkpoint":
                    manifest["command"][manifest["command"].index("--checkpoint") + 1] = "different.safetensors"
                elif mutation == "duplicate-checkpoint":
                    manifest["command"].extend(["--checkpoint", selection["checkpoint"]])
                elif mutation == "reset-command":
                    manifest["command"].append("--reset-history")
                elif mutation == "periodic-reset-command":
                    manifest["command"].extend(["--reset-every", "16"])
                else:
                    quality_path = root / "images/quality.json"
                    quality = json.loads(quality_path.read_text())
                    if mutation == "reset-report":
                        quality["history_mode"] = "reset-every-frame diagnostic"
                    else:
                        quality["learned"]["frames"] = 1
                    quality_path.write_text(json.dumps(quality))
                selection_path.write_text(json.dumps(selection))
                next(item for item in manifest["inputs"] if item["path"] == str(selection_path))["sha256"] = scorer.sha256(selection_path)
                (root / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    scorer.verify_selection(root, selection_path, benchmark_path, benchmark)


if __name__ == "__main__":
    unittest.main()
