#!/usr/bin/env python3
"""Fail-closed checks for the Phase 4 data manifest and capture preflight."""

import copy
import importlib.util
from pathlib import Path
import tempfile
import struct
import unittest

spec = importlib.util.spec_from_file_location("corpus", Path(__file__).with_name("phase4-corpus.py"))
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)


class CorpusTests(unittest.TestCase):
    def test_seed_schedule(self):
        self.assertEqual(corpus.seeds(510001, 4), [510001, 2653991304, 5308496707, 7962993946])

    def test_nested_ancestry_and_capture_not_optimizer_seeds(self):
        value = {"ancestry": [{"capture": {"scene_seeds": [7, 8]}},
                              {"command": ["target/release/ommatidia-data", "--samples", "2", "--seed", "9"]}],
                 "command": ["transport", "--seed", "999", "--samples", "888"]}
        self.assertEqual(corpus.seed_evidence(value), {7, 8, *corpus.seeds(9, 2)})

    def test_only_seed_and_path_change(self):
        original = ["capture", "--samples", "4", "--seed", "7", "--out", "old.omd"]
        changed = corpus.replace(corpus.replace(original, "--seed", 9), "--out", "new.omd")
        self.assertEqual(changed, ["capture", "--samples", "4", "--seed", "9", "--out", "new.omd"])
        self.assertEqual(original[4], "7")

    def test_full_trajectory_visibility(self):
        good = {"scenes": [{"index": i, "kind": "object", "sources": ["abo"],
                            "ids": ["asset"], "families": ["family"], "camera_attempts": 1,
                            "visible_fraction": [0.1] * 64} for i in range(4)]}
        self.assertEqual(corpus.visibility(good, {"family"})["frames"], 256)
        for invalid in (0.0099, float("nan"), float("inf"), 1.1):
            bad = copy.deepcopy(good)
            bad["scenes"][3]["visible_fraction"][63] = invalid
            with self.assertRaises(ValueError):
                corpus.visibility(bad, {"family"})
        bad = copy.deepcopy(good)
        bad["scenes"][3]["visible_fraction"].pop()
        with self.assertRaises(ValueError):
            corpus.visibility(bad, {"family"})
        with self.assertRaises(ValueError):
            corpus.visibility(good, {"holdout"})

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            corpus.write_new(path, {"original": True})
            with self.assertRaises(FileExistsError):
                corpus.write_new(path, {"original": False})
            self.assertEqual(corpus.read(path), {"original": True})

    def test_changed_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            corpus.write_new(path, {})
            item = corpus.identity(path)
            corpus.check_identity(item)
            item["sha256"] = "wrong"
            with self.assertRaises(ValueError):
                corpus.check_identity(item)

    def test_header_size_and_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.omd"
            raw = b"OMMATIDA" + struct.pack("<9I", 2, 2, 128, 128, 2047, 2047, 256, 3, 64)
            raw += bytes(64 - len(raw))
            expected = 64 + 256 * 2 * (128**2 + 256**2) * 27
            with path.open("wb") as stream:
                stream.write(raw)
                stream.truncate(expected)  # sparse; no gigabyte fixture allocation
            self.assertEqual(corpus.header(path)["records"], 256)
            with path.open("r+b") as stream:
                stream.truncate(expected - 1)
            with self.assertRaises(ValueError):
                corpus.header(path)
            with path.open("wb") as stream:
                stream.write(b"not-omd")
            with self.assertRaises(ValueError):
                corpus.header(path)

    def test_finite_reference_admission(self):
        info = {"records": 2, "record_values": 12, "width": 1, "height": 1,
                "scale": 1, "lr_planes": 0, "hr_planes": 1 | (1 << 8) | (1 << 9) | (1 << 10)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "numeric.omd"

            def write(values):
                path.write_bytes(bytes(64) + struct.pack("<24e", *values))

            write([1.0] * 24)
            self.assertEqual(corpus.finite_data(path, info)["finite_records"], 2)
            for invalid in (float("nan"), float("inf"), -1.0):
                values = [1.0] * 24
                values[13] = invalid
                write(values)
                with self.assertRaises(ValueError):
                    corpus.finite_data(path, info)
            write([1.0] * 12 + [0.0] * 3 + [1.0] * 9)
            with self.assertRaises(ValueError):
                corpus.finite_data(path, info)


if __name__ == "__main__":
    unittest.main()
