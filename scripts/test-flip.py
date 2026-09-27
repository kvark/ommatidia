#!/usr/bin/env python3
"""Test display conversion and FLIP against pinned NVlabs reference images.

--reference-dir DIR fetches missing official fixtures, checks every hash and
requires the reference mean within 1e-4. Files and their BSD-3 license remain
outside git. Without the flag, only offline wrapper tests run.
"""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
import urllib.request

import flip_evaluator as flip
import numpy as np
from evaluation_metrics import sha256

spec = importlib.util.spec_from_file_location("score_flip", Path(__file__).with_name("score-flip.py"))
score_flip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score_flip)

REVISION = "b475eb4bf394ab877c42166c9eb0a84a02cc5b14"
FIXTURES = {
    "images/reference.png": "2ef97595be5d17ea33bddb272afcf67b76842d91c625611c249e455072d4d653",
    "images/test.png": "554504b6c0183d66fb931a5f61b204ed805dfa81d3d5f28921cfcf38514243a3",
    "src/tests/correct_ldrflip_cpp.png": "ab3cd035de828d571c455bb8a4cd235fa1258aa8f7bd418f92a88e56f85823a7",
    "src/tests/test.py": "7bb4f36bfe16344316cf98430090fc496ddb761eb5ee08e279455a6932a53753",
    "LICENSE": "13b955078ffb4a3215757038ab2e5fcf0cc66349d0990faec9adba8ad034e578",
}
REFERENCE_DIR = None


class FlipTests(unittest.TestCase):
    def test_fixed_display_transform(self):
        values = np.array([0, 0.0031308 / (1 - 0.0031308), 1, 1000], dtype=np.float32)
        transformed = score_flip.display(values)
        np.testing.assert_allclose(transformed, [0, 0.040449936, 0.735356983, 0.999560696], atol=2e-7)
        self.assertEqual(transformed.dtype, np.float32)

    def test_linear_reader_rejects_invalid_radiance_and_length(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.rgbf32"
            for values in ([0, 1], [0, float("nan"), 1], [0, -1, 1]):
                np.array(values, dtype="<f4").tofile(path)
                with self.assertRaises(ValueError):
                    score_flip.read_image(path, [1, 1])
            np.array([0, 1, 2], dtype="<f4").tofile(path)
            self.assertEqual(score_flip.read_image(path, [1, 1]).shape, (1, 1, 3))

    def test_identical_image_has_zero_error(self):
        image = np.full((16, 16, 3), 0.5, dtype=np.float32)
        self.assertLess(score_flip.ldr_flip(image, image), 1e-7)

    def test_official_ldr_images(self):
        if REFERENCE_DIR is None:
            self.skipTest("pass --reference-dir to validate the official fixtures")
        for name, digest in FIXTURES.items():
            path = REFERENCE_DIR / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with urllib.request.urlopen(f"https://raw.githubusercontent.com/NVlabs/flip/{REVISION}/{name}", timeout=60) as response:
                    with path.open("xb") as stream:
                        stream.write(response.read())
            self.assertEqual(sha256(path), digest, name)
        reference = flip.load(str(REFERENCE_DIR / "images/reference.png"))
        test = flip.load(str(REFERENCE_DIR / "images/test.png"))
        actual = score_flip.ldr_flip(reference, test)
        self.assertLessEqual(abs(actual - 0.159691), 1e-4)
        colored, _, _ = flip.evaluate(reference, test, "LDR", parameters={"ppd": 67.0})
        expected = flip.load(str(REFERENCE_DIR / "src/tests/correct_ldrflip_cpp.png"))
        print(f"official LDR-FLIP mean: {actual:.12f}; expected 0.159691; abs error {abs(actual - 0.159691):.12g}", flush=True)
        # The release CPU wheel need not reproduce the upstream C++ build's
        # quantized magma colors bitwise. This is a diagnostic, not the mean gate.
        self.assertEqual(colored.shape, expected.shape)
        difference = np.abs(np.rint(255 * colored).astype(int) - np.rint(255 * expected).astype(int))
        print(f"upstream magma diagnostic: {np.count_nonzero(difference)}/{difference.size} channels differ; max {difference.max()}/255", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference-dir", type=Path)
    args, remaining = parser.parse_known_args()
    REFERENCE_DIR = args.reference_dir
    unittest.main(argv=[__file__, *remaining])
