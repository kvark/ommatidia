#!/usr/bin/env python3
"""Coverage and encoded-video checks for the matched comparison renderer."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("comparisons", Path(__file__).with_name("render-comparisons.py"))
comparisons = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparisons)


class ReviewTests(unittest.TestCase):
    def test_pages_cover_every_frame_once_in_order(self):
        pages = comparisons.review_pages(64)
        self.assertEqual(len(pages), 16)
        self.assertTrue(all(len(page) == 4 for page in pages))
        self.assertEqual([frame for page in pages for frame in page], list(range(64)))
        self.assertEqual(comparisons.review_pages(5), [[0, 1, 2, 3], [4]])

    def test_decoded_video_matches_publication_contract(self):
        stream = {"width": 768, "height": 288, "nb_read_frames": "64", "avg_frame_rate": "24/1"}
        with patch.object(comparisons.subprocess, "check_output",
                          return_value=json.dumps({"streams": [stream]})) as probe:
            self.assertEqual(comparisons.verify_video(Path("clip.mp4"), 64, 24), stream)
        self.assertIn("-count_frames", probe.call_args.args[0])

    def test_rejects_incomplete_or_retimed_video(self):
        stream = {"width": 768, "height": 288, "nb_read_frames": "64", "avg_frame_rate": "24/1"}
        cases = [{}, {"streams": []}, {"streams": [stream, stream]}]
        for key, value in [("width", 767), ("height", 287), ("nb_read_frames", "63"),
                           ("nb_read_frames", "65"), ("avg_frame_rate", "30/1")]:
            cases.append({"streams": [{**stream, key: value}]})
        for case in cases:
            with self.subTest(case=case):
                with patch.object(comparisons.subprocess, "check_output", return_value=json.dumps(case)):
                    with self.assertRaises(ValueError):
                        comparisons.verify_video(Path("clip.mp4"), 64, 24)


if __name__ == "__main__":
    unittest.main()
