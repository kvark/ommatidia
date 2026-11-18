#!/usr/bin/env python3
"""Verify that paired OMD captures vary only the sparse path samples."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_pair(first, second):
    metadata = [json.loads(path.with_suffix(".transport.json").read_text())
                for path in (first, second)]
    offsets = [m.pop("input_sample_offset") for m in metadata]
    meta = metadata[0]
    require(meta == metadata[1], "capture metadata changed beyond input noise offset")
    require(meta["input_estimator"] == "independent-paths" and meta["input_frames"] == 1
            and meta["matching_path_depth"] and meta["reference_from"] is None,
            "expected 1-spp paths and independently rendered matched-depth targets")
    require(abs(offsets[0] - offsets[1]) >= meta["records"], "input path-frame ranges overlap")
    headers = []
    for path in (first, second):
        with path.open("rb") as stream:
            headers.append(stream.read(64))
    require(headers[0] == headers[1], "different layouts")
    magic, version, scale, w, h, lr, hr, count, source, length = struct.unpack_from("<8s9I", headers[0])
    require(magic == b"OMMATIDA" and version == 2 and source == 3 and count == meta["records"]
            and count == length * len(meta["scene_seeds"]), "invalid sequence header")
    # OMD stores beauty, diffuse, specular and emission before its primary guides.
    radiance_mask = 1 | (1 << 8) | (1 << 9) | (1 << 10)
    require(all(mask & radiance_mask == radiance_mask for mask in (lr, hr)), "missing lobe planes")
    channels = (3, 1, 3, 3, 3, 1, 2, 2, 3, 3, 3)
    def width(mask):
        return sum(n for i, n in enumerate(channels) if mask & (1 << i))
    pixels = w * h
    lr_end = width(lr) * pixels
    record = lr_end + width(hr) * pixels * scale**2
    require(all(p.stat().st_size == 64 + count * record * 2 for p in (first, second)),
            "truncated or extra records")
    a, b = (np.memmap(path, dtype="<u2", mode="r", offset=64, shape=(count, record))
            for path in (first, second))
    changed = [0, 0]
    for frame in range(count):
        for values in (a[frame], b[frame]):
            require(np.isfinite(values.view("<f2")).all(), f"nonfinite data in frame {frame}")
        require(np.array_equal(a[frame, lr_end:], b[frame, lr_end:]), f"HR mismatch at {frame}")
        # LR direct emission and all geometry, materials, motion and jitter follow specular.
        require(np.array_equal(a[frame, 9*pixels:lr_end], b[frame, 9*pixels:lr_end]),
                f"observed LR guides differ at {frame}")
        lobe_diff = [int(np.count_nonzero(a[frame, i*pixels:(i+3)*pixels] !=
                                          b[frame, i*pixels:(i+3)*pixels])) for i in (3, 6)]
        require(sum(lobe_diff) > 0, f"identical input lobes at {frame}")
        changed = [x + y for x, y in zip(changed, lobe_diff)]
    identities = []
    for p in (first, second):
        with p.open("rb") as stream:
            identities.append(hashlib.file_digest(stream, "sha256").hexdigest())
    return {"paths": [str(first), str(second)], "sha256": identities, "offsets": offsets,
            "frames": count, "sequence_length": length, "scene_seeds": meta["scene_seeds"],
            "low_extent": [w, h], "reference_spp": 4 * meta["canonical_frames"],
            "lr_changed_diffuse_and_specular_values": changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, nargs=2, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reports = [check_pair(*paths) for paths in args.pair]
    scenes = [s for r in reports for s in r["scene_seeds"]]
    require(len(scenes) == len(set(scenes)), "repeated scene across split or case")
    with args.out.open("x") as stream:
        json.dump(reports, stream, indent=2)
        stream.write("\n")
    print(f"Verified {len(scenes)} distinct scenes and {sum(r['frames'] for r in reports)} frame pairs")


if __name__ == "__main__":
    main()
