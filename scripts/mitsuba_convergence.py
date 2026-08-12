#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mitsuba==3.9.1",
#   "numpy>=2.0",
# ]
# ///
"""Calibrate reference noise against Mitsuba's production path integrator."""

import argparse
import time
from pathlib import Path

import mitsuba as mi
import numpy as np


def compressed(image: np.ndarray) -> np.ndarray:
    positive = np.maximum(image, 0.0)
    return positive / (1.0 + positive)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--spp", type=int, nargs="+", default=[1024, 4096, 16384])
    parser.add_argument("--out", type=Path, default=Path("target/mitsuba-audit"))
    args = parser.parse_args()
    if len(args.spp) < 2 or any(spp <= 0 for spp in args.spp):
        parser.error("--spp needs at least two positive sample counts")

    mi.set_variant("llvm_ad_rgb")
    description = mi.cornell_box()
    description["sensor"]["film"].update(
        width=args.size,
        height=args.size,
        rfilter={"type": "box"},
    )
    # Mitsuba's Cornell helper already uses max_depth 8, matching the
    # reference-depth policy used for Ommatidium datasets.
    scene = mi.load_dict(description)
    args.out.mkdir(parents=True, exist_ok=True)

    images: dict[int, np.ndarray] = {}
    for spp in args.spp:
        started = time.perf_counter()
        image = np.array(mi.render(scene, spp=spp, seed=args.seed), copy=True)
        images[spp] = image
        mi.util.write_bitmap(str(args.out / f"cornell-{spp}spp.exr"), image)
        mi.util.write_bitmap(str(args.out / f"cornell-{spp}spp.png"), image)
        print(
            f"{spp:>6} spp  {time.perf_counter() - started:>7.2f}s  "
            f"range {float(image.min()):.4f} .. {float(image.max()):.4f}"
        )

    reference_spp = args.spp[-1]
    reference = compressed(images[reference_spp])
    for spp in args.spp[:-1]:
        error = compressed(images[spp]) - reference
        mse = float(np.mean(error * error))
        psnr = -10.0 * np.log10(mse)
        print(f"{spp:>6} vs {reference_spp}: MSE {mse:.9g}, PSNR {psnr:.2f} dB")


if __name__ == "__main__":
    main()
