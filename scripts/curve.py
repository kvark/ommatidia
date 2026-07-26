#!/usr/bin/env python3
"""Extract the training and held-out curves from a curriculum log.

The trainer interleaves loss lines with periodic held-out scores, which is
readable while a run is going but awkward to compare across runs afterwards.
This pulls both out per run so two objectives can be lined up at matched steps
rather than only at whatever step each happened to finish on.

    scripts/curve.py runs/curriculum.log
"""

import re
import sys
from collections import OrderedDict

RUN = re.compile(r"^== (\S+)\s+\((\w+), (\d+) steps\)")
STEP = re.compile(r"^step\s+(\d+): loss ([0-9.eE+-]+)\s+\(([0-9.]+) steps/s, (\d+)m")
HELD = re.compile(
    r"held-out over (\d+) crops in [0-9.]+s: nearest ([0-9.eE+-]+), "
    r"network ([0-9.eE+-]+), ([+-][0-9.]+) dB"
)


def parse(path):
    runs = OrderedDict()
    current = None
    for line in open(path):
        if m := RUN.search(line):
            current = m.group(1)
            runs[current] = {
                "objective": m.group(2),
                "planned": int(m.group(3)),
                "loss": [],
                "held": [],
            }
        elif current is None:
            continue
        elif m := STEP.search(line):
            runs[current]["loss"].append(
                (int(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4)))
            )
        elif m := HELD.search(line):
            # Held-out lines carry no step of their own, so they take the step
            # of the most recent loss line, which is the one they follow.
            step = runs[current]["loss"][-1][0] if runs[current]["loss"] else 0
            runs[current]["held"].append(
                (step, int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
            )
    return runs


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    runs = parse(sys.argv[1])
    if not runs:
        print("no runs found in that log")
        return 1

    for name, run in runs.items():
        loss = run["loss"]
        print(f"\n=== {name} ({run['objective']}, planned {run['planned']} steps)")
        if loss:
            rate = loss[-1][2]
            print(
                f"    {loss[-1][0]} steps done, loss {loss[0][1]:.4f} -> "
                f"{loss[-1][1]:.4f}, {rate:.2f} steps/s, {loss[-1][3]}m elapsed"
            )
        if run["held"]:
            print("    step      crops   nearest    network        dB")
            for step, crops, near, net, db in run["held"]:
                print(f"    {step:>8}  {crops:>5}  {near:.6f}  {net:.6f}  {db:+7.2f}")
            best = max(run["held"], key=lambda h: h[4])
            print(f"    best {best[4]:+.2f} dB at step {best[0]}")

    # Line the runs up by step, so a comparison is at matched compute rather
    # than at whatever step each one happened to end on. Runs scored on
    # different cadences simply leave gaps — requiring a step common to *every*
    # run would drop the table entirely as soon as one run used a different
    # --eval-every.
    if len(runs) > 1:
        names = list(runs)
        steps = sorted({h[0] for n in names for h in runs[n]["held"]})
        if steps:
            print("\n=== by step")
            print("      step  " + "  ".join(f"{n:>18}" for n in names))
            for step in steps:
                cells = []
                for n in names:
                    db = next((h[4] for h in runs[n]["held"] if h[0] == step), None)
                    cells.append(f"{db:+18.2f}" if db is not None else " " * 18)
                print(f"    {step:>6}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
