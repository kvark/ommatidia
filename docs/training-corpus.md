# Phase 4 training corpus

Status: complete, 2026-09-27. This expands data for the existing v4 model; it
does not train a candidate, change architecture, or evaluate confirmation.

The corpus has **200 scenes / 12,800 frames**: the original 40 scenes plus 32 new
scenes for each of static, camera, objects, lights and catalog. Each scene has
64 frames, 1-spp 128² inputs, 256² output surfaces and 1,024-spp references,
matched eight-bounce independent paths and projection jitter.

The [machine-readable manifest](training-corpus.json) contains the ordered 50
training files, scene seeds, hashes and catalog coverage. Admission passes:

- All 12,800 records are finite, with nonnegative/nonblack references. The
  largest reference radiance is 718.5; HDR outliers are retained, not filtered.
- All 2,560 catalog frames meet the 1% visibility floor; the minimum is 1.534%.
  The 40 catalog scenes sample 19 of the existing 22 training families.
- Original training files, development, both audits and protected selections
  are hash-identical to preflight. Confirmation was not evaluated.
- The actual Phase 3 loader checks all 50 file/provenance hashes and all reference
  means, then passes 9,000 crop/augmentation decodes. No optimizer updates or
  model evaluations are involved.

All 40 capture batches succeeded in 84.89 minutes wall time. Full run evidence,
including verifier development failures, is in the [ledger](experiments.md).
The exhaustive data audit took 6.95 minutes; the cold-start mapped-loader
admission check took 9.18 minutes. Neither measures training update throughput.

## Reproduction and admission

`scripts/phase4-corpus.py` derives commands from the recorded
`runs/quality-week-2026-09-26/capture-train-extra-*/manifest.json` commands.
It retains four scenes per capture: eight batches per case, changing **only
the seed and output path**, including reuse of the exact archived executable
(`ad682fd1…`). Its recorded build includes the Naga correction. Mesa 26.2.3
and the RX 7900 XT remain fixed. No new assets or licensing choices are involved.

In case order above, seed bases are 1,010,001 through 1,050,001 in increments
of 10,000; batch `b=0..7` adds `1,000*b`. Each four-scene command generates
actual scene seed `base XOR (i*0x9E3779B9)`, `i=0..3`. Preflight checks actual
seeds, not just bases, against every recorded capture and nested published
ancestry found under `runs/`, `data/` and `docs/`. The initial check finds no
overlap with 305 prior seeds across 943 provenance records. Historical files
without seed provenance cannot independently establish a seed identity; they
are not admitted to the current corpus.

The independent coverage audit confirms the preflight includes all ten current
training captures, five development captures, five captures in each audit,
eight diagnostic records and both published-ancestry reports. Their seeds were
checked before the first new capture. The training pool is also disjoint from
all eight held-out ABO families; unused legacy interior entries in the root
catalog do not enter this corpus.

Catalog uses exactly the existing 22-family training pool, with one object per
scene and target extent 2. Every catalog frame must have measured visible
coverage at least 1%; this is the existing geometric admission threshold, not
a quality-selection threshold. Development and both audits are hash-locked
before capture and checked afterward, without any model evaluation.

```sh
RUSTUP_TOOLCHAIN=1.92.0 python3 scripts/record-run.py runs/v4-phase4/preflight-run \
  --input scripts/phase4-corpus.py --input PLAN.md -- \
  python3 scripts/phase4-corpus.py preflight
RUSTUP_TOOLCHAIN=1.92.0 python3 scripts/record-run.py runs/v4-phase4/capture-run \
  --input scripts/phase4-corpus.py --input runs/v4-phase4/preflight.json -- \
  python3 scripts/phase4-corpus.py capture
RUSTUP_TOOLCHAIN=1.92.0 python3 scripts/record-run.py runs/v4-phase4/verify-run \
  --input scripts/phase4-corpus.py --input runs/v4-phase4/preflight.json -- \
  target/evaluation-env/bin/python scripts/phase4-corpus.py verify
RUSTUP_TOOLCHAIN=1.92.0 python3 scripts/record-run.py runs/v4-phase4/loader-build -- \
  cargo +1.92.0 build --release --locked -p ommatidia-train --bin check-corpus
RUSTUP_TOOLCHAIN=1.92.0 python3 scripts/record-run.py runs/v4-phase4/loader-check \
  --input target/release/check-corpus --input docs/training-corpus.json \
  --input runs/v4-phase4/corpus.json -- \
  target/release/check-corpus docs/training-corpus.json
```

Run directories are immutable; choose a fresh recorder directory on retry.
Capture resumes only completed, matching batches and refuses to overwrite
partial outputs. Preflight and the final corpus manifest are also create-only.
Every individual batch gets its own manifest and ledger row. The final
`runs/v4-phase4/corpus.json` gives the complete admission report; the checked-in
manifest preserves the ordered training paths, scene membership, hashes,
transport provenance, finite/nonnegative/nonblack reference checks, and
full-trajectory catalog coverage. `check-corpus` then opens all files with
the **actual Phase 3 memory-mapped loader**, verifies hashes and corpus means,
and decodes cold/middle/final crops at all corners and centre with gains
0.25, 1 and 4. It does no training or inference.

The new captures occupy 42,077,260,288 bytes (39.19 GiB), and the complete corpus
52,596,575,360 bytes (48.98 GiB). This is larger than system RAM; keep the mapped
crop loader and its immutable-input contract. Cold-start hashing/reference scans
are I/O-bound and take minutes. No pre-tiling or training-throughput claim for
200 scenes is implied by this data-admission step. Phase 5 still starts with the
single-scene sanity fit and investigation of the finite loss spikes from Phase 3.
