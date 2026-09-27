# Closed experiments and the published v3 runtime

[The quality sprint](quality-week.md) is closed without promoting a checkpoint.
[PLAN.md](../../PLAN.md) replaces it. No archived architecture is a runtime option.

| Tag | Commit | Purpose |
|---|---|---|
| `archive/experiments-2026-09-25` | `e0922c639194eefff09334c800b99971de7c81af` | Retired architectures, preserved in Git only. |
| `archive/v3-published-runtime` | `049a7bde6bea1a61bac45c3b5966163ddb252467` | Pre-fix v3 source underlying the retained README gallery. |

## Reproduce the retained gallery

The local archive `runs/archive/v3-published-runtime/` contains `transport`, the
published checkpoint/config, the original `build/` manifest and source patches,
and `archive.json` with every file's SHA-256. The executable was copied, not
rebuilt. Its SHA-256 is
`156864c9bcf887666e5b73a6d5e56850b7974e3b3e19cdc464c2d6c5b5451303`.
Checkpoint SHA-256 remains
`5d0c7411c4581a0a8ad99cd87069e4344222dd43020bc28cc0f16a40e47d1321`.

Source is the tag above plus `build/source-3.patch`: linear f32 export and the
local Blade path, without decoder changes. Its original build manifest is
`runs/quality-week-2026-09-26/initial-linear-build/manifest.json`, SHA-256
`9b82259c03f02f81e7b138dff4ad3b0052deb51214bea603e5bdc58866160bf0`.
The other dependency source diffs are empty. A normal sibling checkout of Blade
at `fbb4f28` needs no absolute-path adjustment; Meganeura is `ee3aea4`, Naga is
`323acfb` without the later compiler patch. This historical binary is for
reproduction only, not a new Vulkan-conformance claim.

Run the commands in [results.json](../results/results.json), replacing
`target/release/transport` with the archived binary and `--out` with a fresh
directory. Use the recorded Mesa driver environment and original data/weights.
For example, from the repository root:

```sh
export LD_LIBRARY_PATH="$PWD/target/quality-mesa-26.2.3/extracted/usr/lib/x86_64-linux-gnu"
export VK_ICD_FILENAMES="$PWD/target/quality-mesa-26.2.3/extracted/usr/share/vulkan/icd.d/radeon_icd.json"
python3 scripts/record-run.py runs/reproduce-published-procedural \
  --input runs/archive/v3-published-runtime/transport \
  --input runs/archive/v3-published-runtime/archive.json -- \
  runs/archive/v3-published-runtime/transport --eval-only \
  --eval-data runs/quality-2026-09-26/data/audit-procedural.omd \
  --checkpoint runs/archive/v3-published-runtime/model.safetensors \
  --out runs/reproduce-published-procedural/eval --device-id 0x744c
```

Phase 0 reran both published checkpoints on both fresh audit sets (512 frame
evaluations). All reported numerical metrics match exactly and all 1,536 PNGs
are byte-identical to their originals, including the six README images.
Evidence: `runs/v4-phase0/published-reproduction/` and
`runs/v4-phase0/published-reproduction.json`. This verifies historical results;
it is not a v4 result, a new checkpoint selection, or a confirmation evaluation.

The model/data/binary archives remain workstation artifacts, not Git LFS assets
or public downloads. Published hashes and source tags do not imply otherwise.
Phase 1 separately archives the **corrected v3 control** and its new evaluator;
do not confuse that control with this pre-fix gallery runtime.
