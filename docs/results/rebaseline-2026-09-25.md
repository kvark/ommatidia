# Correctness and retained-recipe rebaseline — 2026-09-25

## Scope

This experiment used the local legacy-training branch at `cf3da1f`, not the
newer transport/field implementation on remote `main`. A pre-push fetch revealed
149 divergent remote commits ending at `e924d7b`, including removal of this
trainer. The results and evaluator repairs below therefore establish a legacy
control; they do not evaluate or supersede those newer architectures. They are
preserved at `e2b7e4c` (branch `backup/rebaseline-pre-main-2026-09-25` locally).
Rebasing onto `e924d7b` retains the new architectures and does not restore the
retired trainer. The commands below require that historical revision; its reset
repair and rejected-history evaluator are recorded there. The active transport
evaluator already resets both methods at sequence boundaries. Its predictor
already sees history, mixtures are linear, and training supports short BPTT,
so the legacy experiment order below is not a new roadmap for that model.

Re-establish a trustworthy comparison before changing architecture or data.
The control is `dlss-linear-guide-exact-reset-r3-b16`: 319,068 parameters,
three U-Net levels, radius-three linear taps, deterministic-guide mixing,
and surface-validated previous-output mixing. This is an experimental control,
not the published spatial checkpoint. No architecture, loss, or training-data
changes were made for the rerun. It used the already-updated dependency
checkouts without editing their sources.

The current Cargo pins already name upstream Blade/Meganeura main as checked
on this date. Local path patches, however, are what actually compile:

| dependency | effective revision |
|---|---|
| Blade | `fbb4f28c4869e81ae15de58925945b423b9c1ac5` |
| Meganeura | `5253d35d7428652832e5c4774ca2a7db4975a0b8` |
| Naga | `323acfb729a00c3030e362faa27252afe6368792` |

The Meganeura checkout contains the GGUF-parameter-reuse commit on top of the
declared `0dbfcc0` main pin. Its tracked worktree and Blade's were clean.
Historical checkpoints lack equivalent effective-revision manifests, so this
comparison cannot attribute an improvement to one particular backend fix.
The GPU is an RX 7900 XT (`0x744c`), RADV Mesa 26.0.3-1ubuntu1; Rust is 1.95.0.
Training and testing artifacts are under `runs/rebaseline-2026-09-25/`.

## Numerical checks and the remaining validation failure

`kernel_training_matches_reference` exercises the retained model's full
three-level b16/r3 graph at 8x8 input, batch two, including jitter, guide,
history, and mixed valid/rejected pixels. The head is deliberately nonzero:
initializing it to zero would conceal the backbone's gradients.

Four f64 directional finite-difference probes check autodiff independently.
The GPU check compares loss, positive head predictions, and all 53 parameter
gradient tensors with Meganeura's f64 interpreter. It poisons non-parameter
buffers with NaNs and runs both fused and unfused dispatches, each with the
actual loss-only training graph and an instrumented graph exposing predictions.
The latter alone could hide lifetime/reuse bugs. All four variants passed
numerically on RADV and LavaPipe. This is stronger than observing a falling
loss, but small-tile parity is not proof for every production shape or GPU.

Vulkan module validation still **fails** with
`VUID-StandaloneSpirv-None-10684`. A standalone Naga-only reproducer, with no
Ommatidia or Meganeura operations, is enough:

```wgsl
var<workgroup> scratch: array<f32, 4>;
@compute @workgroup_size(4)
fn main(@builtin(local_invocation_index) i: u32) {
    scratch[i] = f32(i);
    workgroupBarrier();
}
```

Writing SPIR-V 1.3 with workgroup zero-initialization disabled, then running
`spirv-val --target-env vulkan1.1` (SPIRV-Tools 2026.1), rejects `ArrayStride`
on the workgroup array.
Naga's array-type writer emits that decoration unconditionally. This violates
the [Vulkan shader-interface layout rules](https://docs.vulkan.org/spec/latest/chapters/interfaces.html).
The runnable source, Cargo files, input hashes, and failure log are retained
in `runs/rebaseline-2026-09-25/shader-repro/`.

The debug LavaPipe oracle process exits successfully after its numerical checks
but emits 40 validation errors. The new recorder correctly marks the run failed
and returns a failure status. CI now records both runtime and model-gradient
tests this way. The backend fix is still outstanding; no validation suppression
or dependency-cache edit was used. Ordinary release training does not enable
Meganeura's debug validation and cannot establish a validation-clean result.

## Evaluation repairs

The recurrent evaluator previously reset its stored output at sequence
boundaries only when recurrence was disabled. A reset frame could therefore
consume the last frame of an unrelated scene wherever reprojection accepted its
surfaces. It now clears history for **every** sequence boundary, while still
rendering and scoring recurrent reset frames. A unit regression covers retaining
within-sequence history and clearing both output and index state at resets.

Valid-history temporal error intentionally excludes rejected pixels. Evaluation
now separately reports spatial compressed-RGB MSE and pixel coverage on rejected
history, using the actual per-subpixel validity mask supplied to reconstruction.
This includes disocclusions **and** out-of-frame/crop motion, not just newly
revealed surfaces. Reset frames remain a separate score. An end-to-end CPU test
puts all reconstruction errors in rejected pixels: temporal error remains zero
while the new metric reports them. Multi-frame recovery time is still unmeasured.

The running trainer was compiled before these evaluator repairs. Its periodic
scores are diagnostics, not the final comparison below; they do not influence
the optimizer or the fixed 8,000-step stopping point. Both old and new final
weights must be scored with the repaired evaluator.

## Protocol

The retained architecture and initializer seed remain unchanged: 8,000 Adam
updates, batch eight, 64x64 input crops, seed zero, cosine learning rate
`3e-4` to `1e-5`, previous-output teacher, four-frame sequences. Neither
`--rollout-training` nor confidence/temporal auxiliary losses are enabled.
The only sidecar difference from the retained model is an explicit
`mix_confidence_weight: 0.0`, which is also the old sidecar's default.

Dataset SHA-256:

- `dlss-native-motion-train-exact-256x4.omd`:
  `84cb8fcc873c86a7c0cedaf38839f5a5b106f995771d0f89f3a668eb6f3f889a`
- `dlss-native-motion-holdout-exact-32x4.omd`:
  `60d1351afb94cad2178b4e3bf9445084f5677ccc899a2eef434a7f2dc1ed2b19`
- `dlss-native-abo-holdout-8x4.omd`:
  `5cb96cae1a0c10e622b535554e911f25ab8de3a9e21d68d54514b75e0a798dde`

These are retained capture bytes, not newly rendered data. Hashes identify them
but do not retroactively establish their renderer-source provenance or physical
correctness. The eight-family ABO set is also a regression holdout, not an
untouched audit: it has already informed earlier experiments and calibration.

Training command (choose an unused run prefix when repeating):

```sh
python3 scripts/record-run.py runs/rebaseline-2026-09-25/train \
  --input data/dlss-native-motion-train-exact-256x4.omd \
  --input data/dlss-native-motion-holdout-exact-32x4.omd -- \
  cargo run --release -p ommatidia-train --locked -- \
  --data data/dlss-native-motion-train-exact-256x4.omd \
  --eval-data data/dlss-native-motion-holdout-exact-32x4.omd \
  --steps 8000 --batch 8 --tile 64 --base-channels 16 \
  --lr 3e-4 --lr-final 1e-5 --prediction kernel \
  --reconstruction-base sample --kernel-radius 3 --demodulate \
  --history-frames 4 --temporal-features variance \
  --previous-output --guide-mix --device-id 0x744c --seed 0 \
  --eval-crops 384 --eval-every 2000 --checkpoint-every 2000 --log-every 250 \
  --out runs/rebaseline-2026-09-25/baseline
```

`record-run.py` requires Python 3.11+ and a new output directory. It records
the command, selected environment, compiler version, Cargo-resolved package
paths, repository HEADs/status/tracked patches, and explicit input hashes.
Small inputs (up to 1 MiB) and the recorder itself are copied beside the log;
large datasets/checkpoints are not duplicated. Untracked source files are
listed, not automatically archived: pass any needed ones with `--input`.
This is optional orchestration; model execution and data loading remain Rust.

## Final matched comparison

Training completed in 1,472 seconds plus final evaluation. Both checkpoints
were then reloaded on the same current backend and scored with repaired resets
at 128x128 input / 256x256 output. All 32 procedural sequences and eight ABO
sequences were visited: 96 and 24 non-reset full frames, plus 32 and eight
separately scored reset frames. No guide/history calibration was applied.
PSNR is in compressed radiance; temporal columns are improvements over the
deterministic temporal guide (positive is better). Energy is linear luminance
relative to the reference, ideally 1.0.

| set | model | PSNR | SSIM | LF PSNR | energy | fine temporal gain | LF temporal gain |
|---|---|---:|---:|---:|---:|---:|---:|
| procedural | TAA guide | 25.48 | 0.7721 | 28.94 | 0.948 | — | — |
| procedural | old 8k | 25.84 | 0.7590 | 29.63 | 0.933 | -0.12 dB | +0.61 dB |
| procedural | new 8k | **27.93** | **0.7867** | **33.70** | **0.967** | **+1.53 dB** | **+5.06 dB** |
| ABO | TAA guide | 33.58 | 0.9435 | 40.26 | 0.998 | — | — |
| ABO | old 8k | 33.22 | 0.9333 | 39.87 | 0.995 | -0.32 dB | -0.01 dB |
| ABO | new 8k | **34.43** | **0.9458** | **40.72** | 1.002 | **+1.36 dB** | **+3.72 dB** |

Retraining improves procedural PSNR by **2.09 dB** and ABO by **1.21 dB**
against the old weights under the same corrected evaluation. Low-frequency
PSNR improves by 4.07 and 0.85 dB. The new checkpoint beats its guide spatially
and temporally on both regression sets without changing the architecture or
training bytes. The old weights no longer suffice to judge what this recipe
can learn on the current stack.

The added rejected-history score also improves, on identical geometry masks:

| set | rejected coverage | guide MSE | old MSE | new MSE |
|---|---:|---:|---:|---:|
| procedural | 1.30% (81,498 pixels) | 0.006748 | 0.005759 | **0.004051** |
| ABO | 1.03% (16,172 pixels) | 0.003353 | 0.002732 | **0.002172** |

Reset PSNR rises from 23.36 to 26.28 dB procedurally and from 32.07 to
33.06 dB on ABO. This is not an across-every-metric release pass: ABO reset
relative MSE remains worse than the guide (0.01409 versus 0.01275), reset
SSIM is essentially tied (0.9351 versus 0.9352), and procedural mature energy
remains 3.3% low. The example previews show improved highlights and smoother
broad regions, but still substantial smoothing/mottling relative to reference.

Both saved optimizers report 8,000 updates and nonzero second moments in all
53 tensors (`optimizer-state/output.log`). The old backbone was not simply
entirely frozen. This one-seed rerun does not isolate which backend correction
caused the gain, and old negative architecture experiments are now provisional.
No claim of new-scene generalization follows from these already-inspected sets;
catalog membership also needs a visible-asset-coverage audit during recapture.

The new experimental control is `runs/rebaseline-2026-09-25/baseline`.
Weight SHA-256 is
`584545710ebdd7f646e491e5d12c424c59042c7926ae2d83f1202a437a293625`;
the old comparison weights are
`5dd441d6d27fba2737527b69e0f31d1fddb9b883d8cb4d9040d0f9e775503b42`.
Full commands, sidecar hashes, logs, and source diffs are in `old-eval/` and
`new-eval/`; paired first-sequence previews are in `old-images/` and
`new-images/`, including their `audit/` directories. These are local run
artifacts, not a newly published release.

To re-score either stem with this protocol:

```sh
cargo run --release -p ommatidia-train --locked -- \
  --data data/dlss-native-motion-train-exact-256x4.omd \
  --eval-data data/dlss-native-motion-holdout-exact-32x4.omd \
  --audit-data data/dlss-native-abo-holdout-8x4.omd \
  --eval-only --eval-tile 128 --eval-crops 96 --seed 0 --device-id 0x744c \
  --out runs/rebaseline-2026-09-25/baseline
```

## Verification and decision

- All 164 non-ignored workspace tests pass; formatting and strict workspace
  Clippy pass.
- All four model-gradient variants pass numerically on RADV and LavaPipe.
- All five GPU runtime parity tests pass numerically, but emit 50 Vulkan
  validation errors and are correctly marked failed by the recorder.
- Recorder smoke checks cover success, child failure, validation-only failure,
  refusing an existing output directory, and small-input/source snapshots.

Keep this retrain as the research control, with the published model unchanged.
Fix the upstream shader-layout validation before claiming a clean deployment
gate. Then use isolated history-visible-predictor and linear-mixture comparisons,
longer and more diverse clips, independent references, and fresh scene families.
Do not enlarge the backbone to compensate for an obsolete training baseline.
