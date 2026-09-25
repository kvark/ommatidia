# Current reconstruction result — 2026-09-25

One architecture, one training run, one selected checkpoint. This replaces the
historical experiment galleries; Git preserves those at `e0922c6`.

## Outcome

Across the two untouched 64-frame audits, PSNR improves by **3.06 dB**
(procedural) and **3.12 dB** (object scenes) over the fixed
guide. Temporal error falls 54.4% / 58.1%. All 128 frames improve
in PSNR; minimum gains are 1.11 / 1.03 dB. These are results on this small
corpus, not a claim about production games or DLSS.

| Metric | Procedural guide | Model | Object guide | Model |
|---|---:|---:|---:|---:|
| PSNR ↑ | 25.18 | 28.24 | 25.92 | 29.04 |
| SSIM ↑ | 0.6753 | 0.7937 | 0.6538 | 0.7879 |
| Low-frequency PSNR ↑ | 29.25 | 32.02 | 29.93 | 32.75 |
| Reset PSNR ↑ | 20.61 | 25.57 | 21.25 | 26.55 |
| Temporal MSE ↓ | 0.001476 | 0.000673 | 0.001544 | 0.000647 |
| Rejected-history MSE ↓ | 0.012934 | 0.007570 | 0.015236 | 0.008642 |
| Linear RGB MSE ↓ | 0.045935 | 0.025643 | 0.033650 | 0.016518 |
| Relative MSE ↓ | 0.230135 | 0.088795 | 0.188087 | 0.078339 |
| Mean energy ratio (ideal 1) | 0.9996 | 1.0102 | 1.0012 | 1.0235 |
| Detail ratio (ideal 1) | 1.1552 | 0.9500 | 1.1094 | 0.9053 |

Both methods run their own history from reset. Rejected-history regions cover
1.53% / 1.62% of non-reset pixels, identical between methods. Reset scores cover
four first frames per audit; temporal scores cover sixty adjacent-frame pairs.
Energy and detail ratios are averaged per frame. Metric definitions are in
[evaluation.md](../evaluation.md).

## Training, data and selection

The 188,160-parameter, width-16 residual U-Net completed 4,000 Adam updates,
seed 7, two-frame BPTT, learning rate 0.0003 with cosine decay to 10%.
The objective weights are compressed RGB 1, linear RGB 0.005, coarse linear
RGB 0.01 and temporal change 0.02. This was not a hyperparameter sweep.

| Split | Procedural + object scenes | Frames/scene | Input → output | Reference |
|---|---:|---:|---|---:|
| Training | 32 + 16 | 8 | 64×64 → 128×128 | 512 spp |
| Development | 4 + 4 | 8 | 64×64 → 128×128 | 1,024 spp |
| Final audit | 4 + 4 | 16 | 128×128 → 256×256 | 4,096 spp |

All inputs are 1 spp at matching eight-bounce depth. Capture includes projection
jitter, camera/object/light motion, textured materials, gloss, canopy shadows
and exact output-resolution primary surfaces. A canopy-camera bug that could
produce all-black frames was fixed before these captures. The trainer refuses
entirely black references.

Training uses 24 ABO families; development and final audit use separate sets of
four families each. The catalog's eight holdout ids were sorted lexicographically:
the first four went to development, the last four to the final audit. Their
exact ids are in the dataset provenance in results.json. Scene-seed and family intersections were checked across all
six captures. This does not establish that every catalog object occupies a
useful visible area; the preselected images are still dominated by primitives.

Selection used development mean frame PSNR only. The best checkpoint is update
3,500 (27.02056 dB); update 4,000 is 27.01284 dB and has slightly better SSIM and
temporal error. The final audit was run only after selecting update 3,500.
All eight development evaluations are retained in [results.json](results.json).

Checkpoint (local run artifact):
`runs/focused-2026-09-25/train/step-3500.safetensors`

SHA-256:
`6c58120fbcb5f18978058749c616d1fab1b51dc77885cc852468b9f1b76fb367`

[Config](model.transport.ron) · [Procedural frames](procedural.csv) ·
[Object frames](abo.csv) · [Commands, captures and hashes](results.json)

## Reference and runtime checks

Two independent 4,096-spp references for the first 16-frame sequence of each audit
have pair PSNR 36.89 dB (procedural) / 43.36 dB (object). Input planes agree exactly.
The equal-variance half-noise estimates are 39.90 / 46.37 dB. Energy B/A is
1.000010 / 0.999961. Reference noise remains finite, but is substantially below
the measured reconstruction error. This noise audit does not cover every scene.

Rust 1.92 workspace tests: 63 passed, three hardware tests separately exercised.
Formatting, all-target checking and Clippy pass. Radeon RX 7900 XT / RADV and
LavaPipe pass the full-width two-frame numerical loss/gradient checks with
fused/unfused lowerings. CPU/WGSL features, 12-frame recurrence, HDR, reset,
loss reduction, bit-exact checkpoint image reload and capture-cache independence
are covered.

**Validation still fails.** The debug Radeon gradient run recorded 20 validation
errors; the full LavaPipe run recorded 40. All were the known Naga
`VUID-StandaloneSpirv-None-10684` Workgroup-array layout failure. Child tests
exited zero, but the recorder correctly marks these runs failed. Release
training/evaluation disable validation by Meganeura's build default, so their
successful exits are not evidence of Vulkan conformance. No release or speed
claim is made.

The run used Blade `fbb4f28`, Naga `323acfb`, and the user's clean Meganeura
checkout `5253d35` (the GGUF parameter-reuse commit atop pinned upstream
`0dbfcc0`). Exact hashes and the training source-patch hash are recorded.
Current remote Blade/Meganeura heads were checked and match the repository pins.

## Pictures and remaining limitations

README uses frame 7 of sequence 0 in both sets, preselected before evaluation.
The six PNGs are byte-for-byte copies of evaluator output, with one fixed
`x/(1+x)` then sRGB transform. All frames exist in the local audit directories;
the full-frame CSVs, selected images and their hashes are checked in.
`python3 scripts/verify-results.py` checks the published evidence.

The model still leaves visible specular sparkle and low-frequency blotches.
Mean energy is high by 1.0% / 2.3%, and detail ratios remain below one. The data
lacks production game traces, broad interior coverage and an asset-visibility
audit. Output-resolution primary surfaces are extra observations, not free
ground truth for a fair comparison with a renderer that supplies only LR guides.
There is no matched DLSS/OIDN comparison. More training on this fixed small
corpus has largely plateaued; the next quality work should improve data coverage
and resolution within this architecture, not add another model family.
