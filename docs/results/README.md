# Current reconstruction result — 2026-09-26

One architecture, one selected checkpoint. The width-16 recurrent
lobe-separated residual U-Net remains unchanged at **188,160 parameters**.
This updates the single result gallery; earlier evidence remains in Git at
`91c123d`, and retired architectures at `b838674`.

## Outcome

On 256 fresh audit frames, fine-tuning improves PSNR over the previous
**trained checkpoint** by **0.47 dB** (procedural) and **0.69 dB** (objects).
Temporal error falls 7.7% / 7.8%. PSNR improves on 121/128 procedural and
125/128 object frames; worst regressions are 0.26 / 0.19 dB.
These are modest, measured improvements, not a production-quality or DLSS claim.

| Metric | Procedural previous | Now | Object previous | Now |
|---|---:|---:|---:|---:|
| PSNR ↑ | 28.1844 | 28.6494 | 28.9113 | 29.6024 |
| SSIM ↑ | 0.8225 | 0.8370 | 0.7972 | 0.8159 |
| Low-frequency PSNR ↑ | 31.9213 | 32.6576 | 33.1506 | 34.3204 |
| Reset PSNR ↑ | 26.0761 | 26.5433 | 27.3614 | 28.2154 |
| Temporal MSE ↓ | 0.000583 | 0.000538 | 0.000699 | 0.000645 |
| Rejected-history MSE ↓ | 0.004839 | 0.004404 | 0.006179 | 0.005761 |
| Linear RGB MSE ↓ | 0.038519 | 0.037317 | 0.011005 | 0.009646 |
| Relative MSE ↓ | 0.118350 | 0.099908 | 0.138055 | 0.121908 |
| Mean energy ratio (ideal 1) | 1.0134 | 0.9960 | 1.0258 | 1.0000 |
| Detail ratio (ideal 1) | 0.8925 | 0.8924 | 0.8517 | 0.8479 |

Both checkpoints receive identical observations and run their own history from
reset. The fixed guide scores 25.93 / 26.09 dB; the new model beats it by
2.72 / 3.51 dB. All control metrics are retained in [results.json](results.json).
Reset scores cover eight initial frames per set; temporal scores cover 120
adjacent-frame pairs per set. Definitions are in [evaluation.md](../evaluation.md).

The original published audit is now a **regression set**, not an untouched test.
Both checkpoints were rerun on those same frames with the current runtime:

| Original published scenes | Previous PSNR | Now | Previous SSIM | Now |
|---|---:|---:|---:|---:|
| Procedural | 28.2367 | 28.6660 | 0.7937 | 0.8105 |
| Objects | 29.0376 | 29.7915 | 0.7879 | 0.8058 |

Temporal error also decreases there. Detail ratios fall from 0.9500 to 0.9341
and 0.9053 to 0.8993; brightness changes from slightly high to slightly low.
Higher PSNR does not mean every diagnostic or frame improves.

## Data correctness before more training

Asset ids in a sidecar did not prove that earlier captures actually showed
those objects. On the tested RX 7900 XT, Mesa 26.0.3 and 26.0.8 silently omitted
an opaque 18,572-triangle chair. An isolated Mesa 26.2.3 + libdrm 2.4.134 stack
restored it, matching LavaPipe's 17.8% primary-ray coverage. The system driver
was not replaced. Driver/environment details are recorded with the captures.

Object captures now frame authored meshes without the extra spheres/boxes,
reject assets without supported opaque triangles, and measure actual visible
coverage. A separate geometry renderer checks the entire camera trajectory
before expensive path tracing, retrying an obscured camera up to eight times.
It does not consume input/reference sampling or camera history. Per-frame
visibility remains checked during capture and in training provenance.

| Object capture | Frames | Minimum coverage | Mean coverage | Maximum coverage |
|---|---:|---:|---:|---:|
| Training | 512 | 1.87% | 14.55% | 30.63% |
| Development | 64 | 9.42% | 15.59% | 25.91% |
| Fresh audit | 128 | 4.19% | 18.52% | 40.48% |

Coverage is the union of catalog first hits, not an object-only quality score.
One training camera was retried because a foreground light obscured the object.
Two fully non-opaque assets, `B0719H3LG2` and `B071F6VV2T`, were excluded;
the original files remain unmodified. See [catalog.md](../catalog.md).

## Training and selection

The previous checkpoint had 3,500 updates on 64→128 captures. This run starts
from those weights, with fresh Adam state, and trains at 128→256:
4,000 updates, seed 17, two-frame BPTT, learning rate 0.0001 with cosine decay
to 10%. Loss weights remain compressed RGB 1, linear RGB 0.005, coarse linear
RGB 0.01 and temporal change 0.02. No graph or parameter-count change.

| Split | Procedural + object scenes | Frames/scene | Input → output | Reference |
|---|---:|---:|---|---:|
| Fine-tuning | 64 + 64 | 8 | 128×128 → 256×256 | 1,024 spp |
| Development | 8 + 8 | 8 | 128×128 → 256×256 | 4,096 spp |
| Fresh audit | 8 + 8 | 16 | 128×128 → 256×256 | 4,096 spp |

All inputs are 1 spp at matching eight-bounce depth, with projection jitter,
camera/object/light motion, textured materials, gloss, canopy shadows and
output-resolution primary surfaces. Training uses 22 supported ABO families;
development and audit use four each. Scene seeds and families are disjoint
across splits and from the warm-start model's original training.
The fresh audit uses new scene seeds, **not new audit families** relative to
the previous publication.

Selection was frozen at **2026-09-26 05:17:25 UTC**, before final model evaluation.
The final checkpoint has the highest joint development PSNR, **29.65290 dB**
versus 29.00100 at the start. All nine evaluations, including the starting
checkpoint, are recorded. Development detail ratio declined from 0.8549 to
0.8361; this trade-off was disclosed before the audit, not used for audit-based
reselection.

A provisional procedural-only run was stopped after its 500-update evaluation
while object capture was being fixed. It and the 64-update performance smoke
tests are not ancestors of the selected model. Incomplete captures were not
used. The training-source commit was `d74f76c`; upstream subsequently rebased
it as `e45ed32` with a byte-identical source tree. Run manifests retain the
actual execution-time commit ids. The earlier result commit `730e84c` likewise
has the same tree as the now-mainline `91c123d`.

Selected checkpoint (local artifact):
`runs/quality-2026-09-26/train/model.safetensors`

SHA-256:
`5d0c7411c4581a0a8ad99cd87069e4344222dd43020bc28cc0f16a40e47d1321`

[Config](model.transport.ron) · [Procedural frames](procedural.csv) ·
[Object frames](abo.csv) · [Commands, captures, selection and hashes](results.json)

## Reference and runtime checks

Independent 4,096-spp references for each audit's first 16-frame sequence have
pair PSNR 39.10 / 39.70 dB. Input planes agree exactly. Equal-variance
single-reference noise-floor estimates are 42.11 / 42.71 dB; energy B/A is
1.000014 / 0.999903. This check does not cover every scene.

Rust 1.92 workspace tests: 66 passed; three hardware tests are separate.
Formatting, Clippy and three Python catalog-support tests pass.
On the updated Radeon driver, optimized tests pass for CPU/WGSL preparation,
12-frame recurrence/reset/HDR, full-width two-frame loss and parameter gradients
with fused/unfused lowerings, loss reduction, and checkpoint image reload.
The trainer now reuses GPU preparation and avoids unused RGB readbacks; this
changes execution cost, not the learned architecture.

**Vulkan validation is still not clean.** Prior debug Radeon/LavaPipe checks
recorded 20/40 Naga `VUID-StandaloneSpirv-None-10684` Workgroup-array layout
errors despite passing numerical assertions. Those failures remain recorded.
Optimized training/evaluation disable validation; their successful exits do not
establish conformance. This remains a release blocker, with no real-time claim.

Blade `fbb4f28`, Naga `323acfb`, and the user's clean Meganeura checkout
`5253d35` (atop pinned upstream `0dbfcc0`) were used. Blade/Meganeura main
heads were rechecked on September 26 and still match the pins. Exact revisions,
compiler, driver library hashes and source diffs are in the run evidence.

## Pictures and limitations

README shows sequence 0, frame 7 in each fresh set, selected before training.
The six PNGs are byte-for-byte evaluator outputs: previous model, selected
model, reference. Native 256×256 resolution, identical `x/(1+x)` then sRGB
display transform, no retouching. All frames and checkpoints remain in the
local run; full-frame comparison CSVs and selected images are checked in.
`python3 scripts/verify-results.py` checks image hashes, dimensions, full-sequence
scores, checkpoint-selection timing, split membership and visibility evidence.

Specular sparkle, low-frequency blotches and missing detail remain visible.
The model is still trained on a small synthetic/object-in-room corpus, without
production game traces, broad interior coverage, matched external denoisers or
perceptual-video evaluation. Output-resolution primary surfaces are extra
observations, not a fair assumption for a system supplied only LR guides.
The next work should improve data realism and reconstruction within this one
architecture, not accumulate alternative model families.
