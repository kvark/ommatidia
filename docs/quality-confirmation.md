# Confirmation audit protocol

The first [frozen audit](quality-benchmark.json) missed the smooth-crop gate:
44.8% lower MSE, versus the required 50%. It remains a regression/diagnostic
set. Its frames, crops and failed selection record are not changed or discarded.
No later checkpoint may be selected on that audit or called an untouched result
on it. The one-week scope and quality thresholds in [TASK.md](../TASK.md) remain.

Before evaluating any follow-up candidate, capture a separate confirmation set
using the same five cases, two independent scenes per case, 64 consecutive
frames, 1-spp 128×128 inputs and 4,096-spp 256×256 references. Preserve eight-bounce
paths, canopy, projection jitter, split radiance and observed HR primary guides.
Use the same pinned dependencies and isolated Mesa 26.2.3 driver.

| Case | Seed | Additional settings |
|---|---:|---|
| Static | 810001 | No textures, gloss or motion |
| Camera | 820001 | Gloss, textures, camera motion 0.01 |
| Objects | 830001 | Gloss, object motion 0.02 |
| Lights | 840001 | Gloss, light motion 0.02 |
| Catalog | 850001 | Gloss, camera motion 0.01, one audit-only object, extent 2 |

The ten resulting scene seeds were checked against all current captures and
the complete published warm-start ancestry before capture; there is no overlap.
The catalog continues to use the existing four-family audit pool, not new asset
families. Verify measured visibility throughout every trajectory before admitting
the captures. Do not replace an inconvenient scene based on prediction quality.

Lock dataset hashes and smooth/edge/texture rectangles after **reference-only**
inspection, before any follow-up candidate evaluation. Include cold starts and
frames 31 and 63; use frame-specific rectangles where geometry moves. Keep the
same compressed, unquantized RGB MSE and gradient definitions and pixel weighting.
Evaluate all 640 frames, both scenes per case, including temporal metrics and
matched full-length videos. Illustrated views remain frame 31 of global
sequences 0, 2 and 8, and full 64-frame videos for sequences 0, 2, 4, 6 and 8.

Use `export-references --data FILE [--data FILE ...] --out NEW_DIRECTORY` to
export the first, middle and last references of every sequence. It does not load
weights or create a GPU context. Its static-audit check matches all six reference
PNGs and all six linear f32 files from the original evaluator byte-for-byte
(`reference-export-check/`). This avoids rendering a candidate while choosing
regions. Keep the exact dataset order when assigning global sequence ids.

The complete captures are now locked in [quality-confirmation.json](quality-confirmation.json),
SHA-256 `06f3c023c2d94d31f8d6c765bd222a8befae868ce9408586d2c165a3817cc67a`:
35 regions and 96 crop/frame pairs. Reference-only overlay review corrected the
late chair-cloth rectangle before the lock; camera movement had taken its
original lower rows off the cloth. No candidate outputs were computed or viewed
for this selection. Minimum catalog visibility is 12.05%, over every frame;
the sampled families are B0718ZKMW1 and B0719WQH8S, disjoint from training/dev.

Select one candidate on development, recording weights, config, executable and
source hashes before either confirmation evaluation or a new regression run.
Report **both** the original frozen audit and confirmation outcomes. Do not use a
favorable confirmation draw to hide the original missed gate, or change any
threshold, weight or crop after candidate inspection. No candidate is promoted
on this protocol alone.
