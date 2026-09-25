#!/usr/bin/env bash
# Fresh, matched-depth moving captures for the single recurrent model.
set -euo pipefail
cd "$(dirname "$0")/.."
capture_root="${CAPTURE_ROOT:-data/catalog-recurrent}"
capture_bin="${CAPTURE_BIN:-target/release/ommatidia-data}"
catalog_path="${CATALOG:-data/catalog/catalog.json}"
if [[ -e "$capture_root" ]]; then
  echo "Choose a fresh CAPTURE_ROOT: $capture_root" >&2
  exit 2
fi
mkdir -p "$capture_root"
common=(
  --lr 64x64 --scale 2 --canonical-frames 256 --canonical-bounces 8
  --input-frames 1 --sequence-frames 8 --hr-gbuffer --split-radiance
  --projection-jitter --random-camera-motion 0.03 --object-motion 0.03
  --light-motion 0.05 --canopy --textures --gloss --ground-patches 8
  --catalog "$catalog_path" --catalog-kind object --catalog-target-extent 2
)
if [[ -n "${DEVICE_ID:-}" ]]; then common+=(--device-id "$DEVICE_ID"); fi
for split in train holdout; do
  count=24; seed=80000
  if [[ "$split" = holdout ]]; then count=8; seed=81000; fi
  python3 scripts/record-run.py "$capture_root/record-$split" \
    --input "$capture_bin" --input "$catalog_path" -- \
    "$capture_bin" "${common[@]}" --out "$capture_root/$split.omd" \
    --catalog-split "$split" --samples "$count" --seed "$seed"
done
