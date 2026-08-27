#!/usr/bin/env bash
# Unattended catalog corpus capture: 16,384-spp references, then 16-frame 1-spp inputs.
# Train and hold-out are separate files so --eval-data cannot leak an asset id.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$ROOT/target/release/ommatidia-data}"
CAT="${CAT:-$ROOT/data/catalog/catalog.json}"
DEVICE="${DEVICE_ID:-}"
LOG="${LOG:-$ROOT/data/catalog/capture.log}"
COMMON=(
  --lr 128x128 --scale 2
  --canonical-bounces 8 --hr-gbuffer --split-radiance
)
if [[ -n "$DEVICE" ]]; then
  COMMON+=(--device-id "$DEVICE")
fi
ACTIVE_CAT="$CAT"

mkdir -p "$ROOT/data/catalog"
exec > >(tee -a "$LOG") 2>&1

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

run() {
  local name="$1"; shift
  log "start $name"
  local started
  started=$(date +%s)
  "$BIN" "${COMMON[@]}" --catalog "$ACTIVE_CAT" "$@"
  log "elapsed $(( $(date +%s) - started )) s for $name"
  log "done  $name"
}

if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN; build ommatidia-data --release" >&2
  exit 1
fi
if [[ ! -f "$CAT" ]]; then
  echo "missing $CAT; run scripts/fetch-catalog.py" >&2
  exit 1
fi

if [[ -n "$DEVICE" ]]; then
  log "catalog capture on device $DEVICE"
else
  log "catalog capture on Blade's default adapter"
fi

# ABO objects in the procedural room. Seed is shared by each reference/input pair
# so --reference-from can verify G-buffer alignment.
run abo-train-reference \
  --out "$ROOT/data/catalog-abo-train-reference.omd" \
  --samples 24 --input-frames 1 --canonical-frames 4096 --sequence-frames 1 \
  --canopy --ground-patches 8 --textures --gloss \
  --catalog-split train --catalog-kind object --seed 80000

run abo-train-input \
  --out "$ROOT/data/catalog-abo-train.omd" \
  --samples 24 --input-frames 1 --canonical-frames 1 --sequence-frames 16 \
  --projection-jitter --canopy --ground-patches 8 --textures --gloss \
  --catalog-split train --catalog-kind object --seed 80000 \
  --reference-from "$ROOT/data/catalog-abo-train-reference.omd"

run abo-holdout-reference \
  --out "$ROOT/data/catalog-abo-holdout-reference.omd" \
  --samples 8 --input-frames 1 --canonical-frames 4096 --sequence-frames 1 \
  --canopy --ground-patches 8 --textures --gloss \
  --catalog-split holdout --catalog-kind object --seed 81000

run abo-holdout-input \
  --out "$ROOT/data/catalog-abo-holdout.omd" \
  --samples 8 --input-frames 1 --canonical-frames 1 --sequence-frames 16 \
  --projection-jitter --canopy --ground-patches 8 --textures --gloss \
  --catalog-split holdout --catalog-kind object --seed 81000 \
  --reference-from "$ROOT/data/catalog-abo-holdout-reference.omd"

# HSSD interiors. Entries with `scene` and `object_root` compose their Habitat
# furniture. Do not let older stage-only catalog entries into this corpus.
FURNISHED_CAT="$ROOT/data/catalog/furnished.json"
jq '{entries: [.entries[] | select(.kind == "interior" and .scene != null)]}' \
  "$CAT" > "$FURNISHED_CAT"
for split in train holdout; do
  if [[ "$(jq --arg split "$split" '[.entries[] | select(.split == $split)] | length' "$FURNISHED_CAT")" == 0 ]]; then
    echo "no furnished HSSD $split entry in $CAT; run scripts/fetch-hssd-scenes.sh" >&2
    exit 1
  fi
done
ACTIVE_CAT="$FURNISHED_CAT"

run hssd-train-reference \
  --out "$ROOT/data/catalog-hssd-train-reference.omd" \
  --samples 8 --input-frames 1 --canonical-frames 4096 --sequence-frames 1 \
  --catalog-split train --catalog-kind interior --seed 82000

run hssd-train-input \
  --out "$ROOT/data/catalog-hssd-train.omd" \
  --samples 8 --input-frames 1 --canonical-frames 1 --sequence-frames 16 \
  --projection-jitter \
  --catalog-split train --catalog-kind interior --seed 82000 \
  --reference-from "$ROOT/data/catalog-hssd-train-reference.omd"

run hssd-holdout-reference \
  --out "$ROOT/data/catalog-hssd-holdout-reference.omd" \
  --samples 4 --input-frames 1 --canonical-frames 4096 --sequence-frames 1 \
  --catalog-split holdout --catalog-kind interior --seed 83000

run hssd-holdout-input \
  --out "$ROOT/data/catalog-hssd-holdout.omd" \
  --samples 4 --input-frames 1 --canonical-frames 1 --sequence-frames 16 \
  --projection-jitter \
  --catalog-split holdout --catalog-kind interior --seed 83000 \
  --reference-from "$ROOT/data/catalog-hssd-holdout-reference.omd"

log "all catalog captures finished"
ls -lh "$ROOT"/data/catalog-*.omd
