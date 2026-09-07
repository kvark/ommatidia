#!/usr/bin/env bash
# Construction controls, not a fresh audit or production-speed benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export OMMATIDIA_REQUIRE_GPU=1
arm="${1:?choose field-midpoint, field-stratified, field-fine, field-wide, transport-control, transport-no-compressed, transport-no-confidence or transport-no-temporal}"
data="${QUALITY_DATA:-target/quality-control-data}"
out="${QUALITY_RESULTS:-target/quality-controls/$arm}"
cargo=(cargo +1.92.0)
mkdir -p "$data" "$out"
case "$arm" in
 field-*)
    samples=32; channels=8; hidden=32; extra=()
    case "$arm" in
      field-midpoint) ;;
      field-stratified) extra=(--stratified) ;;
      field-fine) samples=64; extra=(--stratified) ;;
      field-wide) samples=64; channels=16; hidden=64; extra=(--stratified) ;;
      *) echo "unknown field arm: $arm" >&2; exit 2 ;;
    esac
    if ! test -f "$data/field.omd"; then
      "${cargo[@]}" run -p ommatidia-data --locked -- \
        --out "$data/field.omd" --samples 1 --field-views 8 --surface-labels \
        --lr 64x64 --scale 1 --canonical-frames 64 --seed 20260907 --lighting-seed 43 \
        2>&1 | tee "$data/capture-field.log"
    fi
    [[ "${QUALITY_CAPTURE_ONLY:-0}" == 1 ]] && exit 0
    args=(-p ommatidia-train --bin field --locked -- --data "$data/field.omd" \
      --out "$out" --image 64 --views 3 --channels "$channels" --hidden "$hidden" \
      --rays 32 --samples "$samples" --probes 16 --steps "${QUALITY_UPDATES:-2048}" \
      --seed "${QUALITY_SEED:-7}" --surface-weight 0.05 --emitter-fraction 0.25 --diagnostics "${extra[@]}")
    ;;
 transport-*)
    extra=()
    case "$arm" in
      transport-control) ;;
      transport-no-compressed) extra=(--compressed-weight 0) ;;
      transport-no-confidence) extra=(--confidence-weight 0) ;;
      transport-no-temporal) extra=(--temporal-weight 0) ;;
      *) echo "unknown transport arm: $arm" >&2; exit 2 ;;
    esac
    common=(--lr 32x32 --input-frames 1 --canopy --textures --gloss --ground-patches 4 \
      --random-camera-motion 0.03 --object-motion 0.03 --light-motion 0.05 \
      --projection-jitter --hr-gbuffer --split-radiance)
    if ! test -f "$data/train.omd"; then
      "${cargo[@]}" run -p ommatidia-data --locked -- --out "$data/train.omd" \
        --samples 8 --sequence-frames 8 --seed 7 --canonical-frames 64 "${common[@]}" \
        2>&1 | tee "$data/capture-train.log"
    fi
    if ! test -f "$data/validation.omd"; then
      "${cargo[@]}" run -p ommatidia-data --locked -- --out "$data/validation.omd" \
        --samples 4 --sequence-frames 16 --seed 50000 --canonical-frames 128 "${common[@]}" \
        2>&1 | tee "$data/capture-validation.log"
    fi
    [[ "${QUALITY_CAPTURE_ONLY:-0}" == 1 ]] && exit 0
    args=(-p ommatidia-train --bin transport --locked -- --data "$data/train.omd" \
      --eval-data "$data/validation.omd" --out "$out" --channels 8 --unroll 2 \
      --steps "${QUALITY_UPDATES:-512}" --seed "${QUALITY_SEED:-7}" --fixed-exposure-loss "${extra[@]}")
    ;;
 *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac
{
  echo "Construction-only control; Vulkan LavaPipe; no speed claim; arm=$arm"
  git rev-parse HEAD; git -C ../meganeura rev-parse HEAD
  "${cargo[@]}" --version
  printf '%q ' "${cargo[@]}" run "${args[@]}"; echo
  sha256sum "$data"/*.omd "$data"/*.json
} > "$out/recipe.txt"
"${cargo[@]}" run "${args[@]}" 2>&1 | tee "$out/training.log"
sha256sum "$out"/model.safetensors "$out"/quality.json > "$out/sha256.txt"
