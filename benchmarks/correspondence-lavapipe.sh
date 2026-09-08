#!/usr/bin/env bash
# Fixed-budget development experiments, not unseen-scene or speed claims.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export OMMATIDIA_REQUIRE_GPU=1
arm="${1:?field-visible, field-stereo, selector-lobes or selector-rgb}"
seed="${CORRESPONDENCE_SEED:-7}"
out="${CORRESPONDENCE_RESULTS:-target/correspondence/$arm-$seed}"
if [[ -e "$out" ]]; then echo "refusing to overwrite $out" >&2; exit 2; fi
mkdir -p "$out"
field=(cargo +1.92.0 run -p ommatidia-train --bin field --locked --)
transport=(cargo +1.92.0 run -p ommatidia-train --bin transport --locked --)
[[ -n "${CORRESPONDENCE_FIELD_BIN:-}" ]] && field=("$CORRESPONDENCE_FIELD_BIN")
[[ -n "${CORRESPONDENCE_TRANSPORT_BIN:-}" ]] && transport=("$CORRESPONDENCE_TRANSPORT_BIN")
case "$arm" in
 field-visible|field-stereo)
   data="${CORRESPONDENCE_FIELD_DATA:-target/quality-control-data/field.omd}"
   mode=visible-rgb; [[ "$arm" == field-stereo ]] && mode=stereo-rgb
   args=("${field[@]}" --data "$data" --out "$out" --image 64 --views 3 --channels 8 --hidden 32 --rays 64 --samples 64 --probes 16 --steps "${CORRESPONDENCE_UPDATES:-1024}" --seed "$seed" --surface-weight 0.05 --visibility-weight 0.05 --emitter-fraction 0.25 --diagnostics --stratified --view-fusion "$mode")
   sha256sum "$data" "${data%.omd}.scene.json" > "$out/captures.sha256"
   ;;
 selector-lobes|selector-rgb)
   data="${CORRESPONDENCE_NOISE_DATA:-target/noise-data}"
   args=("${transport[@]}" --out "$out" --steps "${CORRESPONDENCE_UPDATES:-512}" --seed "$seed" --channels 8 --unroll 2 --construction-noise --fixed-exposure-loss --compressed-weight 0 --physical-weight 1 --low-frequency-weight 0 --confidence-weight 0 --temporal-weight 0)
   for i in 0 1 2; do args+=(--data "$data/noise$i.omd"); done
   for i in 3 4 5; do args+=(--eval-data "$data/noise$i.omd"); done
   [[ "$arm" == selector-rgb ]] && args+=(--rgb-loss)
   sha256sum "$data"/*.omd "$data"/*.json > "$out/captures.sha256"
   ;;
 *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac
{ git rev-parse HEAD; printf '%q ' "${args[@]}"; echo; } > "$out/recipe.txt"
"${args[@]}" 2>&1 | tee "$out/run.log"
sha256sum "$out/model.safetensors" "$out/quality.json" > "$out/results.sha256"
