#!/usr/bin/env bash
# Fixed-budget development experiments, not unseen-scene or speed claims.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export OMMATIDIA_REQUIRE_GPU=1
arm="${1:?capture, field-visible, field-stereo, selector-lobes or selector-rgb}"
if [[ "$arm" == capture ]]; then
  field_data="${CORRESPONDENCE_FIELD_DATA:-target/quality-control-data/field.omd}"
  noise_data="${CORRESPONDENCE_NOISE_DATA:-target/noise-data}"
  if [[ -e "$field_data" || -e "$noise_data" ]]; then
    echo 'capture destinations must be new; set explicit paths to avoid overwriting evidence' >&2; exit 2
  fi
  mkdir -p "$(dirname "$field_data")" "$noise_data"
  capture=(cargo +1.92.0 run -p ommatidia-data --locked --)
  args=("${capture[@]}" --out "$field_data" --samples 1 --field-views 8 --surface-labels --lr 64x64 --scale 1 --canonical-frames 64 --seed 20260907 --lighting-seed 43)
  printf '%q ' "${args[@]}" > "$noise_data/capture-recipe.txt"; echo >> "$noise_data/capture-recipe.txt"
  "${args[@]}" 2>&1 | tee "$noise_data/field-capture.log"
  common=(--samples 2 --sequence-frames 4 --lr 16x16 --input-frames 1 --canopy --textures --gloss --ground-patches 4 --random-camera-motion 0.03 --object-motion 0.03 --light-motion 0.05 --projection-jitter --hr-gbuffer --split-radiance --canonical-frames 64 --reference-sample-offset 128 --seed 20260908)
  for i in {0..5}; do
    args=("${capture[@]}" --out "$noise_data/noise$i.omd" --input-sample-offset "$((8*i))" "${common[@]}")
    printf '%q ' "${args[@]}" >> "$noise_data/capture-recipe.txt"; echo >> "$noise_data/capture-recipe.txt"
    "${args[@]}" 2>&1 | tee "$noise_data/capture$i.log"
  done
  sha256sum "$field_data" "${field_data%.omd}.scene.json" "$noise_data"/*.omd "$noise_data"/*.json > "$noise_data/capture.sha256"
  exit 0
fi
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
