#!/usr/bin/env bash
# Construction diagnostics: no new-scene audit, no speed claims.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export OMMATIDIA_REQUIRE_GPU=1
arm="${1:?choose recover-wide, moments, late-rgb or candidate-oracle}"
data="${LATE_DATA:-target/quality-control-data}"
out="${LATE_RESULTS:-target/late-results/$arm}"
mkdir -p "$out"
field=(cargo +1.92.0 run -p ommatidia-train --bin field --locked --)
transport=(cargo +1.92.0 run -p ommatidia-train --bin transport --locked --)
[[ -n "${LATE_FIELD_BIN:-}" ]] && field=("$LATE_FIELD_BIN")
[[ -n "${LATE_TRANSPORT_BIN:-}" ]] && transport=("$LATE_TRANSPORT_BIN")
case "$arm" in
 recover-wide)
   checkpoint="${LATE_RECOVER_DIR:?supply the saved wider run directory}/model.safetensors"
   args=("${field[@]}" --data "$data/field.omd" --out "$out" --eval-checkpoint "$checkpoint" --rays 32 --samples 64 --probes 0 --diagnostics)
   sha256sum "$checkpoint" > "$out/original-checkpoint.sha256"
   ;;
 moments|late-rgb)
   args=("${field[@]}" --data "$data/field.omd" --out "$out" --image 64 --views 3 --channels 8 --hidden 32 --rays 32 --samples 64 --probes 16 --steps "${LATE_UPDATES:-512}" --seed 7 --surface-weight 0.05 --emitter-fraction 0.25 --diagnostics --stratified --view-fusion "$arm")
   ;;
 candidate-oracle)
   source="${LATE_TRANSPORT_DIR:?supply the saved corrected denoiser run directory}"
   cp "$source/model.safetensors" "$source/model.transport.ron" "$out/"
   args=("${transport[@]}" --out "$out" --eval-data "$data/validation.omd" --eval-only --candidate-oracle)
   sha256sum "$out/model.safetensors" > "$out/original-checkpoint.sha256"
   ;;
 *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac
{ git rev-parse HEAD; printf '%q ' "${args[@]}"; echo; sha256sum "$data"/*.omd "$data"/*.json; } > "$out/recipe.txt"
"${args[@]}" 2>&1 | tee "$out/run.log"
sha256sum "$out/quality.json" > "$out/results.sha256"
if [[ "$arm" == recover-wide ]]; then sha256sum -c "$out/original-checkpoint.sha256"; fi
