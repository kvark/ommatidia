#!/usr/bin/env bash
# Development controls, not an unseen-scene or production-speed claim.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export OMMATIDIA_REQUIRE_GPU=1
arm="${1:?late-rgb, visible-rgb, visible-unsupervised, selector-control or selector-projected}"
seed="${SUPPORT_SEED:-7}"
data="${SUPPORT_DATA:-target/quality-control-data}"
out="${SUPPORT_RESULTS:-target/support/$arm-$seed}"
if [[ -e "$out/quality.json" || -e "$out/model.safetensors" ]]; then
  echo "refusing to overwrite an existing experiment: $out" >&2; exit 2
fi
mkdir -p "$out"
field=(cargo +1.92.0 run -p ommatidia-train --bin field --locked --)
transport=(cargo +1.92.0 run -p ommatidia-train --bin transport --locked --)
[[ -n "${SUPPORT_FIELD_BIN:-}" ]] && field=("$SUPPORT_FIELD_BIN")
[[ -n "${SUPPORT_TRANSPORT_BIN:-}" ]] && transport=("$SUPPORT_TRANSPORT_BIN")
case "$arm" in
 late-rgb|visible-rgb|visible-unsupervised)
   mode="$arm"; weight=0.05
   if [[ "$arm" == visible-unsupervised ]]; then mode=visible-rgb; weight=0; fi
   args=("${field[@]}" --data "$data/field.omd" --out "$out" --image 64 --views 3 --channels 8 --hidden 32 --rays 32 --samples 64 --probes 16 --steps "${SUPPORT_UPDATES:-512}" --seed "$seed" --surface-weight 0.05 --emitter-fraction 0.25 --diagnostics --stratified --view-fusion "$mode" --visibility-weight "$weight")
   ;;
 selector-control|selector-projected)
   weight=0; [[ "$arm" == selector-projected ]] && weight=0.1
   args=("${transport[@]}" --data "$data/train.omd" --eval-data "$data/validation.omd" --out "$out" --channels 8 --unroll 2 --steps "${SUPPORT_UPDATES:-512}" --seed "$seed" --fixed-exposure-loss --confidence-weight 0 --projected-weight "$weight" --candidate-oracle)
   ;;
 *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac
{ git rev-parse HEAD; printf '%q ' "${args[@]}"; echo; sha256sum "$data"/*.omd "$data"/*.json; } > "$out/recipe.txt"
"${args[@]}" 2>&1 | tee "$out/run.log"
sha256sum "$out/model.safetensors" "$out/quality.json" > "$out/results.sha256"
