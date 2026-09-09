#!/usr/bin/env bash
# Parameterization controls only. Quality, not software-device timing.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export OMMATIDIA_REQUIRE_GPU=1
fit=(cargo +1.92.0 run -p ommatidia-train --bin fit --locked --)
transport=(cargo +1.92.0 run -p ommatidia-train --bin transport --locked --)
task="${1:?fit, causal or smoke}"
if [[ "$task" == smoke ]]; then
  out=target/masked-smoke
  [[ ! -e "$out" ]] || { echo "refusing to overwrite $out" >&2; exit 2; }
  "${fit[@]}" --task selector-rgb --mixture masked-softmax \
    --data target/catalog-smoke.omd --out "$out/frozen" --steps 8 --frame 0
  data=target/correspondence-smoke
  "${transport[@]}" --data "$data/noise0.omd" --eval-data "$data/noise8.omd" \
    --out "$out/train" --mixture masked-softmax --construction-noise \
    --steps 4 --channels 4 --physical-weight 1 --compressed-weight 0 \
    --low-frequency-weight 0 --confidence-weight 0 --temporal-weight 0
  mkdir -p "$out/reload"
  cp "$out/train/model.safetensors" "$out/train/model.transport.ron" "$out/reload/"
  "${transport[@]}" --eval-only --eval-data "$data/noise8.omd" \
    --out "$out/reload" --construction-noise
  cmp "$out/train/quality.json" "$out/reload/quality.json"
  cmp "$out/train/model.safetensors" "$out/reload/model.safetensors"
  if "${transport[@]}" --out "$out/rejected" --eval-only --mixture masked-softmax \
      > "$out/admission.log" 2>&1; then
    echo 'evaluation accepted an explicit mixture override' >&2; exit 1
  fi
  grep -q 'evaluation reads mixture from the checkpoint sidecar' "$out/admission.log"
  exit 0
fi
mode="${2:?softplus or masked-softmax}"
[[ "$mode" == softplus || "$mode" == masked-softmax ]] || { echo 'invalid mixture' >&2; exit 2; }
seed="${MASKED_SEED:-7}"
frame="${MASKED_FRAME:-0}"
steps="${MASKED_UPDATES:-512}"
data="${MASKED_DATA:-target/geometry/noise-7/data}"
root="${MASKED_RESULTS:-target/masked-selector}"
case "$task" in
  fit)
    parent="$root/masked-fit-$mode-$frame"; out="$parent/$mode-$frame"
    args=("${fit[@]}" --task selector-rgb --mixture "$mode" --data "$data/noise0.omd" --out "$out" --frame "$frame" --seed "$seed" --steps "$steps")
    ;;
  causal)
    parent="$root/masked-causal-$mode-$seed"; out="$parent/$mode-$seed"
    args=("${transport[@]}" --out "$out" --mixture "$mode" --steps "$steps" --seed "$seed" --channels 8 --unroll 2 --construction-noise --fixed-exposure-loss --compressed-weight 0 --physical-weight 1 --low-frequency-weight 0 --confidence-weight 0 --temporal-weight 0)
    for i in 0 1 2; do args+=(--data "$data/noise$i.omd"); done
    for i in 3 4 5; do args+=(--eval-data "$data/noise$i.omd"); done
    ;;
  *) echo 'task must be fit, causal or smoke' >&2; exit 2 ;;
esac
[[ ! -e "$parent" ]] || { echo "refusing to overwrite $parent" >&2; exit 2; }
mkdir -p "$parent"
{ git rev-parse HEAD; git -C ../meganeura rev-parse HEAD; printf '%q ' "${args[@]}"; echo; sha256sum "$data"/*.omd "$data"/*.json; } > "$parent/recipe.txt"
"${args[@]}" 2>&1 | tee "$parent/run.log"
if [[ "$task" == causal ]]; then
  if "${transport[@]}" --out "$out" --eval-only --mixture "$mode" > "$out/admission.log" 2>&1; then
    echo 'evaluation accepted an explicit mixture override' >&2; exit 1
  fi
  grep -q 'evaluation reads mixture from the checkpoint sidecar' "$out/admission.log"
fi
find "$out" \( -name '*.safetensors' -o -name quality.json \) -print0 | sort -z | xargs -0 sha256sum > "$parent/results.sha256"
