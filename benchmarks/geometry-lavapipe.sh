#!/usr/bin/env bash
# Fixed-budget construction diagnostics. No timing or production-quality claim.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export OMMATIDIA_REQUIRE_GPU=1
arm="${1:?field-control, field-consistent or noise}"
seed="${GEOMETRY_SEED:-7}"
out="${GEOMETRY_RESULTS:-target/geometry/$arm-$seed}"
if [[ -e "$out" ]]; then echo "refusing to overwrite $out" >&2; exit 2; fi
mkdir -p "$out"
field=(cargo +1.92.0 run -p ommatidia-train --bin field --locked --)
capture=(cargo +1.92.0 run -p ommatidia-data --locked --)
risk=(cargo +1.92.0 run -p ommatidia-train --bin noise-risk --locked --)
[[ -n "${GEOMETRY_FIELD_BIN:-}" ]] && field=("$GEOMETRY_FIELD_BIN")
[[ -n "${GEOMETRY_CAPTURE_BIN:-}" ]] && capture=("$GEOMETRY_CAPTURE_BIN")
[[ -n "${GEOMETRY_RISK_BIN:-}" ]] && risk=("$GEOMETRY_RISK_BIN")
case "$arm" in
 field-control|field-consistent)
   data="${GEOMETRY_FIELD_DATA:-target/quality-control-data/field.omd}"
   weight=0; [[ "$arm" == field-consistent ]] && weight=0.1
   args=("${field[@]}" --data "$data" --out "$out" --image 64 --views 3 --channels 8 --hidden 32 --rays 64 --samples 64 --probes 16 --steps "${GEOMETRY_UPDATES:-1024}" --seed "$seed" --surface-weight 0.05 --emitter-fraction 0.25 --diagnostics --stratified --view-fusion visible-rgb --consistency-rays 8 --consistency-weight "$weight")
   { git rev-parse HEAD; printf '%q ' "${args[@]}"; echo; sha256sum "$data" "${data%.omd}.scene.json"; } > "$out/recipe.txt"
   "${args[@]}" 2>&1 | tee "$out/run.log"
   sha256sum "$out/model.safetensors" "$out/quality.json" > "$out/SHA256SUMS"
   ;;
 noise)
   checkpoint="${GEOMETRY_CHECKPOINT:?set GEOMETRY_CHECKPOINT to the existing transport model.safetensors}"
   args=(); mkdir -p "$out/data"
   # Six disjoint LR RNG ranges [1,8], [9,16], ... [41,48]. HR starts after128.
   common=(--samples 2 --sequence-frames 4 --lr 16x16 --input-frames 1 --canopy --textures --gloss --ground-patches 4 --random-camera-motion 0.03 --object-motion 0.03 --light-motion 0.05 --projection-jitter --hr-gbuffer --split-radiance --canonical-frames 64 --reference-sample-offset 128 --seed 20260908)
   git rev-parse HEAD > "$out/recipe.txt"
   for ((i=0;i<6;i++)); do
     data="$out/data/noise$i.omd"
     command=("${capture[@]}" --out "$data" --input-sample-offset "$((8*i))" "${common[@]}")
     printf '%q ' "${command[@]}" >> "$out/recipe.txt"; echo >> "$out/recipe.txt"
     "${command[@]}" 2>&1 | tee "$out/data/capture$i.log"
     args+=(--data "$data")
   done
   sha256sum "$out"/data/*.omd "$out"/data/*.json > "$out/captures.sha256"
   sha256sum "$checkpoint" > "$out/checkpoint-before.sha256"
   command=("${risk[@]}" "${args[@]}" --fit-realizations 3 --checkpoint "$checkpoint" --out "$out/scores")
   printf '%q ' "${command[@]}" >> "$out/recipe.txt"; echo >> "$out/recipe.txt"
   "${command[@]}" 2>&1 | tee "$out/run.log"
   sha256sum -c "$out/checkpoint-before.sha256"
   # Same file twice must fail stream independence rather than yield a plausible score.
   if "${risk[@]}" --data "$out/data/noise0.omd" "${args[@]}" --fit-realizations 3 --checkpoint "$checkpoint" --out "$out/invalid" > "$out/rejection.log" 2>&1; then
     echo 'duplicate path stream was accepted' >&2; exit 1
   fi
   grep -q 'overlapping path streams' "$out/rejection.log"
   ;;
 *) echo "unknown experiment: $arm" >&2; exit 2 ;;
esac
