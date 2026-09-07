#!/usr/bin/env bash
# Paired training-loss normalization experiment; quality only.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export OMMATIDIA_REQUIRE_GPU=1
root="${NORMALIZATION_RESULTS:-target/normalization-quality}"
updates="${TRANSPORT_UPDATES:-512}"
train_scenes="${TRANSPORT_TRAIN_SCENES:-8}"
hold_scenes="${TRANSPORT_HOLD_SCENES:-4}"
hold_seed="${TRANSPORT_HOLD_SEED:-50000}"
mkdir -p "$root"
cargo=(cargo +1.92.0)
common=(--lr 32x32 --input-frames 1 --canopy --textures --gloss
        --ground-patches 4 --random-camera-motion 0.03 --object-motion 0.03
        --light-motion 0.05 --projection-jitter --hr-gbuffer --split-radiance)
{
  echo "backend=Vulkan LavaPipe; quality only; no checkpoint selection"
  echo "train=$train_scenes scenes x 8 frames seed 7; validation=$hold_scenes x 16 seed $hold_seed"
  echo "32x32 -> 64x64; 1 spp input; train/reference canonical frames=64/128"
  echo "updates=$updates; unroll=2; channels=8; optimization seed=7; exposure=1"
  echo "Only loss_scale changes: 1/(0.1+target) versus fixed exposure. Primary compressed loss remains."
  git rev-parse HEAD
  git -C ../meganeura rev-parse HEAD
  git -C ../blade rev-parse HEAD
  "${cargo[@]}" --version
  printf 'capture options: '; printf '%q ' "${common[@]}"; echo
} > "$root/recipe.txt"
"${cargo[@]}" run -p ommatidia-data --locked -- \
  --out "$root/train.omd" --samples "$train_scenes" --sequence-frames 8 --seed 7 \
  --canonical-frames 64 "${common[@]}" 2>&1 | tee "$root/capture-train.log"
"${cargo[@]}" run -p ommatidia-data --locked -- \
  --out "$root/validation.omd" --samples "$hold_scenes" --sequence-frames 16 --seed "$hold_seed" \
  --canonical-frames 128 "${common[@]}" 2>&1 | tee "$root/capture-validation.log"
for arm in relative exposure; do
  extra=()
  if [[ "$arm" == exposure ]]; then extra+=(--fixed-exposure-loss); fi
  "${cargo[@]}" run -p ommatidia-train --bin transport --locked -- \
    --data "$root/train.omd" --eval-data "$root/validation.omd" --out "$root/$arm" \
    --steps "$updates" --unroll 2 --channels 8 --seed 7 "${extra[@]}" \
    2>&1 | tee "$root/$arm.log"
done
"${cargo[@]}" run -p ommatidia-data --locked -- \
  --out "$root/reference-repeat.omd" --samples 1 --sequence-frames 16 --seed "$hold_seed" \
  --canonical-frames 128 --reference-sample-offset 128 "${common[@]}" \
  2>&1 | tee "$root/capture-reference-repeat.log"
"${cargo[@]}" run -p ommatidia-train --bin reference-noise --locked -- \
  --a "$root/validation.omd" --b "$root/reference-repeat.omd" \
  2>&1 | tee "$root/reference-noise.txt"
sha256sum "$root"/*.omd "$root"/*/model.safetensors > "$root/sha256.txt"
python3 - "$root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
r, e = [json.loads((root / arm / 'quality.json').read_text()) for arm in ('relative', 'exposure')]
assert r['baseline'] == e['baseline'], 'mismatched deterministic controls'
print(json.dumps({'baseline': r['baseline'], 'relative': r['learned'], 'exposure': e['learned']}, indent=2))
PY
