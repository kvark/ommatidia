#!/usr/bin/env bash
# Quality only: software Vulkan timings are not product performance numbers.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export OMMATIDIA_REQUIRE_GPU=1
root="${TRANSPORT_RESULTS:-target/transport-quality}"
updates="${TRANSPORT_UPDATES:-128}"
train_scenes="${TRANSPORT_TRAIN_SCENES:-4}"
hold_scenes="${TRANSPORT_HOLD_SCENES:-2}"
mkdir -p "$root"
cargo=(cargo +1.92.0)
common=(--lr 32x32 --input-frames 1 --canopy --textures --gloss
        --ground-patches 4 --random-camera-motion 0.03 --object-motion 0.03
        --light-motion 0.05 --projection-jitter --hr-gbuffer --split-radiance)
{
  echo "backend=Vulkan LavaPipe; quality only"
  echo "train=$train_scenes scenes x 8 frames; holdout=$hold_scenes scenes x 16 frames; 32x32 -> 64x64"
  echo "train canonical frames=64; holdout canonical frames=128; input=1 spp"
  echo "updates=$updates; unroll=2; channels=8; seed=7"
  git rev-parse HEAD
  git -C ../blade rev-parse HEAD
  git -C ../meganeura rev-parse HEAD
  "${cargo[@]}" --version
  printf 'capture options: '; printf '%q ' "${common[@]}"; echo
} > "$root/recipe.txt"
"${cargo[@]}" run -p ommatidia-data --locked -- \
  --out "$root/train.omd" --samples "$train_scenes" --sequence-frames 8 --seed 7 \
  --canonical-frames 64 "${common[@]}" 2>&1 | tee "$root/capture-train.log"
"${cargo[@]}" run -p ommatidia-data --locked -- \
  --out "$root/holdout.omd" --samples "$hold_scenes" --sequence-frames 16 --seed 10000 \
  --canonical-frames 128 "${common[@]}" 2>&1 | tee "$root/capture-holdout.log"
"${cargo[@]}" run -p ommatidia-train --bin transport --locked -- \
  --data "$root/train.omd" --eval-data "$root/holdout.omd" \
  --steps "$updates" --unroll 2 --channels 8 --seed 7 --out "$root/model" \
  2>&1 | tee "$root/training.log"
# Repeat the first held-out sequence with disjoint reference path indices.
"${cargo[@]}" run -p ommatidia-data --locked -- \
  --out "$root/reference-repeat.omd" --samples 1 --sequence-frames 16 --seed 10000 \
  --canonical-frames 128 --reference-sample-offset 128 "${common[@]}" \
  2>&1 | tee "$root/capture-reference-repeat.log"
"${cargo[@]}" run -p ommatidia-train --bin reference-noise --locked -- \
  --a "$root/holdout.omd" --b "$root/reference-repeat.omd" \
  2>&1 | tee "$root/reference-noise.txt"
sha256sum "$root"/*.omd "$root/model/model.safetensors" > "$root/sha256.txt"
cat "$root/model/quality.json"
