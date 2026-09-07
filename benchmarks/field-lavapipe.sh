#!/usr/bin/env bash
# Small complete reconstruction experiment. No speed claims on software Vulkan.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export OMMATIDIA_REQUIRE_GPU=1
root="${FIELD_RESULTS:-target/field-quality}"
mkdir -p "$root"
cargo=(cargo +1.92.0)
{
  echo 'Vulkan LavaPipe; quality only; RGB/poses only'
  echo '2 fitting scenes, 1 unseen scene, 6 cameras; 16x16 RGB; 128 reference spp'
  echo '64 updates; 2 context views; channels=4 hidden=16; rays=8 steps=16 probes=8'
  git rev-parse HEAD
  git -C ../meganeura rev-parse HEAD
  "${cargo[@]}" tree -p ommatidia --depth 1
} > "$root/recipe.txt"
"${cargo[@]}" run -p ommatidia-data --locked -- \
  --out "$root/train.omd" --samples 2 --field-views 6 --lr 16x16 --scale 1 \
  --canonical-frames 32 --seed 7 --lighting-seed 19 \
  2>&1 | tee "$root/capture-train.log"
"${cargo[@]}" run -p ommatidia-data --locked -- \
  --out "$root/held.omd" --samples 1 --field-views 6 --lr 16x16 --scale 1 \
  --canonical-frames 32 --seed 10000 --lighting-seed 23 \
  2>&1 | tee "$root/capture-held.log"
"${cargo[@]}" run -p ommatidia-train --bin field --locked -- \
  --data "$root/train.omd" --eval-data "$root/held.omd" --out "$root/model" \
  --image 16 --views 2 --channels 4 --hidden 16 --rays 8 --samples 16 --probes 8 \
  --steps 64 --seed 7 2>&1 | tee "$root/train.log"
sha256sum "$root"/*.omd "$root"/*.scene.json "$root/model/model.safetensors" > "$root/sha256.txt"
cat "$root/model/quality.json"
