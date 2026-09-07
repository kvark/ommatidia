#!/usr/bin/env bash
# Matched fixed-budget ablation. Holdout scores never select a checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export OMMATIDIA_REQUIRE_GPU=1
root="${INCIDENT_RESULTS:-target/incident-quality}"
mkdir -p "$root"
cargo=(cargo +1.92.0)
{
  echo 'LavaPipe; quality only; RGB/poses/acquisition bounds in both arms'
  echo 'Two fitting scenes in two lighting conditions; two untouched scenes'
  echo '6 cameras, 16x16, 128 image reference spp; 24 incident probes at 32 paths each'
  echo 'Two optimization seeds, 128 updates; only incident loss weight changes (0 / 0.1)'
  git rev-parse HEAD
  git -C ../meganeura rev-parse HEAD
} > "$root/recipe.txt"
for lighting in 19 31; do
  "${cargo[@]}" run -p ommatidia-data --locked -- \
    --out "$root/train-$lighting.omd" --samples 2 --field-views 6 --lr 16x16 --scale 1 \
    --canonical-frames 32 --seed 7 --lighting-seed "$lighting" \
    --incident-probes 24 --incident-batches 8 2>&1 | tee "$root/capture-$lighting.log"
done
"${cargo[@]}" run -p ommatidia-data --locked -- \
  --out "$root/held.omd" --samples 2 --field-views 6 --lr 16x16 --scale 1 \
  --canonical-frames 32 --seed 10000 --lighting-seed 23 \
  --incident-probes 24 --incident-batches 8 2>&1 | tee "$root/capture-held.log"
for seed in 7 11; do
  for weight in 0 0.1; do
    "${cargo[@]}" run -p ommatidia-train --bin field --locked -- \
      --data "$root/train-19.omd" --data "$root/train-31.omd" --eval-data "$root/held.omd" \
      --out "$root/seed-$seed-weight-$weight" --image 16 --views 2 --channels 4 --hidden 16 \
      --rays 8 --samples 16 --probes 8 --incident-rays 4 --incident-weight "$weight" \
      --steps 128 --seed "$seed" 2>&1 | tee "$root/train-$seed-$weight.log"
  done
done
sha256sum "$root"/*.omd "$root"/*.scene.json "$root"/seed-*/model.safetensors > "$root/sha256.txt"
python3 - "$root" <<'PY'
import json, math, pathlib, sys
root=pathlib.Path(sys.argv[1]); rows=[]
for seed in [7,11]:
    controls=json.loads((root/f'seed-{seed}-weight-0/quality.json').read_text())
    supervised=json.loads((root/f'seed-{seed}-weight-0.1/quality.json').read_text())
    for a,b in zip(controls['scores'],supervised['scores'],strict=True):
        assert a['scene_seed']==b['scene_seed'] and a['untrained']==b['untrained']
        row={'training_seed':seed,'scene_seed':a['scene_seed']}
        for name,report in [('control',a),('supervised',b)]:
            row[name]={'image_log1p_mse':report['learned']['log1p_mse'],
                       'image_psnr':report['learned']['compressed_psnr'],
                       'incident_log1p_mse':report['incident']['total']['log1p_mse']}
            assert all(math.isfinite(v) for v in row[name].values())
        rows.append(row)
result={'status':'measured ablation, not a promotion gate','rows':rows}
(root/'ablation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PY
