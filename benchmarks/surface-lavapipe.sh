#!/usr/bin/env bash
# Construction diagnostic, not an untouched quality audit. No speed claims.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export OMMATIDIA_REQUIRE_GPU=1
root="${SURFACE_RESULTS:-target/surface-quality}"
image="${FIELD_IMAGE:-64}"
updates="${FIELD_UPDATES:-512}"
mkdir -p "$root"
cargo=(cargo +1.92.0)
{
  echo "Construction-only: one scene, eight cameras, three context views, image=$image, updates=$updates"
  echo 'Surface weight 0/0.05; identical graph, seed, camera pixels, midpoint samples, emission probes'
  echo '8 core channels, 32 hidden; 32 rays, 32 steps; centre RGB at 256 spp'
  git rev-parse HEAD; git -C ../meganeura rev-parse HEAD
} > "$root/recipe.txt"
"${cargo[@]}" run -p ommatidia-data --locked -- \
  --out "$root/fit.omd" --samples 1 --field-views 8 --surface-labels \
  --lr "${image}x${image}" --scale 1 --canonical-frames 64 \
  --seed 20260907 --lighting-seed 43 2>&1 | tee "$root/capture.log"
for weight in 0 0.05; do
  "${cargo[@]}" run -p ommatidia-train --bin field --locked -- \
    --data "$root/fit.omd" --out "$root/weight-$weight" --image "$image" \
    --views 3 --channels 8 --hidden 32 --rays 32 --samples 32 --probes 16 \
    --steps "$updates" --seed 7 --surface-weight "$weight" \
    --emitter-fraction 0.25 --diagnostics 2>&1 | tee "$root/train-$weight.log"
done
sha256sum "$root"/*.omd "$root"/*.scene.json "$root"/weight-*/model.safetensors > "$root/sha256.txt"
python3 - "$root" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); a,b=[json.loads((root/f'weight-{w}/quality.json').read_text()) for w in ['0','0.05']]
assert a['scores'][0]['untrained']==b['scores'][0]['untrained'], 'unmatched initialization'
for p in root.glob('weight-*/*-context.json'):
    obj=json.loads(p.read_text()); assert set(obj)=={'bounds','views'}
    assert all(set(v)=={'camera','rgb'} for v in obj['views'])
result={'role':'construction diagnostic, not promotion','control':a,'surface':b}
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
PY
