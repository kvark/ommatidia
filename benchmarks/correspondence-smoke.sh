#!/usr/bin/env bash
# Bounded training/reload and negative admission tests; needs the surface smoke capture.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
export OMMATIDIA_REQUIRE_GPU=1
out=target/correspondence-smoke
[[ ! -e "$out" ]] || { echo "refusing to overwrite $out" >&2; exit 2; }
mkdir -p "$out"
field=(cargo +1.92.0 run -p ommatidia-train --bin field --locked --)
transport=(cargo +1.92.0 run -p ommatidia-train --bin transport --locked --)
"${field[@]}" --data target/surface-smoke/fit.omd --out "$out/stereo" \
  --image 16 --views 3 --channels 4 --hidden 16 --rays 8 --samples 16 \
  --probes 4 --steps 8 --surface-weight 0.05 --view-fusion stereo-rgb
"${field[@]}" --data target/surface-smoke/fit.omd --out "$out/stereo-reload" \
  --eval-checkpoint "$out/stereo/model.safetensors"
for offset in 0 8; do
  cargo +1.92.0 run -p ommatidia-data --locked -- \
    --out "$out/noise$offset.omd" --samples 1 --sequence-frames 2 --lr 8x8 \
    --canonical-frames 2 --reference-sample-offset 32 --input-sample-offset "$offset" \
    --hr-gbuffer --split-radiance --seed 8675309
done
"${transport[@]}" --data "$out/noise0.omd" --eval-data "$out/noise8.omd" \
  --out "$out/rgb" --construction-noise --rgb-loss --fixed-exposure-loss \
  --steps 2 --channels 4 --physical-weight 1 --compressed-weight 0 \
  --low-frequency-weight 0 --confidence-weight 0 --temporal-weight 0
mkdir -p "$out/rgb-reload"
cp "$out/rgb/model.safetensors" "$out/rgb/model.transport.ron" "$out/rgb-reload/"
"${transport[@]}" --eval-only --eval-data "$out/noise8.omd" \
  --out "$out/rgb-reload" --construction-noise
cmp "$out/rgb/quality.json" "$out/rgb-reload/quality.json"
if "${transport[@]}" --data "$out/noise0.omd" --eval-data "$out/noise0.omd" \
    --out "$out/rejected-noise" --construction-noise --steps 0 > "$out/rejected-noise.log" 2>&1; then
  echo 'duplicate noise was accepted' >&2; exit 1
fi
grep -q 'overlapping path streams' "$out/rejected-noise.log"
if "${transport[@]}" --data "$out/noise0.omd" --eval-data "$out/noise8.omd" \
    --out "$out/rejected-scene" --steps 0 > "$out/rejected-scene.log" 2>&1; then
  echo 'ordinary scene admission was relaxed' >&2; exit 1
fi
grep -q 'scene seeds overlap' "$out/rejected-scene.log"
python3 - <<'PY'
from pathlib import Path
import struct
root=Path('target/correspondence-smoke')
data=bytearray((root/'noise8.omd').read_bytes())
assert data[:8]==b'OMMATIDA'
version,scale,w,h,lr,hr=struct.unpack_from('<6I',data,8)
assert version==2 and hr&1
channels=[3,1,3,3,3,1,2,2,3,3,3]
# HR colour is the first HR plane. Change one finite reference component only.
offset=64+2*w*h*sum(n for i,n in enumerate(channels) if lr>>i&1)
value=struct.unpack_from('<e',data,offset)[0]
struct.pack_into('<e',data,offset,value+1.0)
(root/'altered.omd').write_bytes(data)
(root/'altered.transport.json').write_bytes((root/'noise8.transport.json').read_bytes())
PY
if "${transport[@]}" --data "$out/noise0.omd" --eval-data "$out/altered.omd" \
    --out "$out/rejected-truth" --construction-noise --steps 0 > "$out/rejected-truth.log" 2>&1; then
  echo 'changed reference was accepted as another noise realization' >&2; exit 1
fi
grep -q 'changed reference or observed geometry' "$out/rejected-truth.log"
