#!/usr/bin/env bash
# Verify capture-order independence without touching the user's normal asset cache.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}"
root="${CACHE_RESULTS:-target/cache-quality}"
if test -e "$root"; then echo "Choose a fresh CACHE_RESULTS directory: $root" >&2; exit 2; fi
mkdir -p "$root"
capture() {
  OMMATIDIA_ASSET_CACHE="$root/cache-$3" cargo +1.92.0 run -p ommatidia-data --locked -- \
    --out "$root/$1.omd" --samples 1 --seed "$2" --lr 16x16 --scale 2 \
    --canonical-frames 2 --input-frames 1 --textures --gloss --canopy \
    --ground-patches 4 --hr-gbuffer --split-radiance 2>&1 | tee "$root/$1.log"
}
capture primer 20260907 warm
capture warm 7 warm
capture cold 7 cold
python3 - "$root" "${CACHE_EXPECT:-stable}" <<'PY'
import hashlib, json, struct, sys
from pathlib import Path
root=Path(sys.argv[1]); a=(root/'warm.omd').read_bytes(); b=(root/'cold.omd').read_bytes()
assert a[:64]==b[:64] and len(a)==len(b), 'capture layouts differ'
_,_,scale,w,h,lmask,hmask,count,*_=struct.unpack('<8s14I', a[:64])
order=[0,8,9,10,1,2,3,4,5,6,7]; channels=[3,1,3,3,3,1,2,2,3,3,3]
record=(len(a)-64)//count; changed={}
offset=0
for label,mask,pixels in [('lr',lmask,w*h),('hr',hmask,w*h*scale*scale)]:
    for plane in order:
        if not (mask>>plane&1): continue
        size=2*pixels*channels[plane]
        changed[f'{label}.{plane}']=sum(
            x!=y for r in range(count)
            for x,y in zip(a[64+r*record+offset:64+r*record+offset+size],
                           b[64+r*record+offset:64+r*record+offset+size]))
        offset+=size
assert offset==record
result={'expected':sys.argv[2], 'byte_identical':a==b, 'changed_bytes_by_plane':changed,
        'warm_sha256':hashlib.sha256(a).hexdigest(), 'cold_sha256':hashlib.sha256(b).hexdigest()}
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert changed['lr.1']==changed['hr.1']==0, 'geometry changed in a texture-cache test'
if sys.argv[2]=='stable': assert a==b, 'a preceding capture changed the output'
elif sys.argv[2]=='stale': assert changed['lr.3']>0, 'negative control did not reproduce stale albedo'
else: raise ValueError('CACHE_EXPECT must be stable or stale')
PY
