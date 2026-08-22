#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
    echo "usage: benchmarks/run-curated.sh OIDN_DENOISE [VULKAN_DEVICE_ID] [OUT_DIR] [CHECKPOINT]" >&2
    exit 2
fi

oidn_denoise=$1
vulkan_device=${2:-0x744c}
output_dir=${3:-runs/comparison-suite}
checkpoint=${4:-runs/rich-kernel-b16-demod025}
path_data=data/rich-4spp-validation-128.omd
restir_data=data/rich-restir-svgf-curated-16.omd

if [ ! -f "$path_data" ]; then
    echo "$path_data is missing; see benchmarks/README.md for the Hugging Face download" >&2
    exit 1
fi
if [ ! -x "$oidn_denoise" ]; then
    echo "$oidn_denoise is not an executable oidnDenoise" >&2
    exit 1
fi

if [ ! -f "$restir_data" ]; then
    cargo run --release -q -p ommatidia-data -- \
        --device-id "$vulkan_device" \
        --out "$restir_data" \
        --samples 16 --lr 128x128 --scale 2 \
        --input-frames 4 --canonical-frames 1024 --canonical-bounces 8 \
        --hr-gbuffer --canopy --textures --gloss --seed 10000 \
        --svgf-input --reference-from "$path_data"
fi

cargo run --release -q -p ommatidia-train --bin compare -- \
    --device-id "$vulkan_device" \
    --data "$path_data" \
    --restir-svgf-data "$restir_data" \
    --checkpoint "$checkpoint" \
    --oidn "$oidn_denoise" --oidn-device 0 \
    --out "$output_dir" \
    --case canopy-shadow:0 \
    --case glossy-contact:4 \
    --case local-light:5 \
    --case textured-gloss:7 \
    --case indoor-detail:9 \
    --case hard-shadow:15

awk -F, '
    NR == 1 { next }
    !($2 in seen) { seen[$2] = ++method_count; order[method_count] = $2 }
    {
        count[$2]++
        for (column = 3; column <= 10; column++) sum[$2, column] += $column
    }
    END {
        print "method,psnr_db,ssim,relative_mse,detail_ratio,energy_ratio,low_frequency_psnr_db"
        for (ordinal = 1; ordinal <= method_count; ordinal++) {
            method = order[ordinal]
            printf "%s,%.3f,%.6f,%.6f,%.6f,%.6f,%.3f\n", method, \
                sum[method, 4] / count[method], sum[method, 5] / count[method], \
                sum[method, 6] / count[method], sum[method, 7] / count[method], \
                sum[method, 8] / count[method], sum[method, 10] / count[method]
        }
    }
' "$output_dir/metrics.csv" > "$output_dir/summary.csv"
