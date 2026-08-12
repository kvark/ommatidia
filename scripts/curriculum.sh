#!/bin/bash
# Long unattended training curriculum for ommatidia.
#
# Runs sequentially, not in parallel: two trainings on one GPU contend for the
# same cores and the same VRAM, so serialising costs nothing in throughput and
# keeps the footprint to one model at a time — which matters, because another
# training process shares this device.
set -u

ROOT=/x/Code/ommatidia
OUT=${OUT:-$ROOT/runs}
BIN=$ROOT/target/release/ommatidia-train
DATA=${DATA:-$ROOT/data/raw-restir-2400.omd}
DEVICE_ID=${DEVICE_ID:-0x744c}

# Deployment baseline: 649k parameters, one forward pass, and the same held-out
# quality as the old 6.5M-parameter model once both are trained to convergence.
BASE=${BASE:-24}
BATCH=${BATCH:-8}
LEVELS=${LEVELS:-3}
BLOCKS=${BLOCKS:-1}
TILE=64

# Refuse to start if the device is already close to full — better to report
# than to fall over an allocation failure hours in.
guard_vram() {
  local d=/sys/class/drm/card0/device
  local free=$(( ( $(cat $d/mem_info_vram_total) - $(cat $d/mem_info_vram_used) ) / 1024 / 1024 ))
  if [ "$free" -lt 4096 ]; then
    echo "!! only ${free}MiB VRAM free, waiting for room"
    while [ "$free" -lt 4096 ]; do
      sleep 60
      free=$(( ( $(cat $d/mem_info_vram_total) - $(cat $d/mem_info_vram_used) ) / 1024 / 1024 ))
    done
  fi
  echo "   ${free}MiB VRAM free"
}

run() {
  local name=$1 objective=$2 steps=$3 evalevery=$4; shift 4
  echo
  echo "=================================================================="
  echo "== $name  ($objective, $steps steps)  started $(date -u +%H:%M:%S)"
  echo "=================================================================="
  guard_vram
  "$BIN" --data "$DATA" \
    --device-id "$DEVICE_ID" \
    --objective "$objective" --steps "$steps" \
    --batch $BATCH --tile $TILE --base-channels $BASE --levels $LEVELS --blocks $BLOCKS \
    --lr 3e-4 --lr-final 1e-5 \
    --log-every $(( steps / 60 + 1 )) --eval-every "$evalevery" \
    --checkpoint-every "$evalevery" --eval-crops 64 \
    --out "$OUT/$name" --eval-out "$OUT/eval-$name" \
    "$@"
  echo "== $name finished $(date -u +%H:%M:%S)"
}

echo "curriculum started $(date -u)"
"$@"
echo "curriculum finished $(date -u)"
