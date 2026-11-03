#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RL=${PYTHON_BIN:-python}
MODEL_DIR="$ROOT/runs/formation/models"
LOGDIR="$ROOT/runs/formation/logs"
mkdir -p "$MODEL_DIR" "$LOGDIR"

export SPLIT_JSON="$PWD/data/typhoon_split_500.json"
export SEGMENTS_CSV="$PWD/data/segments_500.csv"
export DEPLOY_OCEAN_ONLY=1
export DEPLOY_MIN_COAST_KM=100
export SAFETY_MARGIN_KM=100
export OBS_AGENT_ID=0
export SST_MODE=none
export USE_MSLP=0
export STORAGE16=0
export NENV=${NENV:-1}
export FLEET_MODEL_DIR="$MODEL_DIR"

for speed in ${SPEEDS:-12 20}; do
  for formation in ${FORMATIONS:-line grid}; do
    name="surface_${formation}_n8_v${speed}_s0"
    model="$MODEL_DIR/${name}.zip"
    [ -f "$model" ] && continue
    echo "start $name"
    FLEET_VMAX="$speed" FORMATION="$formation" SEED=0 \
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      "$RL" -m code.surface.train_fleet_segment_actfilter \
      2000000 "${name}.zip" 0.8 8 800 \
      >"$LOGDIR/train_${name}.log" 2>&1
  done
done
echo "ALL FORMATION TRAINING DONE"
