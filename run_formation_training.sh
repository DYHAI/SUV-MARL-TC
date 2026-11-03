#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
RL=/home/ding/main/miniconda3/envs/RL/bin/python
mkdir -p ppo_formation_ablation logs_formation_ablation

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
export FLEET_MODEL_DIR="$PWD/ppo_formation_ablation"

for speed in ${SPEEDS:-12 20}; do
  for formation in ${FORMATIONS:-line grid}; do
    name="surface_${formation}_n8_v${speed}_s0"
    model="ppo_formation_ablation/${name}.zip"
    [ -f "$model" ] && continue
    echo "start $name"
    FLEET_VMAX="$speed" FORMATION="$formation" SEED=0 \
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      "$RL" train_fleet_segment_actfilter.py \
      2000000 "${name}.zip" 0.8 8 800 \
      >"logs_formation_ablation/train_${name}.log" 2>&1
  done
done
echo "ALL FORMATION TRAINING DONE"
