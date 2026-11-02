#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
RL=/home/ding/main/miniconda3/envs/RL/bin/python
OUT=classical_eval
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"

DEPTHS=(0 10 20 30 40 50 60)
VMAXES=(1.5 2.0)
MAX_JOBS=${MAX_JOBS:-1}
BATCH_STORMS=${BATCH_STORMS:-5}

for depth in "${DEPTHS[@]}"; do
  for vmax in "${VMAXES[@]}"; do
    out="$OUT/underwater_astar_test_d${depth}_v${vmax}_s10.csv"
    log="$LOGS/underwater_astar_d${depth}_v${vmax}.log"
    if [ -s "$out" ]; then
      echo "skip $out"
      continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
      sleep 5
    done
    (
      "$RL" eval_classical_underwater.py astar \
        --depth "$depth" --vmax "$vmax" --split-key test \
        --seeds 10 --batch-storms "$BATCH_STORMS" --out "$out"
    ) >"$log" 2>&1 &
    echo "started astar depth=$depth Vmax=$vmax pid=$!"
  done
done

wait
echo "ALL A* UNDERWATER JOBS DONE"
