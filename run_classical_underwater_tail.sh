#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
RL=/home/ding/main/miniconda3/envs/RL/bin/python
OUT=classical_eval
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"

PLANNER=${PLANNER:?set PLANNER=potential or PLANNER=astar}
DEPTHS=(${DEPTHS:-30 40 50 60})
VMAXES=(1.5 2.0)
MAX_JOBS=${MAX_JOBS:-2}
BATCH_STORMS=${BATCH_STORMS:-10}

for depth in "${DEPTHS[@]}"; do
  for vmax in "${VMAXES[@]}"; do
    out="$OUT/underwater_${PLANNER}_test_d${depth}_v${vmax}_s10.csv"
    log="$LOGS/underwater_${PLANNER}_d${depth}_v${vmax}.log"
    if [ -s "$out" ]; then
      echo "skip $out"
      continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
      sleep 5
    done
    (
      "$RL" eval_classical_underwater.py "$PLANNER" \
        --depth "$depth" --vmax "$vmax" --split-key test \
        --seeds 10 --batch-storms "$BATCH_STORMS" --out "$out"
    ) >"$log" 2>&1 &
    echo "started $PLANNER depth=$depth Vmax=$vmax pid=$!"
  done
done

wait
echo "ALL $PLANNER TAIL JOBS DONE"
