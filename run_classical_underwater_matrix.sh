#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
RL=/home/ding/main/miniconda3/envs/RL/bin/python
OUT=classical_eval
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"

DEPTHS=(0 10 20 30 40 50 60)
VMAXES=(1.5 2.0)
PLANNERS=(potential astar)
MAX_JOBS=${MAX_JOBS:-4}

for planner in "${PLANNERS[@]}"; do
  for depth in "${DEPTHS[@]}"; do
    for vmax in "${VMAXES[@]}"; do
      while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 5
      done
      out="$OUT/underwater_${planner}_test_d${depth}_v${vmax}_s10.csv"
      log="$LOGS/underwater_${planner}_d${depth}_v${vmax}.log"
      if [ -s "$out" ]; then
        echo "skip $out"
        continue
      fi
      (
        "$RL" eval_classical_underwater.py "$planner" \
          --depth "$depth" --vmax "$vmax" --split-key test \
          --seeds 10 --batch-storms 10 --out "$out"
      ) >"$log" 2>&1 &
      echo "started $planner depth=$depth Vmax=$vmax pid=$!"
    done
  done
done

wait
echo "ALL UNDERWATER CLASSICAL JOBS DONE"
