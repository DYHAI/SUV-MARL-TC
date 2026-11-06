#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RL=${PYTHON_BIN:-python}
OUT="$ROOT/runs/baselines"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"

DEPTHS=(0 10 20 30 40 50 60)
VMAXES=(1.5 2.0)
read -ra PLANNERS <<< "${PLANNERS:-potential astar coverage greedy}"
MAX_JOBS=${MAX_JOBS:-4}
GRID_DEG=${GRID_DEG:-0.25}

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
        "$RL" -m code.baselines.eval_classical_underwater "$planner" \
          --depth "$depth" --vmax "$vmax" --split-key test \
          --grid-deg "$GRID_DEG" \
          --seeds 10 --batch-storms 10 --out "$out"
      ) >"$log" 2>&1 &
      echo "started $planner depth=$depth Vmax=$vmax pid=$!"
    done
  done
done

wait
echo "ALL UNDERWATER CLASSICAL JOBS DONE"
