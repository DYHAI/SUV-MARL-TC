#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/official_env.sh"
RL=${PYTHON_BIN:-python}
OUT="$ROOT/runs/baselines"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"

FLEETS=(4 8 12 16)
VMAXES=(12 20)
read -ra PLANNERS <<< "${PLANNERS:-potential astar coverage greedy}"
MAX_JOBS=${MAX_JOBS:-6}

for planner in "${PLANNERS[@]}"; do
  for n_auv in "${FLEETS[@]}"; do
    for vmax in "${VMAXES[@]}"; do
      out="$OUT/surface_${planner}_test_n${n_auv}_v${vmax}_s1.csv"
      log="$LOGS/surface_${planner}_n${n_auv}_v${vmax}.log"
      if [ -s "$out" ]; then
        echo "skip $out"
        continue
      fi
      while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 5
      done
      (
        "$RL" -m code.baselines.eval_classical_surface "$planner" \
          --n-auv "$n_auv" --vmax "$vmax" \
          --radii 100 200 400 600 800 --seeds 1 --out "$out"
      ) >"$log" 2>&1 &
      echo "started $planner N=$n_auv Vmax=$vmax pid=$!"
    done
  done
done

wait
echo "ALL SURFACE CLASSICAL JOBS DONE"
