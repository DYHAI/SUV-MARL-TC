#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
RL=/home/ding/main/miniconda3/envs/RL/bin/python
OUT=classical_eval
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"

FLEETS=(4 8 12 16)
VMAXES=(12 20)
PLANNERS=(potential astar)
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
        "$RL" eval_classical_surface.py "$planner" \
          --n-auv "$n_auv" --vmax "$vmax" \
          --radii 100 200 400 600 800 --seeds 1 --out "$out"
      ) >"$log" 2>&1 &
      echo "started $planner N=$n_auv Vmax=$vmax pid=$!"
    done
  done
done

wait
echo "ALL SURFACE CLASSICAL JOBS DONE"
