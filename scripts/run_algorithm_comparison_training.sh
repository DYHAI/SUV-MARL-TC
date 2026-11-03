#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RL=${PYTHON_BIN:-python}
OUT="$ROOT/runs/algorithm_comparison"
LOGS="$ROOT/runs/algorithm_comparison/logs"
mkdir -p "$OUT" "$LOGS"

for task in ${TASKS:-surface underwater}; do
  for algorithm in ${ALGORITHMS:-sac td3}; do
    for seed in ${SEEDS:-0 1}; do
      if [ "$task" = surface ]; then
        name="surface_${algorithm}_n8_v12_s${seed}"
      else
        name="underwater_${algorithm}_n4_v1.5_d30_s${seed}"
      fi
      model="$OUT/${name}.zip"
      [ -f "$model" ] && continue
      echo "start $name"
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$RL" \
        -m code.algorithm_comparison.train_offpolicy_representative "$task" "$algorithm" \
        --steps 2000000 --workers "${WORKERS:-4}" --seed "$seed" \
        >"$LOGS/train_${name}.log" 2>&1
    done
  done
done
echo "ALL ALGORITHM COMPARISON TRAINING DONE"
