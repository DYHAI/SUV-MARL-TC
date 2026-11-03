#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
RL=/home/ding/main/miniconda3/envs/RL/bin/python
mkdir -p ppo_algorithm_comparison logs_algorithm_comparison

for task in ${TASKS:-surface underwater}; do
  for algorithm in ${ALGORITHMS:-sac td3}; do
    for seed in ${SEEDS:-0 1}; do
      if [ "$task" = surface ]; then
        name="surface_${algorithm}_n8_v12_s${seed}"
      else
        name="underwater_${algorithm}_n4_v1.5_d30_s${seed}"
      fi
      model="ppo_algorithm_comparison/${name}.zip"
      [ -f "$model" ] && continue
      echo "start $name"
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$RL" \
        train_offpolicy_representative.py "$task" "$algorithm" \
        --steps 2000000 --workers "${WORKERS:-4}" --seed "$seed" \
        >"logs_algorithm_comparison/train_${name}.log" 2>&1
    done
  done
done
echo "ALL ALGORITHM COMPARISON TRAINING DONE"
