#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/official_env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SPLIT_JSON="$ROOT/data/typhoon_split_400.json"
export SEGMENTS_CSV="$ROOT/data/segments_primary_400.csv"

OUT="${CLASSICAL_OUT_DIR:-$ROOT/runs/classical/underwater}"
mkdir -p "$OUT"

for planner in potential astar coverage greedy; do
  for depth in 0 10 20 30 40 50 60; do
    for speed in 1.5 2.0; do
      "${PYTHON_BIN:-python}" -m code.baselines.eval_classical_underwater \
        "$planner" --depth "$depth" --vmax "$speed" --split-key test --seeds 10 \
        --out "$OUT/${planner}_d${depth}_v${speed}.csv"
    done
  done
done
