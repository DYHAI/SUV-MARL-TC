#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/official_env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SPLIT_JSON="$ROOT/data/typhoon_split_400.json"
export SEGMENTS_CSV="$ROOT/data/segments_primary_400.csv"

OUT="${CLASSICAL_OUT_DIR:-$ROOT/runs/classical/surface}"
mkdir -p "$OUT"

for planner in potential astar coverage greedy; do
  for n in 4 8 12 16; do
    for speed in 12 20; do
      "${PYTHON_BIN:-python}" -m code.baselines.eval_classical_surface \
        "$planner" --n-auv "$n" --vmax "$speed" --split-key test --seeds 10 \
        --out "$OUT/${planner}_n${n}_v${speed}.csv"
    done
  done
done
