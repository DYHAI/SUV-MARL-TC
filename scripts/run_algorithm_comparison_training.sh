#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/official_env.sh"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DYN_COMPARISON_SCOPE=all
export DYN_SKIP_FORMATION=1
export DYN_ALGO_SUBDIR=algorithms

exec "${PYTHON_BIN:-python}" -m code.official_experiments.run_comparison_training
