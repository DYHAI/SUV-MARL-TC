#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/official_env.sh"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DYN_SURFACE_ENABLED=1
export DYN_UNDERWATER_ENABLED=0
export DYN_SURFACE_MODEL_SUBDIR=surface_main
export DYN_SURFACE_LOG_SUBDIR=surface_main
export DYN_COMPLETE_MARKER="$ROOT/runs/official/SURFACE_PRIMARY.done"

exec "${PYTHON_BIN:-python}" -m code.official_experiments.run_main_training
