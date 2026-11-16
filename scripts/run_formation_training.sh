#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/official_env.sh"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DYN_COMPARISON_SCOPE=surface
export DYN_SKIP_OFFPOLICY=1
export DYN_SURFACE_TAG=rigid_formation

exec "${PYTHON_BIN:-python}" -m code.official_experiments.run_comparison_training
