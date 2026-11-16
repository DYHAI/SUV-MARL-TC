#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/official_env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"${PYTHON_BIN:-python}" -m code.analysis.aggregate_primary_median --only surface
"${PYTHON_BIN:-python}" -m code.analysis.make_surface_distance_figure
