#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/official_env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"${PYTHON_BIN:-python}" -m code.formation.aggregate_formation_full_matrix
"${PYTHON_BIN:-python}" -m code.formation.make_formation_comparison_figure
