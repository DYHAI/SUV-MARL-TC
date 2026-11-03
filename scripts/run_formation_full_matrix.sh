#!/usr/bin/env bash
# Complete the surface formation study:
#   3 formations x 2 speed limits x 5 fixed deployment radii
# Every condition uses a policy trained with deployment offsets sampled inside
# an 800 km radius and is evaluated on the same 768 held-out TC segments.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RL=${PYTHON_BIN:-python}
OUTDIR="$ROOT/runs/formation/eval"
LOGDIR="$ROOT/runs/formation/logs/full_matrix"
SURFACE_MODEL_DIR=${FLEET_MODEL_DIR:-"$ROOT/runs/surface/models"}
FORMATION_MODEL_DIR=${FORMATION_MODEL_DIR:-"$ROOT/runs/formation/models"}
MAXJOBS=${MAXJOBS:-3}
mkdir -p "$OUTDIR" "$LOGDIR"

model_path() {
  local formation=$1
  local speed=$2
  case "${formation}:${speed}" in
    ring:12)
      echo "$SURFACE_MODEL_DIR/ppo_fleet_seg8_ring_s005_d800_t10min_n500_dscan_mthr100_ab_windonly_noid_vmax12_s0.zip"
      ;;
    ring:20)
      echo "$SURFACE_MODEL_DIR/ppo_fleet_seg8_ring_s005_d800_t10min_n500_dscan_mthr100_ab_windonly_noid.zip"
      ;;
    line:12|grid:12)
      echo "$FORMATION_MODEL_DIR/surface_${formation}_n8_v12_s0.zip"
      ;;
    line:20|grid:20)
      echo "$FORMATION_MODEL_DIR/surface_${formation}_n8_v20_s0.zip"
      ;;
    *)
      return 1
      ;;
  esac
}

wait_for_model() {
  local model=$1
  while [ ! -s "$model" ]; do
    echo "$(date '+%F %T') waiting for $model"
    sleep 60
  done
}

launch_eval() {
  local model=$1
  local formation=$2
  local speed=$3
  local radius=$4
  local out="${OUTDIR}/formation_${formation}_v${speed}_d${radius}.csv"
  local log="${LOGDIR}/eval_${formation}_v${speed}_d${radius}.log"

  if [ -s "$out" ] && [ "$(wc -l < "$out")" -ge 769 ]; then
    echo "$(date '+%F %T') skip complete $out"
    return 0
  fi

  while [ "$(jobs -pr | wc -l)" -ge "$MAXJOBS" ]; do
    sleep 15
  done

  echo "$(date '+%F %T') start ${formation} ${speed} m/s ${radius} km"
  DEPLOY_OCEAN_ONLY=1 \
  DEPLOY_MIN_COAST_KM=100 \
  SAFETY_MARGIN_KM=100 \
  FLEET_VMAX="$speed" \
  OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 \
  "$RL" -m code.surface.eval_surface_robustness "$model" \
    --n-auv 8 \
    --vmax "$speed" \
    --radius "$radius" \
    --formation "$formation" \
    --conditions clean \
    --out "$out" \
    >"$log" 2>&1 &
}

for speed in 12 20; do
  for formation in ring line grid; do
    model=$(model_path "$formation" "$speed")
    wait_for_model "$model"
    for radius in 100 200 400 600 800; do
      launch_eval "$model" "$formation" "$speed" "$radius"
    done
  done
done

wait

for speed in 12 20; do
  for formation in ring line grid; do
    for radius in 100 200 400 600 800; do
      out="${OUTDIR}/formation_${formation}_v${speed}_d${radius}.csv"
      rows=$(wc -l < "$out" 2>/dev/null || echo 0)
      if [ "$rows" -lt 769 ]; then
        echo "$(date '+%F %T') incomplete: $out ($rows lines)" >&2
        exit 1
      fi
    done
  done
done

echo "$(date '+%F %T') formation evaluation matrix complete"
"$RL" -m code.formation.aggregate_formation_full_matrix
touch "$LOGDIR/ALL.done"
