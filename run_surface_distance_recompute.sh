#!/bin/bash
# Re-evaluate all final 12/20 m/s surface models while recording both the
# time-averaged and final TC-center distances.
set -u
cd "$(dirname "$0")"

RL=/home/ding/main/miniconda3/envs/RL/bin/python
LOGDIR=logs_surface_final_distance
mkdir -p "$LOGDIR"
MAXJOBS=${MAXJOBS:-6}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
FLEETS=${FLEETS:-"4 8 12 16"}
RADII=${RADII:-"100 200 400 600 800"}

running() {
  pgrep -f "^$RL eval_deploy_xfer.py" 2>/dev/null | wc -l
}

for V in 12 20; do
  for N in $FLEETS; do
    for S in $SEEDS; do
      if [ "$V" -eq 12 ]; then
        MODEL="ppo_fleet/ppo_fleet_seg${N}_ring_s005_d800_t10min_n500_dscan_mthr100_ab_windonly_noid_vmax12_s${S}.zip"
      elif [ "$S" -gt 0 ]; then
        MODEL="ppo_fleet/ppo_fleet_seg${N}_ring_s005_d800_t10min_n500_dscan_mthr100_ab_windonly_noid_s${S}.zip"
      else
        MODEL="ppo_fleet/ppo_fleet_seg${N}_ring_s005_d800_t10min_n500_dscan_mthr100_ab_windonly_noid.zip"
      fi
      [ -f "$MODEL" ] || { echo "missing $MODEL" >> "$LOGDIR/run.log"; continue; }
      for D in $RADII; do
        TAG="v${V}_n${N}_s${S}_d${D}"
        LOG="$LOGDIR/${TAG}.log"
        [ -f "$LOGDIR/${TAG}.done" ] && continue
        while [ "$(running)" -ge "$MAXJOBS" ]; do
          sleep 20
        done
        echo "$(date '+%F %T') run $TAG" >> "$LOGDIR/run.log"
        setsid bash -c "cd '$PWD' || exit 1; \
          FLEET_VMAX='$V' XFER_R_TRAIN=800 DEPLOY_FIXED_KM='$D' \
          DEPLOY_OCEAN_ONLY=1 DEPLOY_MIN_COAST_KM=100 SAFETY_MARGIN_KM=100 \
          STORAGE16=0 SST_MODE=none USE_MSLP=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
          '$RL' eval_deploy_xfer.py '$MODEL' '$N' '$D' 1 \
          > '$LOG' 2>&1; touch '$LOGDIR/${TAG}.done'" < /dev/null &
        sleep 4
      done
    done
  done
done

while [ "$(running)" -gt 0 ]; do
  sleep 30
done

"$RL" Latex_Yihui_MARL/tools/make_surface_distance_figure.py \
  >> "$LOGDIR/figure.log" 2>&1
touch "$LOGDIR/ALL.done"
echo "$(date '+%F %T') complete" >> "$LOGDIR/run.log"
