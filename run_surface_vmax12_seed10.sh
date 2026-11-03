#!/bin/bash
# Blue Whale-class surface-speed ablation:
# FLEET_VMAX=12 m/s, 4/8/12/16 vehicles, 10 seeds, wind-only, d800 training.
# Existing Vmax=20 m/s weights remain untouched. New weights carry _vmax12.
set -u
cd "$(dirname "$0")"

RL=/home/ding/main/miniconda3/envs/RL/bin/python
export SPLIT_JSON="$PWD/data/typhoon_split_500.json"
export SEGMENTS_CSV="$PWD/data/segments_500.csv"
export FLEET_VMAX=12

MTHR=100
STEPS=2000000
NENV=${NENV:-4}
TAG=n500_dscan
MAXJOBS=${MAXJOBS:-2}
NEED_GB=${NEED_GB:-26}
SEED_LIST=${SEED_LIST:-"0 1 2 3 4 5 6 7 8 9"}
FLEETS=${FLEETS:-"4 8 12 16"}

mkdir -p logs_vmax12_seed10
LOG=logs_vmax12_seed10/run.log

nrun () {
  pgrep -af "[t]rain_fleet_segment_actfilter.py" 2>/dev/null \
    | grep -oE "ppo_fleet_seg(4|8|12|16)_[A-Za-z0-9._]+" \
    | sort -u | wc -l
}
mem_avail_gb () {
  awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo
}
inflight () {
  pgrep -af "[t]rain_fleet_segment_actfilter.py" | grep -q "$1"
}

echo "$(date '+%F %T') 与现有 20 m/s 任务并行启动 Vmax=12 m/s 训练（NENV=$NENV）" >> "$LOG"

for N in $FLEETS; do
  for S in $SEED_LIST; do
    OUT="ppo_fleet_seg${N}_ring_s005_d800_t10min_${TAG}_mthr${MTHR}_ab_windonly_noid_vmax12_s${S}.zip"
    [ -f "ppo_fleet/$OUT" ] && continue
    inflight "$OUT" && continue
    while :; do
      nr=$(nrun)
      av=$(mem_avail_gb)
      if [ "$nr" -lt "$MAXJOBS" ] && \
         awk -v a="$av" -v nd="$NEED_GB" 'BEGIN{exit !(a>nd)}'; then
        break
      fi
      sleep 60
    done
    L="logs_vmax12_seed10/train_${N}au_s${S}.log"
    echo "$(date '+%F %T') 投递 $OUT (在跑 $nr/$MAXJOBS, 可用 $(mem_avail_gb)G)" >> "$LOG"
    setsid bash -c "cd $PWD && SPLIT_JSON=$SPLIT_JSON SEGMENTS_CSV=$SEGMENTS_CSV \
      FLEET_VMAX=12 DEPLOY_OCEAN_ONLY=1 DEPLOY_MIN_COAST_KM=${MTHR} SAFETY_MARGIN_KM=${MTHR} \
      OBS_AGENT_ID=0 SST_MODE=none USE_MSLP=0 STORAGE16=0 SEED=${S} NENV=${NENV} \
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $RL train_fleet_segment_actfilter.py ${STEPS} \
      $OUT 0.8 ${N} 800 > $L 2>&1" < /dev/null &
    sleep 20
  done
done

echo "$(date '+%F %T') 40 组全部投递，等待训练收尾" >> "$LOG"
while [ "$(nrun)" -gt 0 ]; do sleep 120; done
echo "$(date '+%F %T') 训练完成，开始 200 格评测" >> "$LOG"

MAXJ=6
enrun () {
  pgrep -f "[e]val_deploy_xfer.py" | wc -l
}
: > seed_surface_vmax12_10.psv

for N in $FLEETS; do
  for S in $SEED_LIST; do
    M="ppo_fleet/ppo_fleet_seg${N}_ring_s005_d800_t10min_${TAG}_mthr${MTHR}_ab_windonly_noid_vmax12_s${S}.zip"
    if [ ! -f "$M" ]; then
      echo "$(date '+%F %T') 缺权重 $M" >> "$LOG"
      continue
    fi
    for D in 100 200 400 600 800; do
      T="n${N}_s${S}_d${D}"
      L="logs_vmax12_seed10/eval_${T}.log"
      [ -f "logs_vmax12_seed10/eval_${T}.done" ] && continue
      while :; do
        av=$(mem_avail_gb)
        nr=$(enrun)
        if [ "$nr" -lt "$MAXJ" ] && \
           awk -v a="$av" 'BEGIN{exit !(a>9.0)}'; then
          break
        fi
        sleep 20
      done
      setsid bash -c "cd $PWD && FLEET_VMAX=12 XFER_R_TRAIN=800 DEPLOY_FIXED_KM=$D \
        STORAGE16=0 SST_MODE=none USE_MSLP=0 DEPLOY_OCEAN_ONLY=1 \
        DEPLOY_MIN_COAST_KM=100 SAFETY_MARGIN_KM=100 \
        OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 $RL eval_deploy_xfer.py $M $N $D 1 \
        > $L 2>&1; touch logs_vmax12_seed10/eval_${T}.done" < /dev/null &
      sleep 4
    done
  done
done

while [ "$(enrun)" -gt 0 ]; do sleep 30; done

for N in $FLEETS; do
  for S in $SEED_LIST; do
    for D in 100 200 400 600 800; do
      L="logs_vmax12_seed10/eval_n${N}_s${S}_d${D}.log"
      [ -f "$L" ] || continue
      ROW=$(grep -m1 "^ROW|" "$L")
      [ -n "$ROW" ] || continue
      echo "$ROW" | awk -F'|' -v n="$N" -v s="$S" \
        '{printf "n%s_s%s_d%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n",
          n,s,$3,$4,s,$3,$5,$6,$7,$9,$11,$12,$14}' >> seed_surface_vmax12_10.psv
    done
  done
done

echo "$(date '+%F %T') Vmax=12 训练和 200 格评测全部完成" >> "$LOG"
touch logs_vmax12_seed10/ALL.done
