#!/bin/bash
# Build the seven-depth underwater comparison after the 0--60 m HYCOM
# current-only archive has completed.
set -u
cd "$(dirname "$0")"

RL=/home/ding/main/miniconda3/envs/RL/bin/python
NAS=/mnt/nas/multi_auv_typhoon_tracking/union_current_depths
COMPLETE=logs_hycom_depth_10_20_40_60/ALL_CURRENT_DEPTHS_COMPLETE
LOGDIR=logs_uwflow_7depth
mkdir -p "$LOGDIR"

echo "$(date '+%F %T') waiting for complete 0--60 m current archive" >> "$LOGDIR/run.log"
while [ ! -f "$COMPLETE" ]; do
  sleep 300
done

# Expose NAS-only files through the data/union_region paths consumed by the
# environment loader. Existing local files take precedence.
while IFS= read -r src; do
  tag=$(basename "$(dirname "$src")")
  depth=$(basename "$src" | sed -E 's/hycom_uv_([0-9]+)m\.nc/\1/')
  dst="data/union_region/$tag/hycom_uv_${depth}m.nc"
  if [ ! -e "$dst" ] && [ ! -L "$dst" ]; then
    mkdir -p "$(dirname "$dst")"
    ln -s "$src" "$dst"
  fi
done < <(find "$NAS" -mindepth 2 -maxdepth 2 -type f -name 'hycom_uv_*m.nc' | sort)

# The 0/30/50 m caches already exist. Build caches for the four new depths.
for D in 10 20 40 60; do
  if [ ! -d "/home/ding/main/data/uwflow_cache/d${D}_g0.25_c3" ]; then
    echo "$(date '+%F %T') building cache depth=$D" >> "$LOGDIR/run.log"
    "$RL" build_uwflow_cache.py "$D" 0.25 3 >> "$LOGDIR/cache_d${D}.log" 2>&1
  fi
done

# Train all seven depths with the corrected unit-disk action projection and
# causal frame-history update. The pre-fix 0/30/50 m rC/sb3 weights were moved
# to ppo_underwater_fleet/pre_unitdisk_static_history_backup for recovery.
SPLIT="$PWD/data/typhoon_split_500.json"
SEGS="$PWD/data/segments_500.csv"
STEPS=2000000
START_KM=200
NENV=4
MAXJOBS=${MAXJOBS:-3}

nrun() {
  pgrep -af "[t]rain_uwflow_sb3.py" 2>/dev/null |
    grep -oE 'ppo_uwflow_500_[A-Za-z0-9._]+' |
    sed 's/\.log$//' | sort -u | wc -l
}

for D in 0 10 20 30 40 50 60; do
  for V in 2.0 1.5; do
    for S in $(seq 0 9); do
      NAME="ppo_uwflow_500_d${D}_v${V}_rC_20h_sb3_s${S}"
      [ -f "ppo_underwater_fleet/${NAME}.zip" ] && continue
      while [ "$(nrun)" -ge "$MAXJOBS" ]; do
        sleep 30
      done
      echo "$(date '+%F %T') training $NAME" >> "$LOGDIR/run.log"
      setsid bash -c "cd '$PWD' || exit 1; \
        SPLIT_JSON='$SPLIT' SEGMENTS_CSV='$SEGS' PRELOAD=1 TRAIN_SEED='$S' \
        OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 '$RL' train_uwflow_sb3.py \
        '$STEPS' '$NAME' '$V' C '$D' 20 '$S' 0.25 '$START_KM' '$NENV' \
        >> '$LOGDIR/train_${NAME}.log' 2>&1" < /dev/null &
      sleep 20
    done
  done
done

while [ "$(nrun)" -gt 0 ]; do
  sleep 60
done

echo "$(date '+%F %T') evaluating all seven depths" >> "$LOGDIR/run.log"
"$RL" run_uwflow_eval.py 10 "0 10 20 30 40 50 60" "1.5 2.0" "20" \
  >> "$LOGDIR/eval.log" 2>&1
"$RL" summarize_uwflow_500.py >> "$LOGDIR/summarize.log" 2>&1
"$RL" Latex_Yihui_MARL/tools/aggregate_underwater_method_ablation.py \
  >> "$LOGDIR/aggregate.log" 2>&1

echo "$(date '+%F %T') seven-depth analysis complete" >> "$LOGDIR/run.log"
touch "$LOGDIR/ALL.done"
