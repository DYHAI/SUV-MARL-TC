#!/bin/bash
# Train newly available depths as soon as each complete 500-event archive is
# ready, then run the shared seven-depth evaluation after every model exists.
set -eu
cd "$(dirname "$0")"

RL=/home/ding/main/miniconda3/envs/RL/bin/python
NAS=/mnt/nas/multi_auv_typhoon_tracking/union_current_depths
LOGDIR=logs_uwflow_7depth
SPLIT="$PWD/data/typhoon_split_500.json"
SEGS="$PWD/data/segments_500.csv"
STEPS=2000000
START_KM=200
NENV=4
MAXJOBS=${MAXJOBS:-3}
PENDING_DEPTHS=${PENDING_DEPTHS:-"10 20 40 60"}
mkdir -p "$LOGDIR"

nrun() {
  pgrep -af "[t]rain_uwflow_sb3.py" 2>/dev/null |
    awk '$2 ~ /\/python$/ && $3 == "train_uwflow_sb3.py" {print $5}' |
    sort -u | wc -l
}

active_model() {
  pgrep -af "[t]rain_uwflow_sb3.py" 2>/dev/null |
    awk -v name="$1" \
      '$2 ~ /\/python$/ && $3 == "train_uwflow_sb3.py" && $5 == name {found=1}
       END {exit !found}'
}

expose_depth() {
  local depth=$1
  while IFS= read -r src; do
    tag=$(basename "$(dirname "$src")")
    dst="data/union_region/$tag/hycom_uv_${depth}m.nc"
    if [ ! -e "$dst" ] && [ ! -L "$dst" ]; then
      mkdir -p "$(dirname "$dst")"
      ln -s "$src" "$dst"
    fi
  done < <(find "$NAS" -mindepth 2 -maxdepth 2 -type f \
    -name "hycom_uv_${depth}m.nc" | sort)
}

cache_count() {
  find "/home/ding/main/data/uwflow_cache/d${1}_g0.25_c3" \
    -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l
}

for depth in $PENDING_DEPTHS; do
  while true; do
    if "$RL" audit_hycom_currents.py --depths "$depth" \
      > "$LOGDIR/audit_d${depth}.log" 2>&1; then
      break
    fi
    echo "$(date '+%F %T') waiting for complete depth=${depth} m" \
      >> "$LOGDIR/pending_depths_run.log"
    sleep 300
  done

  expose_depth "$depth"
  cache="/home/ding/main/data/uwflow_cache/d${depth}_g0.25_c3"
  while [ "$(cache_count "$depth")" -lt 500 ]; do
    echo "$(date '+%F %T') building/refreshing cache depth=${depth} m; current=$(cache_count "$depth")/500" \
      >> "$LOGDIR/pending_depths_run.log"
    for shard in 0 1 2 3; do
      CACHE_SHARD_INDEX="$shard" CACHE_SHARD_COUNT=4 \
        "$RL" build_uwflow_cache.py "$depth" 0.25 3 \
        >> "$LOGDIR/cache_d${depth}.log" 2>&1 &
    done
    wait
    if [ "$(cache_count "$depth")" -lt 500 ]; then
      echo "$(date '+%F %T') cache still incomplete at $(cache_count "$depth")/500; waiting for repair/download" \
        >> "$LOGDIR/pending_depths_run.log"
      sleep 300
      expose_depth "$depth"
    fi
  done

  for vmax in 2.0 1.5; do
    for seed in $(seq 0 9); do
      name="ppo_uwflow_500_d${depth}_v${vmax}_rC_20h_sb3_s${seed}"
      [ -f "ppo_underwater_fleet/${name}.zip" ] && continue
      active_model "$name" && continue
      while [ "$(nrun)" -ge "$MAXJOBS" ]; do
        sleep 30
      done
      echo "$(date '+%F %T') training depth=${depth} m: $name" \
        >> "$LOGDIR/pending_depths_run.log"
      setsid bash -c "cd '$PWD' || exit 1; \
        SPLIT_JSON='$SPLIT' SEGMENTS_CSV='$SEGS' PRELOAD=1 TRAIN_SEED='$seed' \
        OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 '$RL' train_uwflow_sb3.py \
        '$STEPS' '$name' '$vmax' C '$depth' 20 '$seed' 0.25 '$START_KM' '$NENV' \
        >> '$LOGDIR/train_${name}.log' 2>&1" < /dev/null &
      sleep 20
    done
  done
done

while [ "$(nrun)" -gt 0 ]; do
  sleep 60
done

while true; do
  count=$(find ppo_underwater_fleet -maxdepth 1 \
    -name 'ppo_uwflow_500_d*_v*_rC_20h_sb3_s*.zip' | wc -l)
  echo "$(date '+%F %T') corrected models=${count}/140" \
    >> "$LOGDIR/pending_depths_run.log"
  [ "$count" -ge 140 ] && break
  sleep 300
done

echo "$(date '+%F %T') evaluating all seven depths" \
  >> "$LOGDIR/pending_depths_run.log"
"$RL" run_uwflow_eval.py 10 "0 10 20 30 40 50 60" "1.5 2.0" "20" \
  >> "$LOGDIR/eval.log" 2>&1
eval_count=$(find eval_out -maxdepth 1 -type f \
  -name 'eval_info_info_d*_500.csv' | wc -l)
[ "$eval_count" -eq 14 ] || {
  echo "expected 14 final evaluation files, found $eval_count" \
    >> "$LOGDIR/pending_depths_run.log"
  exit 1
}
"$RL" summarize_uwflow_500.py >> "$LOGDIR/summarize.log" 2>&1
"$RL" analyze_uwflow_final.py >> "$LOGDIR/analyze.log" 2>&1
"$RL" Latex_Yihui_MARL/tools/aggregate_underwater_method_ablation.py \
  >> "$LOGDIR/aggregate.log" 2>&1
"$RL" verify_uwflow_completion.py >> "$LOGDIR/verify.log" 2>&1

echo "$(date '+%F %T') seven-depth analysis complete" \
  >> "$LOGDIR/pending_depths_run.log"
touch "$LOGDIR/ALL.done"
