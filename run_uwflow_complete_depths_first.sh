#!/bin/bash
# Train corrected underwater policies for depths whose 500-TC archives are
# already complete. The seven-depth watcher remains responsible for later
# filling the remaining depths and running the final evaluation.
set -u
cd "$(dirname "$0")"

RL=${PYTHON_BIN:-python}
CACHE_ROOT=${UWFLOW_CACHE_DIR:-$HOME/.cache/auv_tc/uwflow}
LOGDIR=logs_uwflow_7depth
SPLIT="$PWD/data/typhoon_split_500.json"
SEGS="$PWD/data/segments_500.csv"
STEPS=2000000
START_KM=200
NENV=4
MAXJOBS=${MAXJOBS:-3}
DEPTHS=${DEPTHS:-"0 30 50"}
mkdir -p "$LOGDIR"

nrun() {
  pgrep -af "[t]rain_uwflow_sb3.py" 2>/dev/null |
    grep -oE 'ppo_uwflow_500_[A-Za-z0-9._]+' | sort -u | wc -l
}

for D in $DEPTHS; do
  for V in 2.0 1.5; do
    for S in $(seq 0 9); do
      NAME="ppo_uwflow_500_d${D}_v${V}_rC_20h_sb3_s${S}"
      [ -f "ppo_underwater_fleet/${NAME}.zip" ] && continue
      while [ "$(nrun)" -ge "$MAXJOBS" ]; do
        sleep 30
      done
      echo "$(date '+%F %T') training complete-archive depth $D: $NAME" \
        >> "$LOGDIR/complete_depths_run.log"
      setsid bash -c "cd '$PWD' || exit 1; \
        SPLIT_JSON='$SPLIT' SEGMENTS_CSV='$SEGS' PRELOAD=1 TRAIN_SEED='$S' \
        UWFLOW_CACHE_DIR='$CACHE_ROOT' OMP_NUM_THREADS=3 \
        MKL_NUM_THREADS=3 '$RL' train_uwflow_sb3.py \
        '$STEPS' '$NAME' '$V' C '$D' 20 '$S' 0.25 '$START_KM' '$NENV' \
        >> '$LOGDIR/train_${NAME}.log' 2>&1" < /dev/null &
      sleep 20
    done
  done
done

while [ "$(nrun)" -gt 0 ]; do
  sleep 60
done
echo "$(date '+%F %T') complete-archive depth training finished" \
  >> "$LOGDIR/complete_depths_run.log"
