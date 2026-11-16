# -*- coding: utf-8 -*-
"""Underwater 4-AUV max-flow matrix -- batch evaluation driver (2026-10-10).

For each (depth, Vmax, duration) combo run eval_uwflow_info.py ONCE; that single
pass evaluates idle(drift) / random / every trained model sharing that combo,
all paired on the same deployment seeds.  The number of deployment seeds is
independent of the number of training-seed models in constrained experiments.

Usage:
    python run_uwflow_eval.py <n_seed> [depths] [vmaxes] [durations]
    # default: 10 "0 30 50" "1.5 2.0" "10 20"
Output:
    eval_out/eval_info_info_d<depth>_start200_<dur>h_v<vmax>_<n_test>.csv
"""
import os, sys, glob, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(BASE)
RL = "python"
OUT = os.path.abspath(
    os.environ.get("UWFLOW_EVAL_DIR", os.path.join(BASE, "eval_out"))
)
os.makedirs(OUT, exist_ok=True)
LOGDIR = os.path.abspath(
    os.environ.get("UWFLOW_EVAL_LOG_DIR", os.path.join(BASE, "logs_uwflow_eval"))
)
os.makedirs(LOGDIR, exist_ok=True)

SPLIT = os.environ.get(
    "SPLIT_JSON", os.path.join(BASE, "data", "typhoon_split_400.json")
)
SEGS = os.environ.get(
    "SEGMENTS_CSV", os.path.join(BASE, "data", "segments_primary_400.csv")
)
SPLIT_KEY = os.environ.get("SPLIT_KEY", "test")
with open(SPLIT, encoding="utf-8") as stream:
    N_TEST = len(json.load(stream)[SPLIT_KEY])
MODELDIR = os.path.abspath(
    os.environ.get("UWFLOW_MODEL_DIR", os.path.join(BASE, "ppo_underwater_fleet"))
)

N_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 10
DEPTHS = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0 30 50").split()]
VMAXES = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1.5 2.0").split()]
DURS = [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "10 20").split()]
START_KM = 200.0
PARALLEL = int(os.environ.get("UWFLOW_EVAL_WORKERS", "4"))
INCLUDE_IDLE_RANDOM = os.environ.get("INCLUDE_IDLE_RANDOM", "1") == "1"


def models_for(depth, vmax, dur):
    # Final manuscript comparison only. Exclude historical reward/algorithm
    # ablations and all shorter-duration runs from the same model directory.
    if os.environ.get("CONSTRAINED_UWFLOW_NAMES", "0") == "1":
        pat = os.path.join(MODELDIR, "uw_d%g_v%g_s*.zip" % (depth, vmax))
    else:
        pat = os.path.join(
            MODELDIR,
            "ppo_uwflow_500_d%g_v%.1f_rC_%gh_sb3_s*.zip" % (depth, vmax, dur),
        )
    files = sorted(glob.glob(pat))
    return [os.path.basename(f) for f in files]


def evaluate_combo(depth, vmax, dur, models):
    tag = "d%g_v%.1f_%gh" % (depth, vmax, dur)
    log = os.path.join(LOGDIR, tag + ".log")
    result_name = (
        "eval_info_info_d%g_start%g_%gh_v%g_%d.csv"
        % (depth, START_KM, dur, vmax, N_TEST)
    )
    env = dict(os.environ)
    env.update({
        # GRID_DEG controls the policy observation interpolation. Coverage is
        # computed independently on the native HYCOM 0.04 x 0.08 degree grid.
        "VMAX": str(vmax), "REWARD": "C", "GRID_DEG": "0.25",
        "SPLIT_JSON": SPLIT, "SEGMENTS_CSV": SEGS,
        "EVAL_TAG": "_v%g_%d" % (vmax, N_TEST),
        "INCLUDE_IDLE_RANDOM": "1" if INCLUDE_IDLE_RANDOM else "0",
        "OMP_NUM_THREADS": os.environ.get("OMP_THREADS", "4"),
        "MKL_NUM_THREADS": os.environ.get("OMP_THREADS", "4"),
    })
    argv = [RL, "eval_uwflow_info.py", str(N_SEED), "%g" % depth,
            "%g" % START_KM, "%g" % dur] + models
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(argv, env=env, stdout=f, stderr=subprocess.STDOUT)
    return tag, rc, (time.time() - t0) / 60.0, log, result_name


def main():
    todo = []
    failures = []
    for depth in DEPTHS:
        for vmax in VMAXES:
            for dur in DURS:
                ms = models_for(depth, vmax, dur)
                if not ms:
                    raise SystemExit(
                        f"No final models for depth={depth:g}, "
                        f"vmax={vmax:g}, duration={dur:g}h"
                    )
                # Historically N_SEED was both the number of training models
                # and the number of deployment seeds.  A single-training-seed
                # validation run still needs multiple paired deployment seeds,
                # so the two quantities are now independent.
                if os.environ.get("CONSTRAINED_UWFLOW_NAMES", "0") != "1":
                    if len(ms) != N_SEED:
                        raise SystemExit(
                            f"Expected {N_SEED} final rC/SB3 models for "
                            f"depth={depth:g}, vmax={vmax:g}, duration={dur:g}h; "
                            f"found {len(ms)}"
                        )
                todo.append((depth, vmax, dur, ms))
    print("[eval] combos=%d n_seed=%d" % (len(todo), N_SEED), flush=True)

    pending = []
    for depth, vmax, dur, ms in todo:
        tag = "d%g_v%.1f_%gh" % (depth, vmax, dur)
        result_name = (
            "eval_info_info_d%g_start%g_%gh_v%g_%d.csv"
            % (depth, START_KM, dur, vmax, N_TEST)
        )
        result_path = os.path.join(OUT, result_name)
        if os.path.isfile(result_path):
            cached = pd.read_csv(result_path)
            expected_rows = N_SEED * N_TEST * (
                len(ms) + (2 if INCLUDE_IDLE_RANDOM else 0)
            )
            if len(cached) == expected_rows and \
               cached["policy"].nunique() == len(ms) + 2 and \
               cached["seed"].nunique() == N_SEED:
                print("      -> reuse complete result %s" % result_name, flush=True)
                continue
        pending.append((tag, depth, vmax, dur, ms))

    workers = max(1, min(PARALLEL, len(pending)))
    print("[eval] pending=%d parallel=%d" % (len(pending), workers), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(evaluate_combo, depth, vmax, dur, ms): tag
            for tag, depth, vmax, dur, ms in pending
        }
        for future in as_completed(futures):
            tag, rc, elapsed, log, _result_name = future.result()
            print(
                "      -> %s rc=%d  %.1f min  %s"
                % (tag, rc, elapsed, log),
                flush=True,
            )
            if rc:
                failures.append(tag)
    if failures:
        raise SystemExit("evaluation failed for: " + ", ".join(failures))
    print("[eval] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
