# -*- coding: utf-8 -*-
"""Underwater 4-AUV max-flow matrix -- batch evaluation driver (2026-10-10).

For each (depth, Vmax, duration) combo run eval_uwflow_info.py ONCE; that single
pass evaluates idle(drift) / random / every trained model sharing that combo
(reward B/C x algo IPPO/SB3 x 3 seeds), all paired on the same episodes.

Usage:
    python run_uwflow_eval.py <n_seed> [depths] [vmaxes] [durations]
    # default: 10 "0 30 50" "1.5 2.0" "10 20"
Output:
    eval_out/eval_info_info_d<depth>_start200_<dur>h_v<vmax>_500.csv
"""
import os, sys, glob, subprocess, time

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
RL = os.environ.get("PYTHON_BIN", sys.executable)
OUT = os.path.join(BASE, "eval_out")
os.makedirs(OUT, exist_ok=True)
LOGDIR = os.path.join(BASE, "logs_uwflow_eval")
os.makedirs(LOGDIR, exist_ok=True)

SPLIT = os.path.join(BASE, "data", "typhoon_split_500.json")
SEGS = os.path.join(BASE, "data", "segments_500.csv")
MODELDIR = os.path.join(BASE, "ppo_underwater_fleet")

N_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 10
DEPTHS = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0 30 50").split()]
VMAXES = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1.5 2.0").split()]
DURS = [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "10 20").split()]
START_KM = 200.0


def models_for(depth, vmax, dur):
    pat = os.path.join(
        MODELDIR,
        "ppo_uwflow_500_d%g_v%.1f_rC_%gh_sb3_s*.zip" % (depth, vmax, dur),
    )
    files = sorted(glob.glob(pat))
    return [os.path.basename(f) for f in files]


def main():
    todo = []
    for depth in DEPTHS:
        for vmax in VMAXES:
            for dur in DURS:
                ms = models_for(depth, vmax, dur)
                if ms:
                    todo.append((depth, vmax, dur, ms))
    print("[eval] combos=%d n_seed=%d" % (len(todo), N_SEED), flush=True)

    for i, (depth, vmax, dur, ms) in enumerate(todo, 1):
        tag = "d%g_v%.1f_%gh" % (depth, vmax, dur)
        log = os.path.join(LOGDIR, tag + ".log")
        print("[%d/%d] %s  models=%d" % (i, len(todo), tag, len(ms)), flush=True)
        env = dict(os.environ)
        env.update({
            "VMAX": str(vmax), "REWARD": "C", "GRID_DEG": "0.25",
            "SPLIT_JSON": SPLIT, "SEGMENTS_CSV": SEGS,
            "EVAL_TAG": "_v%g_500" % vmax,
            "OMP_NUM_THREADS": os.environ.get("OMP_THREADS", "4"),
            "MKL_NUM_THREADS": os.environ.get("OMP_THREADS", "4"),
        })
        argv = [RL, "eval_uwflow_info.py", str(N_SEED), "%g" % depth,
                "%g" % START_KM, "%g" % dur] + ms
        t0 = time.time()
        with open(log, "w") as f:
            rc = subprocess.call(argv, env=env, stdout=f, stderr=subprocess.STDOUT)
        print("      -> rc=%d  %.1f min  %s" % (rc, (time.time() - t0) / 60.0, log), flush=True)
    print("[eval] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
