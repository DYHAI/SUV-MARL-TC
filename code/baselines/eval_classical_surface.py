#!/usr/bin/env python3
"""Evaluate causal classical planners on the surface TC-tracking task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from code.baselines.classical_planning_baselines import make_planner
from code.surface.fleet_segment_env import (
    D2KM,
    R_BUFFER,
    FleetSegmentEnv,
    deploy_formation,
    land_interp,
    safety_filter,
    seg_idx,
)
from code.paths import REPO_ROOT, RUNS_ROOT


ROOT = REPO_ROOT
SPLIT = ROOT / "data" / "typhoon_split_500.json"
SEGCSV = ROOT / "data" / "segments_500.csv"
OUTDIR = RUNS_ROOT / "baselines"


def deploy_fixed(d, t0, n_auv, radius_km, seed, formation="ring", ring_radius=100.0):
    lo, la = d["lon"], d["lat"]
    elo = float(d["track_lon"][t0])
    ela = float(d["track_lat"][t0])
    np.random.seed(seed)
    best = None
    for _ in range(300):
        _ = np.random.uniform(0.0, 1.0) * radius_km
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        clon = elo + radius_km * np.cos(angle) / (
            D2KM * np.cos(np.deg2rad(ela))
        )
        clat = ela + radius_km * np.sin(angle) / D2KM
        lons, lats = deploy_formation(
            clon, clat, n_auv, formation, 100.0, ring_radius
        )
        lons = np.clip(lons, lo.min(), lo.max())
        lats = np.clip(lats, la.min(), la.max())
        clearance = float(
            land_interp()(np.stack([lats, lons], axis=1)).min()
        )
        if best is None or clearance > best[0]:
            best = (clearance, lons.copy(), lats.copy())
        if clearance > 100.0:
            break
    return best[1], best[2], float(best[0])


def run_episode(planner, env, row, idx, deployment_seed, radius_km):
    d = env.data[row["storm"]]
    env.d = d
    env.t0 = seg_idx(row["t0"])
    env.t1 = seg_idx(row["t1"])
    env.t = env.t0
    env.vxs = np.zeros(env.n_auv, dtype=np.float32)
    env.vys = np.zeros(env.n_auv, dtype=np.float32)
    seed = 100000 + 97 * idx + deployment_seed
    env.lons, env.lats, initial_clearance = deploy_fixed(
        d, env.t0, env.n_auv, radius_km, seed
    )
    env.prev_actions = np.zeros((env.n_auv, 2), dtype=np.float32)
    planner.reset(env)
    obs = env._obs().reshape(-1).astype(np.float32)
    executed = np.zeros((env.n_auv, 2), dtype=np.float32)
    distances = []
    path = 0.0
    land_auv_steps = 0
    auv_steps = 0
    steps = 0
    while True:
        raw = np.asarray(planner.act(obs, env), dtype=np.float32).reshape(
            env.n_auv, 2
        )
        executed = 0.8 * executed + 0.2 * raw
        norm = np.linalg.norm(executed, axis=1, keepdims=True)
        action = executed / np.maximum(1.0, norm)
        action = safety_filter(
            action, env.lons, env.lats, env.dist_interp, 100.0
        )
        path += float(
            np.mean(np.linalg.norm(action, axis=1))
            * env.vmax
            * 10.0
            * 60.0
            / 1000.0
        )
        obs, _reward, terminated, truncated, info = env.step(action)
        obs = obs.reshape(-1).astype(np.float32)
        distances.append(float(np.mean(info["dist_km"])))
        land_auv_steps += int((info["land_dist"] <= 0.0).sum())
        auv_steps += env.n_auv
        steps += 1
        if terminated or truncated:
            break
    duration_h = steps * 10.0 / 60.0
    return {
        "planner": planner.__class__.__name__,
        "n_auv": env.n_auv,
        "vmax_ms": float(env.vmax),
        "radius_km": float(radius_km),
        "deployment_seed": deployment_seed,
        "idx": idx,
        "storm": row["storm"],
        "t0": env.t0,
        "distance_avg_km": float(np.mean(distances)),
        "final_distance_km": float(distances[-1]),
        "path_km": path,
        "mean_speed_ms": path * 1000.0 / max(duration_h * 3600.0, 1e-9),
        "on_land_fraction": land_auv_steps / max(auv_steps, 1),
        "initial_clearance_km": initial_clearance,
        "duration_h": duration_h,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("planner", choices=("potential", "astar", "coverage", "greedy"))
    parser.add_argument("--n-auv", type=int, required=True)
    parser.add_argument("--vmax", type=float, required=True)
    parser.add_argument("--radii", type=float, nargs="+", default=[100, 200, 400, 600, 800])
    parser.add_argument("--split-key", choices=("train", "test", "all"), default="test")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    split = json.loads(SPLIT.read_text())
    storms = (
        split["train"] + split["test"]
        if args.split_key == "all"
        else split[args.split_key]
    )
    seg = pd.read_csv(SEGCSV)
    seg = seg[seg.storm.isin(storms)].reset_index(drop=True)
    if args.limit > 0:
        seg = seg.iloc[: args.limit].reset_index(drop=True)
    storms = sorted(seg.storm.unique())

    os.environ["DEPLOY_OCEAN_ONLY"] = "1"
    os.environ["DEPLOY_MIN_COAST_KM"] = "100"
    os.environ["SAFETY_MARGIN_KM"] = "100"
    env = FleetSegmentEnv(
        storms=storms,
        segments_df=seg,
        use_uv=True,
        n_auv=args.n_auv,
        smooth_penalty=0.05,
        spacing_deg=0.25,
        patch_n=1,
        deploy_radius=800.0,
        formation="ring",
        ring_radius=100.0,
        sst_mode="none",
        use_mslp=False,
        vmax=args.vmax,
    )
    planner = make_planner(args.planner, "surface")
    rows = []
    for radius in args.radii:
        for idx, row in seg.iterrows():
            for deployment_seed in range(args.seeds):
                rows.append(
                    run_episode(
                        planner, env, row, idx, deployment_seed, radius
                    )
                )
            if (idx + 1) % 100 == 0:
                print(
                    f"[surface/{args.planner}] vmax={args.vmax:g} "
                    f"N={args.n_auv} D={radius:g} {idx + 1}/{len(seg)}",
                    flush=True,
                )
    out = args.out or (
        OUTDIR
        / f"surface_{args.planner}_{args.split_key}_n{args.n_auv}"
        f"_v{args.vmax:g}_s{args.seeds}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out, flush=True)


if __name__ == "__main__":
    main()
