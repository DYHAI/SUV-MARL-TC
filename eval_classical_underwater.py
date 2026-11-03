#!/usr/bin/env python3
"""Evaluate causal classical planners on the seven-depth subsurface task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from classical_planning_baselines import make_planner
from underwater_fleet_flow_env import D2KM, UnderwaterFleetFlowEnv
from underwater_flow_env import STEP_MIN


ROOT = Path(__file__).resolve().parent
SPLIT = ROOT / "data" / "typhoon_split_500.json"
SEGCSV = ROOT / "data" / "segments_500.csv"
OUTDIR = ROOT / "classical_eval"


def selected_segments(storms: list[str]) -> pd.DataFrame:
    seg = pd.read_csv(SEGCSV)
    seg = seg[seg.storm.isin(storms)].reset_index(drop=True)
    picks = [part.iloc[len(part) // 2] for _, part in seg.groupby("storm")]
    return pd.DataFrame(picks).reset_index(drop=True)


def run_episode(planner, env, row, seed):
    env.segments = pd.DataFrame([row]).reset_index(drop=True)
    np.random.seed(seed)
    obs, _ = env.reset()
    planner.reset(env)
    lats, lons, speeds, actions = [], [], [], []
    done = False
    while not done:
        action = np.asarray(planner.act(obs, env), dtype=np.float64).reshape(
            env.n_auv, 2
        )
        lats.append(np.asarray(env.lats).copy())
        lons.append(np.asarray(env.lons).copy())
        actions.append(np.linalg.norm(action, axis=1))
        obs, _reward, terminated, truncated, info = env.step(action)
        speeds.append(np.asarray(info["spd_at"], dtype=np.float64))
        done = bool(terminated or truncated)

    LAT = np.asarray(lats, dtype=np.float64)
    LON = np.asarray(lons, dtype=np.float64)
    SPD = np.asarray(speeds, dtype=np.float64)
    ACT = np.asarray(actions, dtype=np.float64)
    if len(LAT) > 1:
        step_km = np.hypot(
            (LAT[1:] - LAT[:-1]) * D2KM,
            (LON[1:] - LON[:-1])
            * D2KM
            * np.cos(np.deg2rad(LAT[:-1])),
        )
        path_km = float(step_km.sum(axis=0).mean())
    else:
        path_km = 0.0
    duration_s = max(len(LAT) - 1, 1) * STEP_MIN * 60.0

    cov_deg = float(env.cov_deg)
    lat0 = float(np.min(env.d["lat"]))
    lon0 = float(np.min(env.d["lon"]))
    cells = {
        (
            int(np.floor((lat - lat0) / cov_deg)),
            int(np.floor((lon - lon0) / cov_deg)),
        )
        for lat, lon in zip(LAT.ravel(), LON.ravel())
    }
    return {
        "planner": planner.__class__.__name__,
        "depth_m": float(env.depth_m),
        "vmax_ms": float(env.vmax),
        "deployment_seed": seed,
        "storm": row["storm"],
        "mean_current_ms": float(SPD.mean()),
        "coverage_area_km2": float(len(cells) * env._cell_area_km2),
        "path_km": path_km,
        "mean_ground_speed_ms": path_km * 1000.0 / max(duration_s, 1e-9),
        "command_proxy": float(np.mean(ACT * ACT)),
        "duration_h": len(ACT) * STEP_MIN / 60.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("planner", choices=("potential", "astar", "coverage", "greedy"))
    parser.add_argument("--depth", type=float, required=True)
    parser.add_argument("--vmax", type=float, required=True)
    parser.add_argument("--split-key", choices=("train", "test", "all"), default="test")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-storms", type=int, default=50)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    split = json.loads(SPLIT.read_text())
    storms = (
        split["train"] + split["test"]
        if args.split_key == "all"
        else split[args.split_key]
    )
    if args.limit > 0:
        storms = storms[: args.limit]
    picks = selected_segments(storms)
    planner = make_planner(args.planner, "subsurface")
    rows = []
    for start in range(0, len(storms), args.batch_storms):
        batch = storms[start : start + args.batch_storms]
        sub = picks[picks.storm.isin(batch)].reset_index(drop=True)
        env = UnderwaterFleetFlowEnv(
            storms=batch,
            segments_df=sub,
            n_auv=4,
            vmax=args.vmax,
            reward_mode="C",
            N=1,
            use_frame_stack=True,
            frame_k=5,
            start_radius=200.0,
            crop_margin=3.0,
            depth_m=args.depth,
            grid_deg=0.05,
            duration_h=20.0,
        )
        for _, row in sub.iterrows():
            for deployment_seed in range(args.seeds):
                # Match eval_uwflow_info.py exactly: deployment index sd uses
                # NumPy seed 100+sd for the learned-policy evaluation.
                rows.append(
                    run_episode(planner, env, row, 100 + deployment_seed)
                )
                rows[-1]["deployment_index"] = deployment_seed
            if (len(rows) // max(args.seeds, 1)) % 10 == 0:
                print(
                    f"[subsurface/{args.planner}] depth={args.depth:g} "
                    f"vmax={args.vmax:g} episodes={len(rows)}",
                    flush=True,
                )
    out = args.out or (
        OUTDIR
        / f"underwater_{args.planner}_{args.split_key}_d{args.depth:g}"
        f"_v{args.vmax:g}_s{args.seeds}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out, flush=True)


if __name__ == "__main__":
    main()
