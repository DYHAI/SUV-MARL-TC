#!/usr/bin/env python3
"""Evaluate causal classical planners on the seven-depth subsurface task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from code.baselines.classical_planning_baselines import make_planner
from code.underwater.underwater_fleet_flow_env import D2KM, UnderwaterFleetFlowEnv
from code.underwater.underwater_coverage import (
    HYCOM_LAT_DEG,
    HYCOM_LON_DEG,
    native_hycom_cells,
    native_hycom_coverage_area,
    native_hycom_weighted_coverage,
)
from code.underwater.underwater_flow_env import STEP_MIN


ROOT = Path(__file__).resolve().parents[2]
SPLIT = Path(
    os.environ.get("SPLIT_JSON", ROOT / "data" / "typhoon_split_400.json")
)
SEGCSV = Path(
    os.environ.get("SEGMENTS_CSV", ROOT / "data" / "segments_primary_400.csv")
)
OUTDIR = Path(
    os.environ.get("CLASSICAL_OUT_DIR", str(ROOT / "classical_eval"))
)


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
    max_prop_speed = 0.0
    max_accel = 0.0
    max_yaw_rate = 0.0
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
        if env.dynamics is not None:
            max_prop_speed = max(
                max_prop_speed, float(np.max(info["speed"]))
            )
            max_accel = max(max_accel, float(np.max(info["accel_peak"])))
            max_yaw_rate = max(
                max_yaw_rate, float(np.max(info["yaw_rate_peak"]))
            )
        done = bool(terminated or truncated)

    # Include the state reached by the final action in path and coverage.
    lats.append(np.asarray(env.lats).copy())
    lons.append(np.asarray(env.lons).copy())
    final_u = env._at2m("u", env.lats, env.lons)
    final_v = env._at2m("v", env.lats, env.lons)
    speeds.append(np.hypot(final_u, final_v).astype(np.float64))
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
    duration_s = max(len(ACT), 1) * STEP_MIN * 60.0

    land_distance = env._at(
        env.d["dist_to_land"].astype(np.float32)
        if getattr(env, "storage16", False)
        else env.d["dist_to_land"],
        LAT.ravel(),
        LON.ravel(),
    )
    ocean_valid = np.asarray(land_distance, dtype=np.float64) > 0.0
    cells = native_hycom_cells(LAT.ravel(), LON.ravel(), ocean_valid)
    coverage_area = native_hycom_coverage_area(cells)
    mean_current = float(SPD.ravel()[ocean_valid].mean())
    cellwise_weighted = native_hycom_weighted_coverage(
        LAT.ravel(),
        LON.ravel(),
        SPD.ravel(),
        ocean_valid,
    )
    result = {
        "planner": planner.__class__.__name__,
        "depth_m": float(env.depth_m),
        "vmax_ms": float(env.vmax),
        "deployment_seed": seed,
        "storm": row["storm"],
        "mean_current_ms": mean_current,
        "coverage_area_km2": coverage_area,
        "current_weighted_coverage": cellwise_weighted,
        "current_weighted_coverage_product": (
            mean_current * coverage_area
        ),
        "coverage_lat_deg": HYCOM_LAT_DEG,
        "coverage_lon_deg": HYCOM_LON_DEG,
        "path_km": path_km,
        "mean_ground_speed_ms": path_km * 1000.0 / max(duration_s, 1e-9),
        "command_proxy": float(np.mean(ACT * ACT)),
        "duration_h": duration_s / 3600.0,
        "dynamics_mode": (
            env.dynamics.config.mode if env.dynamics is not None else "kinematic"
        ),
        "max_prop_speed_ms": max_prop_speed,
        "max_accel_mps2": max_accel,
        "max_yaw_rate_dps": np.degrees(max_yaw_rate),
    }
    if env.dynamics is not None:
        config = env.dynamics.config
        assert max_prop_speed <= env.vmax + 1e-6
        assert max_accel <= max(
            config.accel_mps2, config.decel_mps2
        ) + 1e-6
        assert np.degrees(max_yaw_rate) <= np.degrees(
            config.yaw_rate_max_rad_s
        ) + 1e-6
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("planner", choices=("potential", "astar", "coverage", "greedy"))
    parser.add_argument("--depth", type=float, required=True)
    parser.add_argument("--vmax", type=float, required=True)
    parser.add_argument("--split-key", choices=("train", "test", "all"), default="test")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-storms", type=int, default=50)
    parser.add_argument("--grid-deg", type=float, default=0.25)
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
            grid_deg=args.grid_deg,
            cov_deg=args.grid_deg,
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
    expected = len(picks) * args.seeds
    assert len(rows) == expected, f"underwater rows: {len(rows)}/{expected}"
    temp = out.with_name(out.name + ".tmp")
    pd.DataFrame(rows).to_csv(temp, index=False)
    temp.replace(out)
    print(out, flush=True)


if __name__ == "__main__":
    main()
