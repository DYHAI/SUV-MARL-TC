#!/usr/bin/env python3
"""Evaluate frozen surface IPPO policies under deployment and OOD perturbations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, TD3

from code.surface.fleet_segment_env import (
    D2KM,
    R_BUFFER,
    FleetSegmentEnv,
    deploy_formation,
    land_interp,
    seg_idx,
)
from code.paths import REPO_ROOT, RUNS_ROOT


ROOT = REPO_ROOT
SPLIT = ROOT / "data" / "typhoon_split_500.json"
SEGCSV = ROOT / "data" / "segments_500.csv"
OUTDIR = RUNS_ROOT / "surface" / "robustness"
WIND_NORM = 50.0


def model_seed(path: str) -> int:
    match = re.search(r"_s(\d+)$", Path(path).stem)
    return int(match.group(1)) if match else 0


def load_policy(path: str):
    for policy in (PPO, SAC, TD3):
        try:
            return policy.load(path, device="cpu")
        except Exception:
            continue
    raise RuntimeError(f"cannot load supported policy: {path}")


CONDITIONS = {
    "clean": {},
    "wind_bias_low": {"wind_bias": 0.80},
    "wind_bias_high": {"wind_bias": 1.20},
    "sensor_noise_1ms": {"sensor_noise_ms": 1.0},
    "action_delay_1": {"action_delay": 1},
    "dead_zone_0p10": {"dead_zone": 0.10},
    "speed_calib_0p85": {"speed_calib": 0.85},
    "speed_calib_1p15": {"speed_calib": 1.15},
    "ood_early_pre2015": {"ood": "early"},
    "ood_recent_2020plus": {"ood": "recent"},
    "ood_weak_tc": {"ood": "weak"},
    "ood_intense_tc": {"ood": "intense"},
    "ood_south_lt20n": {"ood": "south"},
    "ood_north_ge20n": {"ood": "north"},
}


def deploy_fixed(d, t0, n_auv, radius_km, seed, formation="ring"):
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
            clon, clat, n_auv, formation, 100.0, 100.0
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
    return best[1], best[2]


def storm_metadata(storms: list[str]) -> pd.DataFrame:
    rows = []
    for storm in storms:
        path = ROOT / "data" / "selected_typhoons" / storm / "track.csv"
        frame = pd.read_csv(path, low_memory=False)
        season = int(pd.to_numeric(frame["SEASON"], errors="coerce").dropna().iloc[0])
        winds = pd.concat(
            [
                pd.to_numeric(frame.get("WMO_WIND"), errors="coerce"),
                pd.to_numeric(frame.get("USA_WIND"), errors="coerce"),
            ]
        )
        rows.append(
            {"storm": storm, "season": season, "max_wind_kt": float(winds.max())}
        )
    return pd.DataFrame(rows)


def filter_segments(
    seg: pd.DataFrame, meta: pd.DataFrame, ood: str | None
) -> pd.DataFrame:
    if ood is None:
        return seg.reset_index(drop=True)
    merged = seg.merge(meta, on="storm", how="left")
    if ood == "early":
        out = merged[merged["season"] < 2015]
    elif ood == "recent":
        out = merged[merged["season"] >= 2020]
    elif ood == "weak":
        out = merged[merged["max_wind_kt"] < 96.0]
    elif ood == "intense":
        out = merged[merged["max_wind_kt"] >= 96.0]
    elif ood == "south":
        out = merged[merged["start_lat"] < 20.0]
    elif ood == "north":
        out = merged[merged["start_lat"] >= 20.0]
    else:
        raise ValueError(ood)
    return out.drop(columns=["season", "max_wind_kt"], errors="ignore").reset_index(
        drop=True
    )


def perturb_observation(
    obs: np.ndarray,
    n_auv: int,
    *,
    wind_bias: float,
    sensor_noise_ms: float,
) -> np.ndarray:
    out = np.asarray(obs, dtype=np.float32).copy()
    wind_channels = 3 * n_auv
    if wind_bias != 1.0:
        out[..., :wind_channels] *= wind_bias
    if sensor_noise_ms > 0:
        out[..., :wind_channels] += np.random.normal(
            0.0, sensor_noise_ms / WIND_NORM, size=out[..., :wind_channels].shape
        ).astype(np.float32)
    return out


def run_segment(
    model: PPO,
    env: FleetSegmentEnv,
    row: pd.Series,
    idx: int,
    radius_km: float,
    cfg: dict,
    formation: str = "ring",
) -> dict:
    d = env.data[row["storm"]]
    env.d = d
    env.t0 = seg_idx(row["t0"])
    env.t1 = seg_idx(row["t1"])
    env.t = env.t0
    env.vxs = np.zeros(env.n_auv, dtype=np.float32)
    env.vys = np.zeros(env.n_auv, dtype=np.float32)
    seed = 100000 + 97 * idx
    env.lons, env.lats = deploy_fixed(
        d, env.t0, env.n_auv, radius_km, seed, formation
    )
    env.prev_actions = np.zeros((env.n_auv, 2), dtype=np.float32)
    obs = env._obs()
    delayed = np.zeros((env.n_auv, 2), dtype=np.float32)
    executed = np.zeros((env.n_auv, 2), dtype=np.float32)
    distance, path, land_steps, steps = [], 0.0, 0, 0
    formation_radii: list[float] = []
    pairwise_spacings: list[float] = []
    while True:
        obs_in = perturb_observation(
            obs,
            env.n_auv,
            wind_bias=float(cfg.get("wind_bias", 1.0)),
            sensor_noise_ms=float(cfg.get("sensor_noise_ms", 0.0)),
        )
        raw, _ = model.predict(obs_in.reshape(-1), deterministic=True)
        raw = np.asarray(raw, dtype=np.float32).reshape(env.n_auv, 2)
        delay = int(cfg.get("action_delay", 0))
        if delay == 0:
            command = raw
        else:
            command = delayed.copy()
            delayed = raw
        executed = 0.2 * command + 0.8 * executed
        action = executed.copy()
        dead_zone = float(cfg.get("dead_zone", 0.0))
        if dead_zone > 0:
            action = np.where(np.abs(action) < dead_zone, 0.0, action)
        action /= np.maximum(1.0, np.linalg.norm(action, axis=1, keepdims=True))
        path += float(np.mean(np.linalg.norm(action, axis=1))) * env.vmax * 10.0 * 60.0 / 1000.0
        obs, _reward, terminated, truncated, info = env.step(action)
        distance.append(float(np.mean(info["dist_km"])))
        lat0 = float(np.mean(env.lats))
        dx = (env.lons - float(np.mean(env.lons))) * D2KM * np.cos(
            np.deg2rad(lat0)
        )
        dy = (env.lats - lat0) * D2KM
        formation_radii.append(float(np.sqrt(dx * dx + dy * dy).mean()))
        if env.n_auv > 1:
            ii, jj = np.triu_indices(env.n_auv, k=1)
            pair_km = np.hypot(
                (env.lons[ii] - env.lons[jj])
                * D2KM
                * np.cos(np.deg2rad(env.lats[ii])),
                (env.lats[ii] - env.lats[jj]) * D2KM,
            )
            pairwise_spacings.append(float(pair_km.mean()))
        land_steps += int((info["land_dist"] <= 0.0).sum())
        steps += 1
        if terminated or truncated:
            break
    return {
        "final_distance_km": distance[-1],
        "distance_avg_km": float(np.mean(distance)),
        "path_km": path,
        "on_land_fraction": land_steps / max(steps * env.n_auv, 1),
        "formation_radius_km": float(np.mean(formation_radii)),
        "pairwise_spacing_km": float(np.mean(pairwise_spacings)),
        "duration_h": steps * 10.0 / 60.0,
        "storm": row["storm"],
        "segment_index": idx,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--n-auv", type=int, default=8)
    parser.add_argument("--vmax", type=float, default=20.0)
    parser.add_argument("--radius", type=float, default=400.0)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--formation", choices=("ring", "line", "grid", "scatter"), default="ring")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUTDIR / "surface_robustness.csv")
    args = parser.parse_args()

    split = json.loads(SPLIT.read_text())
    test_storms = split["test"]
    meta = storm_metadata(test_storms)
    seg_all = pd.read_csv(SEGCSV)
    seg_all = seg_all[seg_all["storm"].isin(test_storms)].reset_index(drop=True)

    rows: list[dict] = []
    completed: set[tuple[str, str]] = set()
    if args.out.is_file() and args.out.stat().st_size > 0:
        existing = pd.read_csv(args.out)
        rows = existing.to_dict("records")
        completed = set(zip(existing["model"], existing["condition"]))
    for model_path in args.models:
        model_name = Path(model_path).name
        model = load_policy(model_path)
        for condition in args.conditions:
            if (model_name, condition) in completed:
                continue
            cfg = CONDITIONS[condition]
            seg = filter_segments(seg_all, meta, cfg.get("ood"))
            if args.limit > 0:
                seg = seg.iloc[: args.limit].reset_index(drop=True)
            storms = sorted(seg["storm"].unique())
            env = FleetSegmentEnv(
                storms=storms,
                segments_df=seg,
                use_uv=True,
                n_auv=args.n_auv,
                smooth_penalty=0.05,
                spacing_deg=0.25,
                patch_n=1,
                deploy_radius=800.0,
                formation=args.formation,
                ring_radius=100.0,
                sst_mode="none",
                use_mslp=False,
                vmax=args.vmax * float(cfg.get("speed_calib", 1.0)),
            )
            for idx, row in seg.iterrows():
                result = run_segment(
                    model, env, row, idx, args.radius, cfg, args.formation
                )
                result.update(
                    {
                        "model": model_name,
                        "model_seed": model_seed(model_path),
                        "condition": condition,
                        "n_auv": args.n_auv,
                        "vmax_ms": args.vmax,
                        "radius_km": args.radius,
                    }
                )
                rows.append(result)
                if (idx + 1) % 100 == 0:
                    print(
                        f"[{condition}] {Path(model_path).name} {idx + 1}/{len(seg)}",
                        flush=True,
                    )
            print(
                f"[{condition}] completed {Path(model_path).name}: {len(seg)} segments",
                flush=True,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(args.out)


if __name__ == "__main__":
    main()
