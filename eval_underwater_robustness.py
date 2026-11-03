#!/usr/bin/env python3
"""Evaluate frozen underwater IPPO policies under environmental and OOD shifts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, TD3

from underwater_fleet_flow_env import UnderwaterFleetFlowEnv
from underwater_flow_env import D2KM, STEP_MIN


ROOT = Path(__file__).resolve().parent
SPLIT = ROOT / "data" / "typhoon_split_500.json"
SEGCSV = ROOT / "data" / "segments_500.csv"
OUTDIR = ROOT / "robustness_eval"


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
    "current_bias_low": {"current_bias": 0.80},
    "current_bias_high": {"current_bias": 1.20},
    "sensor_noise_0p05ms": {"sensor_noise_ms": 0.05},
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


def selected_segments(storms: list[str]) -> pd.DataFrame:
    seg = pd.read_csv(SEGCSV)
    seg = seg[seg["storm"].isin(storms)].reset_index(drop=True)
    picks = [part.iloc[len(part) // 2] for _, part in seg.groupby("storm")]
    return pd.DataFrame(picks).reset_index(drop=True)


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


def filter_storms(meta: pd.DataFrame, ood: str | None) -> list[str]:
    if ood is None:
        return meta["storm"].tolist()
    if ood == "early":
        out = meta[meta["season"] < 2015]
    elif ood == "recent":
        out = meta[meta["season"] >= 2020]
    elif ood == "weak":
        out = meta[meta["max_wind_kt"] < 96.0]
    elif ood == "intense":
        out = meta[meta["max_wind_kt"] >= 96.0]
    elif ood == "south":
        # Classification is applied at the selected segment rather than event.
        return meta["storm"].tolist()
    elif ood == "north":
        return meta["storm"].tolist()
    else:
        raise ValueError(ood)
    return out["storm"].tolist()


def perturb_observation(obs: np.ndarray, cfg: dict) -> np.ndarray:
    out = np.asarray(obs, dtype=np.float32).copy()
    if "current_bias" in cfg:
        out[..., :15] *= float(cfg["current_bias"])
    if "sensor_noise_ms" in cfg:
        out[..., :15] += np.random.normal(
            0.0,
            float(cfg["sensor_noise_ms"]),
            size=out[..., :15].shape,
        ).astype(np.float32)
    return out


def run_episode(env, model, row, seed, cfg):
    env.segments = pd.DataFrame([row]).reset_index(drop=True)
    np.random.seed(seed)
    obs, _ = env.reset()
    current_scale = float(cfg.get("current_bias", 1.0))
    if current_scale != 1.0:
        for history in env.flow_hist:
            for frame in history:
                frame *= current_scale
    lats, lons, speeds, action_magnitudes = [], [], [], []
    delayed = np.zeros((env.n_auv, 2), dtype=np.float32)
    done = False
    while not done:
        obs_in = perturb_observation(obs, cfg)
        raw, _ = model.predict(obs_in.reshape(-1), deterministic=True)
        raw = np.asarray(raw, dtype=np.float32).reshape(env.n_auv, 2)
        if int(cfg.get("action_delay", 0)) > 0:
            action = delayed.copy()
            delayed = raw
        else:
            action = raw
        dead_zone = float(cfg.get("dead_zone", 0.0))
        if dead_zone > 0:
            action = np.where(np.abs(action) < dead_zone, 0.0, action)
        action /= np.maximum(1.0, np.linalg.norm(action, axis=1, keepdims=True))
        lats.append(np.asarray(env.lats).copy())
        lons.append(np.asarray(env.lons).copy())
        action_magnitudes.append(np.linalg.norm(action, axis=1))
        obs, _reward, terminated, truncated, info = env.step(action)
        speeds.append(np.asarray(info["spd_at"], dtype=np.float64))
        done = bool(terminated or truncated)

    LAT = np.asarray(lats, dtype=np.float64)
    LON = np.asarray(lons, dtype=np.float64)
    SPD = np.asarray(speeds, dtype=np.float64)
    ACT = np.asarray(action_magnitudes, dtype=np.float64)
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
        "mean_current_ms": float(SPD.mean()),
        "coverage_area_km2": float(len(cells) * env._cell_area_km2),
        "path_km": path_km,
        "mean_ground_speed_ms": path_km * 1000.0 / max(duration_s, 1e-9),
        "command_proxy": float(np.mean(ACT * ACT)),
        "duration_h": len(ACT) * STEP_MIN / 60.0,
        "storm": row["storm"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--depth", type=float, default=30.0)
    parser.add_argument("--vmax", type=float, default=1.5)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUTDIR / "underwater_robustness.csv")
    args = parser.parse_args()

    split = json.loads(SPLIT.read_text())
    test_storms = split["test"]
    meta = storm_metadata(test_storms)
    picks_all = selected_segments(test_storms)
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
            storms = filter_storms(meta, cfg.get("ood"))
            picks = picks_all[picks_all["storm"].isin(storms)].reset_index(drop=True)
            if cfg.get("ood") == "south":
                picks = picks[picks["end_lat"] < 20.0].reset_index(drop=True)
            elif cfg.get("ood") == "north":
                picks = picks[picks["end_lat"] >= 20.0].reset_index(drop=True)
            if args.limit > 0:
                picks = picks.iloc[: args.limit].reset_index(drop=True)
            env = UnderwaterFleetFlowEnv(
                storms=sorted(picks["storm"].unique()),
                segments_df=picks,
                n_auv=4,
                vmax=args.vmax * float(cfg.get("speed_calib", 1.0)),
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
            current_scale = float(cfg.get("current_bias", 1.0))
            if current_scale != 1.0:
                original_at2m = env._at2m

                def scaled_at2m(
                    key, lats, lons, _original=original_at2m, _scale=current_scale
                ):
                    return _original(key, lats, lons) * _scale

                env._at2m = scaled_at2m
            for _, row in picks.iterrows():
                for deployment_index in range(args.seeds):
                    result = run_episode(
                        env,
                        model,
                        row,
                        100 + deployment_index,
                        cfg,
                    )
                    result.update(
                        {
                            "model": model_name,
                            "model_seed": model_seed(model_path),
                            "condition": condition,
                            "depth_m": args.depth,
                            "vmax_ms": args.vmax,
                            "deployment_index": deployment_index,
                        }
                    )
                    rows.append(result)
                if len(rows) % (max(args.seeds, 1) * 10) == 0:
                    print(
                        f"[{condition}] {Path(model_path).name} "
                        f"episodes={sum(r['condition'] == condition and r['model'] == Path(model_path).name for r in rows)}",
                        flush=True,
                    )
            print(
                f"[{condition}] completed {Path(model_path).name}: "
                f"{len(picks) * args.seeds} episodes",
                flush=True,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(args.out)


if __name__ == "__main__":
    main()
