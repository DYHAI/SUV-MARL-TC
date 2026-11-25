#!/usr/bin/env python3
"""Render representative high-error surface-tracking cases for the Discussion."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DYNAMICS_MODE", "constrained")
os.environ.setdefault("DYN_SURFACE_CURRENT", "0")
os.environ.setdefault("OBS_HEADING", "1")
os.environ.setdefault("RIGID_FORMATION", "1")
os.environ.setdefault("DEPLOY_OCEAN_ONLY", "1")
os.environ.setdefault("DEPLOY_MIN_COAST_KM", "100")
os.environ.setdefault("SAFETY_MARGIN_KM", "100")
os.environ.setdefault("FLEET_VMAX", "12")
os.environ.setdefault(
    "SPLIT_JSON",
    str(Path(__file__).resolve().parents[2] / "data" / "typhoon_split_400.json"),
)
os.environ.setdefault(
    "SEGMENTS_CSV",
    str(
        Path(__file__).resolve().parents[2]
        / "data"
        / "segments_primary_400.csv"
    ),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from code.surface.fleet_segment_env import (  # noqa: E402
    D2KM,
    FleetSegmentEnv,
    coerce_formation_action,
    deploy_formation,
    formation_safety_filter,
    land_interp,
    seg_idx,
)
from code.surface.surface_wind_env import load_storm  # noqa: E402


DOC = ROOT / "paper"
OUT = DOC / "figures" / "FigB2_Limitations_Cases.pdf"
MODEL = (
    ROOT
    / "runs"
    / "official"
    / "models"
    / "surface_primary_circle"
    / "surface_n4_v12_s0.zip"
)
DEPLOY_KM = 800.0
BETA = 0.8
COMMON_LON_SPAN_DEG = 27.0
COMMON_LAT_SPAN_DEG = 16.0
CASES = (
    {
        "storm": "MARIA_2012_2012287",
        "display": "MARIA_2012",
        "t0": 745.0,
        "label": "(a)",
    },
    {
        "storm": "KUJIRA_2020",
        "t0": 604.0,
        "label": "(b)",
    },
)


def deploy_fixed(d, t0, n_auv, radius, seed):
    lo = d["lon"]
    la = d["lat"]
    eye_lon = float(d["track_lon"][t0])
    eye_lat = float(d["track_lat"][t0])
    np.random.seed(seed)
    best = None
    for _ in range(300):
        _ = np.random.uniform(0.0, 1.0) * radius
        bearing = np.random.uniform(0.0, 2.0 * np.pi)
        center_lon = eye_lon + radius * np.cos(bearing) / (
            D2KM * np.cos(np.deg2rad(eye_lat))
        )
        center_lat = eye_lat + radius * np.sin(bearing) / D2KM
        lons, lats = deploy_formation(
            center_lon, center_lat, n_auv, "ring", 100.0, 100.0
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


def run_case(model, all_segments, test_storms, case):
    storm = case["storm"]
    target_time = case["t0"]
    test_segments = all_segments[
        all_segments.storm.isin(test_storms)
    ].reset_index(drop=True)
    matches = test_segments[
        (test_segments.storm == storm)
        & np.isclose(test_segments.t0, target_time)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{storm} t0={target_time}: found {len(matches)} rows")
    index = int(matches.index[0])
    row = matches.iloc[0]
    env = FleetSegmentEnv(
        storms=[storm],
        segments_df=pd.DataFrame([row]),
        use_uv=True,
        n_auv=4,
        smooth_penalty=0.05,
        spacing_deg=0.25,
        patch_n=1,
        deploy_radius=DEPLOY_KM,
        formation="ring",
        ring_radius=100.0,
        sst_mode="none",
        use_mslp=False,
        vmax=12.0,
    )
    d = env.data[storm]
    env.d = d
    env.t0 = seg_idx(row["t0"])
    env.t1 = seg_idx(row["t1"])
    env.t = env.t0
    seed = 100000 + 97 * index
    if env.dynamics is not None:
        env.dynamics.reset(seed=seed)
    env.vxs = np.zeros(env.n_auv, dtype=np.float32)
    env.vys = np.zeros(env.n_auv, dtype=np.float32)
    env.lons, env.lats = deploy_fixed(
        d, env.t0, env.n_auv, DEPLOY_KM, seed
    )
    env._rigid_offsets_km = None
    env.prev_actions = np.zeros((env.n_auv, 2), dtype=np.float32)
    obs = env._obs()[0]
    executed = np.zeros((env.n_auv, 2), dtype=np.float32)
    trajectories = []
    eyes = []
    step_times = []
    while True:
        raw, _ = model.predict(obs, deterministic=True)
        action = coerce_formation_action(raw, env.n_auv)
        executed = (1.0 - BETA) * action + BETA * executed
        executed = np.clip(executed, -1.0, 1.0)
        executed /= np.maximum(
            1.0, np.linalg.norm(executed, axis=1, keepdims=True)
        )
        common = formation_safety_filter(
            executed[0],
            env.lons,
            env.lats,
            env.dist_interp,
            100.0,
        )
        executed = np.repeat(common[None, :], env.n_auv, axis=0)
        trajectories.append(np.stack([env.lons.copy(), env.lats.copy()], axis=1))
        eyes.append(
            (
                float(d["track_lon"][env.t]),
                float(d["track_lat"][env.t]),
            )
        )
        step_times.append(
            pd.Timestamp(d["times"][int(d["h0"][env.t])])
            + pd.Timedelta(
                hours=float(d["frac"][env.t])
                * (int(d["h1"][env.t]) - int(d["h0"][env.t]))
            )
        )
        obs, _, terminated, truncated, _ = env.step(executed)
        if terminated or truncated:
            trajectories.append(np.stack([env.lons.copy(), env.lats.copy()], axis=1))
            eyes.append(
                (
                    float(d["track_lon"][env.t]),
                    float(d["track_lat"][env.t]),
                )
            )
            step_times.append(
                pd.Timestamp(d["times"][int(d["h0"][env.t])])
                + pd.Timedelta(
                    hours=float(d["frac"][env.t])
                    * (int(d["h1"][env.t]) - int(d["h0"][env.t]))
                )
            )
            break
    return (
        np.asarray(trajectories),
        np.asarray(eyes),
        np.asarray(step_times),
        load_storm(storm, use_uv=False, crop_margin_km=2400.0),
        int(env.t),
        row,
        (env.lons.copy(), env.lats.copy()),
    )


def wind_snapshot(d, time_index):
    h0 = int(d["h0"][time_index])
    h1 = int(d["h1"][time_index])
    frac = float(d["frac"][time_index])
    return (1.0 - frac) * d["spd"][h0] + frac * d["spd"][h1]


def competitor_track(case, times):
    times = np.asarray(times)
    if case["storm"].startswith("MARIA_2012"):
        path = (
            ROOT
            / "data"
            / "selected_typhoons"
            / "PRAPIROON_2012_2012279"
            / "track.csv"
        )
        track = pd.read_csv(path, low_memory=False)
        track["time"] = pd.to_datetime(track["ISO_TIME"], errors="coerce")
        return track[
            (track.time >= pd.Timestamp(times[0]))
            & (track.time <= pd.Timestamp(times[-1]))
        ]
    return None


def interpolate_competitor(case, times):
    if not case["storm"].startswith("MARIA_2012"):
        return None, None
    path = (
        ROOT
        / "data"
        / "selected_typhoons"
        / "PRAPIROON_2012_2012279"
        / "track.csv"
    )
    track = pd.read_csv(path, low_memory=False)
    track = track[track["LON"].notna() & track["LAT"].notna()].copy()
    if track.empty:
        return None, None
    query = pd.to_datetime(pd.Series(times)).astype("int64").to_numpy()
    base = pd.to_datetime(track["ISO_TIME"]).astype("int64").to_numpy()
    lon = np.interp(query, base, track["LON"].to_numpy(dtype=float))
    lat = np.interp(query, base, track["LAT"].to_numpy(dtype=float))
    return lon, lat


def case_extent(result, case):
    trajectories, eyes, _step_times, d, final_t, row, _initial = result
    start_t = seg_idx(row["t0"])
    all_lon = trajectories[:, :, 0].ravel()
    all_lat = trajectories[:, :, 1].ravel()
    target_lon = d["track_lon"][start_t : final_t + 1]
    target_lat = d["track_lat"][start_t : final_t + 1]
    # Expand the plotted domain to show a wider surrounding wind field.
    pad = 3.2
    lon_min = min(all_lon.min(), target_lon.min()) - pad
    lon_max = max(all_lon.max(), target_lon.max()) + pad
    lat_min = min(all_lat.min(), target_lat.min()) - pad
    lat_max = max(all_lat.max(), target_lat.max()) + pad
    other = competitor_track(
        case,
        d["times"][
            int(d["h0"][start_t]) : int(d["h1"][final_t]) + 1
        ],
    )
    if other is not None and len(other):
        lon_min = min(lon_min, float(other["LON"].min()) - pad)
        lon_max = max(lon_max, float(other["LON"].max()) + pad)
        lat_min = min(lat_min, float(other["LAT"].min()) - pad)
        lat_max = max(lat_max, float(other["LAT"].max()) + pad)
    center_lon = 0.5 * (lon_min + lon_max)
    center_lat = 0.5 * (lat_min + lat_max)
    lon_min = min(lon_min, center_lon - 0.5 * COMMON_LON_SPAN_DEG)
    lon_max = max(lon_max, center_lon + 0.5 * COMMON_LON_SPAN_DEG)
    lat_min = min(lat_min, center_lat - 0.5 * COMMON_LAT_SPAN_DEG)
    lat_max = max(lat_max, center_lat + 0.5 * COMMON_LAT_SPAN_DEG)
    return lon_min, lon_max, lat_min, lat_max


def draw_snapshot(ax, result, case, frame, extent):
    trajectories, eyes, step_times, d, final_t, row, initial = result
    start_t = seg_idx(row["t0"])
    frame = min(int(frame), len(trajectories) - 1)
    time_index = start_t + frame
    lon_min, lon_max, lat_min, lat_max = extent
    lon_grid = np.linspace(lon_min, lon_max, 260)
    lat_grid = np.linspace(lat_min, lat_max, 260)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    interp = RegularGridInterpolator(
        (d["lat"], d["lon"]),
        wind_snapshot(d, time_index),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    speed = interp(np.stack([LAT, LON], axis=-1))
    mesh = ax.pcolormesh(
        lon_grid,
        lat_grid,
        speed,
        cmap="YlGnBu",
        norm=mcolors.Normalize(0.0, 30.0),
        shading="gouraud",
        rasterized=True,
    )
    ax.contour(
        lon_grid,
        lat_grid,
        speed,
        levels=np.arange(5.0, 31.0, 5.0),
        colors="white",
        linewidths=0.32,
        alpha=0.42,
    )
    for vehicle in range(trajectories.shape[1]):
        traj = trajectories[: frame + 1, vehicle]
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color="white",
            lw=2.0,
            alpha=0.78,
            zorder=4,
        )
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color="#1F5A99",
            lw=1.15,
            zorder=5,
        )
        ax.scatter(
            [traj[-1, 0]],
            [traj[-1, 1]],
            s=22,
            facecolors="#1F5A99",
            edgecolors="white",
            linewidths=0.55,
            zorder=7,
        )
    if frame == 0:
        ax.scatter(
            trajectories[0, :, 0],
            trajectories[0, :, 1],
            s=32,
            facecolors="white",
            edgecolors="#1F5A99",
            linewidths=0.8,
            zorder=8,
        )
    if frame == len(trajectories) - 1:
        ax.scatter(
            trajectories[-1, :, 0],
            trajectories[-1, :, 1],
            s=34,
            marker="^",
            facecolors="#1F5A99",
            edgecolors="white",
            linewidths=0.6,
            zorder=8,
        )
    target_lon = d["track_lon"][start_t : time_index + 1]
    target_lat = d["track_lat"][start_t : time_index + 1]
    ax.plot(
        target_lon,
        target_lat,
        color="#E63946",
        lw=1.35,
        label="Target TC track",
        zorder=9,
    )
    ax.scatter(
        [eyes[frame, 0]],
        [eyes[frame, 1]],
        marker="*",
        s=155,
        color="#E63946",
        edgecolors="white",
        linewidths=0.75,
        zorder=10,
    )
    target_name = case["storm"].split("_")[0]
    ax.annotate(
        target_name,
        (eyes[frame, 0], eyes[frame, 1]),
        xytext=(5, 5),
        textcoords="offset points",
        color="#B3202A",
        fontsize=19.0,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
        },
        zorder=13,
    )
    other_lon_all, other_lat_all = interpolate_competitor(case, step_times)
    if other_lon_all is not None:
        ax.plot(
            other_lon_all[: frame + 1],
            other_lat_all[: frame + 1],
            color="#F28E2B",
            lw=1.25,
            linestyle="--",
            label="PRAPIROON track",
            zorder=9,
        )
        if frame == 0:
            ax.scatter(
                [other_lon_all[0]],
                [other_lat_all[0]],
                marker="D",
                s=30,
                color="#F28E2B",
                edgecolors="white",
                linewidths=0.5,
                zorder=10,
            )
        ax.annotate(
            "PRAPIROON",
            (other_lon_all[frame], other_lat_all[frame]),
            xytext=(5, -11),
            textcoords="offset points",
            color="#B86512",
            fontsize=18.5,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
            zorder=13,
        )
    current_center = trajectories[frame].mean(axis=0)
    target_distance = float(
        np.hypot(
            (current_center[0] - eyes[frame, 0])
            * D2KM
            * np.cos(np.deg2rad(eyes[frame, 1])),
            (current_center[1] - eyes[frame, 1]) * D2KM,
        )
    )
    metrics_text = f"Target distance: {target_distance:.0f} km"
    if other_lon_all is not None:
        other_distance = float(
            np.hypot(
                (current_center[0] - other_lon_all[frame])
                * D2KM
                * np.cos(np.deg2rad(other_lat_all[frame])),
                (current_center[1] - other_lat_all[frame]) * D2KM,
            )
        )
        metrics_text += f"\nPRAPIROON distance: {other_distance:.0f} km"
    ax.text(
        0.02,
        0.02,
        metrics_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18.0,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": "0.45",
            "alpha": 0.88,
        },
        zorder=12,
    )
    elapsed_h = (
        pd.Timestamp(step_times[frame]) - pd.Timestamp(step_times[0])
    ).total_seconds() / 3600.0
    title = f"t = {elapsed_h:.1f} h"
    if frame == 0:
        display = case.get("display", case["storm"])
        title = f"{case['label']} {display} | {title}"
    ax.set_title(
        title,
        loc="left",
        fontsize=21.0,
        fontweight="bold",
        pad=3,
    )
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    # Both cases use the same 27 x 16 degree window. Fill identical axes boxes
    # so all eight panels have exactly the same physical size.
    ax.set_aspect("auto")
    ax.set_box_aspect(1.0 / 1.5)
    ax.grid(color="white", alpha=0.15, lw=0.35)
    return mesh


def main():
    plt.rcParams.update(
        {
            "font.size": 20.0,
            "axes.labelsize": 20.0,
            "xtick.labelsize": 17.0,
            "ytick.labelsize": 17.0,
        }
    )
    split = json.loads(Path(os.environ["SPLIT_JSON"]).read_text())
    all_segments = pd.read_csv(os.environ["SEGMENTS_CSV"])
    model = PPO.load(MODEL, device="cpu")
    results = [
        run_case(model, all_segments, split["test"], case)
        for case in CASES
    ]
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(20.5, 24.8),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(
        h_pad=0.02,
        w_pad=0.03,
        hspace=0.03,
        wspace=0.03,
    )
    mesh = None
    for case_index, (result, case) in enumerate(zip(results, CASES)):
        trajectories = result[0]
        frame_indices = np.unique(
            np.linspace(0, len(trajectories) - 1, 4).round().astype(int)
        )
        while len(frame_indices) < 4:
            frame_indices = np.unique(
                np.linspace(
                    0,
                    len(trajectories) - 1,
                    len(frame_indices) + 1,
                ).round().astype(int)
            )
        extent = case_extent(result, case)
        for panel in range(4):
            block_row = panel // 2
            block_col = panel % 2
            ax = axes[case_index * 2 + block_row, block_col]
            if panel >= len(frame_indices):
                ax.axis("off")
                continue
            mesh = draw_snapshot(
                ax,
                result,
                case,
                frame_indices[panel],
                extent,
            )
            if block_col == 0:
                ax.set_ylabel("Latitude")
            else:
                ax.tick_params(labelleft=False)
            if case_index == len(CASES) - 1:
                ax.set_xlabel("Longitude")
            else:
                ax.tick_params(labelbottom=False)
    handles = [
        Line2D([0], [0], marker="o", color="#1F5A99", linestyle="None",
               markerfacecolor="white", markeredgewidth=1.0,
               markersize=6, label="Initial SUV position"),
        Line2D([0], [0], marker="^", color="#1F5A99", linestyle="None",
               markerfacecolor="#1F5A99", markeredgecolor="white",
               markersize=6, label="Final SUV position"),
        Line2D(
            [0],
            [0],
            color="#E63946",
            lw=1.5,
            label="Target TC track/core",
        ),
        Line2D([0], [0], color="#F28E2B", lw=1.4, linestyle="--",
               label="Competing TC track"),
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=4,
        frameon=False,
        fontsize=24.0,
        borderaxespad=0.0,
    )
    cbar = fig.colorbar(
        mesh,
        ax=axes,
        orientation="horizontal",
        fraction=0.025,
        pad=0.025,
        aspect=55,
    )
    cbar.set_label("ERA5 10 m wind speed (m s$^{-1}$)")
    cbar.set_ticks(np.arange(0.0, 31.0, 5.0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
