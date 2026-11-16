#!/usr/bin/env python3
"""Compare surface trajectories with and without execution-layer filtering."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("FLEET_VMAX", "20")
os.environ.setdefault("RIGID_FORMATION", "1")
os.environ.setdefault("DEPLOY_OCEAN_ONLY", "1")
os.environ.setdefault("DEPLOY_MIN_COAST_KM", "100")
os.environ.setdefault("SAFETY_MARGIN_KM", "100")
os.environ.setdefault("SST_MODE", "none")
os.environ.setdefault("USE_MSLP", "0")
os.environ.setdefault("DYNAMICS_MODE", "constrained")
os.environ.setdefault("DYN_SURFACE_CURRENT", "0")
os.environ.setdefault("OBS_HEADING", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator
from stable_baselines3 import PPO

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from code.surface.fleet_segment_env import (  # noqa: E402
    D2KM,
    FleetSegmentEnv,
    coerce_formation_action,
    formation_safety_filter,
    policy_observation,
    rigid_formation_enabled,
    safety_filter,
    sample_deployment,
    seg_idx,
)


DOC = ROOT / "paper"
FIGDIR = DOC / "figures"
SEGMENTS = ROOT / "data" / "segments_primary_400.csv"
MODEL = Path(
    os.environ.get(
        "SURFACE_MODEL_DIR",
        ROOT
        / "runs"
        / "surface"
        / "models",
    )
) / "surface_n4_v20_s0.zip"
STORM = "MATMO_2025"
T0 = 561
RADIUS_KM = 400.0
BETA = 0.8
COLORS = ("#4E79A7", "#F28E2B", "#59A14F", "#E15759")
WIND_CMAP = plt.get_cmap("YlGnBu")


def wind_snapshot(data: dict, time_index: int) -> np.ndarray:
    h0 = int(data["h0"][time_index])
    h1 = int(data["h1"][time_index])
    fraction = float(data["frac"][time_index])
    return (1.0 - fraction) * data["spd"][h0] + fraction * data["spd"][h1]


def draw_wind(ax, data, time_index, extent):
    lon, lat = data["lon"], data["lat"]
    speed = wind_snapshot(data, time_index)
    lo0, lo1, la0, la1 = extent
    jm = (lon >= lo0) & (lon <= lo1)
    im = (lat >= la0) & (lat <= la1)
    j0, j1 = np.where(jm)[0][[0, -1]]
    i0, i1 = np.where(im)[0][[0, -1]]
    lon_c, lat_c = lon[j0:j1 + 1], lat[i0:i1 + 1]
    speed_c = speed[i0:i1 + 1, j0:j1 + 1]
    lon_f = np.linspace(lon_c[0], lon_c[-1], 6 * len(lon_c))
    lat_f = np.linspace(lat_c[0], lat_c[-1], 6 * len(lat_c))
    interpolator = RegularGridInterpolator(
        (lat_c, lon_c),
        speed_c,
        method="cubic",
        bounds_error=False,
        fill_value=np.nan,
    )
    speed_f = interpolator(
        np.stack(np.meshgrid(lat_f, lon_f, indexing="ij"), axis=-1)
    )
    speed_f = np.clip(speed_f, 0.0, 30.0)
    mesh = ax.pcolormesh(
        lon_f,
        lat_f,
        speed_f,
        cmap=WIND_CMAP,
        norm=mcolors.Normalize(0.0, 30.0),
        shading="gouraud",
        rasterized=True,
        zorder=0,
    )
    ax.contour(
        lon_f,
        lat_f,
        speed_f,
        levels=np.arange(5.0, 31.0, 5.0),
        colors="white",
        linewidths=0.4,
        alpha=0.48,
        zorder=1,
    )
    return mesh


def rollout(filtered: bool):
    row = pd.read_csv(SEGMENTS)
    row = row[(row.storm == STORM) & (row.t0 == T0)].iloc[0]
    env = FleetSegmentEnv(
        storms=[STORM],
        segments_df=pd.DataFrame([row]),
        use_uv=True,
        n_auv=4,
        smooth_penalty=0.05,
        spacing_deg=0.25,
        patch_n=1,
        deploy_radius=800.0,
        formation="ring",
        ring_radius=100.0,
        sst_mode="none",
        use_mslp=False,
        vmax=20.0,
    )
    data = env.data[STORM]
    env.d = data
    env.t0 = seg_idx(row["t0"])
    env.t1 = seg_idx(row["t1"])
    env.t = env.t0
    np.random.seed(17)
    if env.dynamics is not None:
        env.dynamics.reset(seed=17)
    env.lons, env.lats = sample_deployment(
        data,
        env.t0,
        env.n_auv,
        RADIUS_KM,
        formation="ring",
        ring_radius=100.0,
        deploy_mode="fixed",
        ocean_only=True,
        min_coast_km=100.0,
        fixed_km=RADIUS_KM,
    )
    env.vxs = np.zeros(env.n_auv)
    env.vys = np.zeros(env.n_auv)
    env.prev_actions = np.zeros((env.n_auv, 2), dtype=np.float32)
    model = PPO.load(MODEL, device="cpu")
    rigid = rigid_formation_enabled()
    obs = policy_observation(env._obs(), rigid)
    previous = np.zeros((env.n_auv, 2), dtype=np.float32)
    trajectories = [[(env.lons[i], env.lats[i])] for i in range(env.n_auv)]
    eyes = [(float(data["track_lon"][env.t]), float(data["track_lat"][env.t]))]
    while True:
        raw, _ = model.predict(obs, deterministic=True)
        if rigid:
            raw = coerce_formation_action(raw, env.n_auv)
        else:
            raw = np.asarray(raw, dtype=np.float32).reshape(env.n_auv, 2)
        if filtered:
            previous = (1.0 - BETA) * raw + BETA * previous
        else:
            previous = raw.copy()
        command = np.clip(previous, -1.0, 1.0)
        norm = np.linalg.norm(command, axis=1, keepdims=True)
        command /= np.maximum(1.0, norm)
        if rigid:
            common = formation_safety_filter(
                command[0], env.lons, env.lats, env.dist_interp, 100.0
            )
            command = np.repeat(common[None, :], env.n_auv, axis=0)
        else:
            command = safety_filter(
                command, env.lons, env.lats, env.dist_interp, 100.0
            )
        obs, _reward, _terminated, truncated, info = env.step(command)
        obs = policy_observation(obs, rigid)
        for index in range(env.n_auv):
            trajectories[index].append((env.lons[index], env.lats[index]))
        eyes.append((float(info["eye_lon"]), float(info["eye_lat"])))
        if truncated:
            break
    trajectories = [np.asarray(trajectory) for trajectory in trajectories]
    eyes = np.asarray(eyes)
    path_km = np.mean(
        [
            np.sum(
                np.hypot(
                    np.diff(trajectory[:, 0])
                    * D2KM
                    * np.cos(np.deg2rad(trajectory[:-1, 1])),
                    np.diff(trajectory[:, 1]) * D2KM,
                )
            )
            for trajectory in trajectories
        ]
    )
    rel_x = (
        (env.lons - eyes[-1, 0])
        * D2KM
        * np.cos(np.deg2rad(eyes[-1, 1]))
    )
    rel_y = (env.lats - eyes[-1, 1]) * D2KM
    final_error = float(np.hypot(np.mean(rel_x), np.mean(rel_y)))
    return trajectories, eyes, data, final_error, path_km


def main() -> None:
    unfiltered = rollout(filtered=False)
    filtered = rollout(filtered=True)
    points = []
    for result in (unfiltered, filtered):
        trajectories, eyes, _, _, _ = result
        points.extend([trajectory for trajectory in trajectories])
        points.append(eyes)
    all_points = np.concatenate(points, axis=0)
    lo = np.min(all_points, axis=0) - 0.8
    hi = np.max(all_points, axis=0) + 0.8
    extent = (lo[0], hi[0], lo[1], hi[1])

    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, 5.2), constrained_layout=True
    )
    mesh = None
    for ax, result, title in zip(
        axes,
        (unfiltered, filtered),
        ("Without filtering", rf"With filtering ($\beta={BETA}$)"),
    ):
        trajectories, eyes, data, final_error, path_km = result
        final_time = min(env_time := int(len(eyes) - 1), len(eyes) - 1)
        t_final = int(min(result[1].shape[0] - 1, len(data["track_lon"]) - 1))
        # The eye history runs in simulation steps, while the environmental
        # field is indexed by the corresponding segment offset.
        t_field = min(seg_idx(T0) + final_time, len(data["track_lon"]) - 1)
        mesh = draw_wind(ax, data, t_field, extent)
        ax.plot(
            data["track_lon"][seg_idx(T0):t_field + 1],
            data["track_lat"][seg_idx(T0):t_field + 1],
            color="white",
            lw=1.6,
            ls="--",
            alpha=0.85,
            zorder=3,
        )
        for index, trajectory in enumerate(trajectories):
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color="white",
                lw=2.5,
                alpha=0.58,
                zorder=4,
            )
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=COLORS[index],
                lw=1.3,
                zorder=5,
            )
            ax.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                facecolors="none",
                edgecolors="white",
                s=40,
                linewidths=0.9,
                zorder=7,
            )
            ax.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                color=COLORS[index],
                marker="^",
                s=44,
                edgecolors="white",
                linewidths=0.5,
                zorder=7,
            )
        ax.scatter(
            eyes[-1, 0],
            eyes[-1, 1],
            marker="*",
            s=180,
            color="#E63946",
            edgecolors="white",
            linewidths=0.7,
            zorder=9,
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_box_aspect(0.72)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.text(
            0.02,
            0.03,
            f"Mean path: {path_km:.1f} km\n"
            f"Final tracking error: {final_error:.1f} km",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "0.7",
                "alpha": 0.88,
            },
            zorder=12,
        )
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label("10 m wind speed (m s$^{-1}$)")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="0.35",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="white",
            markersize=6,
            label="Initial position",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="0.35",
            linestyle="None",
            markersize=6,
            label="Final position",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="#E63946",
            linestyle="None",
            markersize=11,
            label="TC core",
        ),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
    )
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGDIR / "Fig6_Filter_Path_Comparison.pdf", bbox_inches="tight"
    )
    fig.savefig(
        FIGDIR / "Fig6_Filter_Path_Comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(FIGDIR / "Fig6_Filter_Path_Comparison.pdf")


if __name__ == "__main__":
    main()
