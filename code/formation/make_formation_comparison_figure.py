#!/usr/bin/env python3
"""Render a representative ring/line/grid formation comparison."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("DEPLOY_OCEAN_ONLY", "1")
os.environ.setdefault("DEPLOY_MIN_COAST_KM", "100")
os.environ.setdefault("SAFETY_MARGIN_KM", "100")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator
from stable_baselines3 import PPO

from code.surface.eval_surface_robustness import deploy_fixed
from code.surface.fleet_segment_env import D2KM, FleetSegmentEnv, seg_idx
from code.paths import PAPER_ROOT, REPO_ROOT, RESULTS_ROOT, RUNS_ROOT


ROOT = REPO_ROOT
OUT = PAPER_ROOT / "figures"
STORM = "CHAMPI_2015_2015286"
SEED = 917
MODELS = {
    "Ring": RUNS_ROOT
    / "surface"
    / "models"
    / "ppo_fleet_seg8_ring_s005_d800_t10min_n500_dscan_mthr100_ab_windonly_noid_vmax12_s0.zip",
    "Line": RUNS_ROOT / "formation" / "models" / "surface_line_n8_v12_s0.zip",
    "Grid": RUNS_ROOT / "formation" / "models" / "surface_grid_n8_v12_s0.zip",
}
WIND_CMAP = plt.get_cmap("YlGnBu")


def simulate(formation: str, model_path: Path):
    seg = pd.read_csv(ROOT / "data" / "segments_500.csv")
    row = seg[seg.storm == STORM].iloc[0]
    env = FleetSegmentEnv(
        storms=[STORM],
        segments_df=pd.DataFrame([row]),
        use_uv=True,
        n_auv=8,
        smooth_penalty=0.05,
        spacing_deg=0.25,
        patch_n=1,
        deploy_radius=800.0,
        formation=formation.lower(),
        ring_radius=100.0,
        sst_mode="none",
        use_mslp=False,
        vmax=12.0,
    )
    d = env.data[STORM]
    env.d = d
    env.t0 = seg_idx(row["t0"])
    env.t1 = seg_idx(row["t1"])
    env.t = env.t0
    env.vxs = np.zeros(env.n_auv, dtype=np.float32)
    env.vys = np.zeros(env.n_auv, dtype=np.float32)
    env.lons, env.lats = deploy_fixed(
        d, env.t0, env.n_auv, 400.0, SEED, formation.lower()
    )
    env.prev_actions = np.zeros((env.n_auv, 2), dtype=np.float32)
    model = PPO.load(model_path, device="cpu")
    obs = env._obs()
    executed = np.zeros((env.n_auv, 2), dtype=np.float32)
    traj = []
    while True:
        raw, _ = model.predict(obs.reshape(-1), deterministic=True)
        raw = np.asarray(raw, dtype=np.float32).reshape(env.n_auv, 2)
        executed = 0.2 * raw + 0.8 * executed
        executed /= np.maximum(
            1.0, np.linalg.norm(executed, axis=1, keepdims=True)
        )
        obs, _reward, terminated, truncated, info = env.step(executed)
        eye_lon = float(info["eye_lon"])
        eye_lat = float(info["eye_lat"])
        x = (env.lons - eye_lon) * D2KM * np.cos(np.deg2rad(eye_lat))
        y = (env.lats - eye_lat) * D2KM
        traj.append(np.stack([x, y], axis=1))
        if terminated or truncated:
            break
    return np.asarray(traj), row, d, int(env.t)


def wind_snapshot(d: dict, t: int) -> np.ndarray:
    """Linearly interpolate the ERA5 wind-speed field to simulation time t."""
    h0, h1 = int(d["h0"][t]), int(d["h1"][t])
    frac = float(d["frac"][t])
    return (1.0 - frac) * d["spd"][h0] + frac * d["spd"][h1]


def draw_wind_background(ax, d: dict, t: int, xlim, ylim):
    """Draw final-time ERA5 wind speed in TC-relative km coordinates."""
    center_lon = float(d["track_lon"][t])
    center_lat = float(d["track_lat"][t])
    xe = np.linspace(xlim[0], xlim[1], 280)
    yn = np.linspace(ylim[0], ylim[1], 280)
    X, Y = np.meshgrid(xe, yn)
    lon = center_lon + X / (D2KM * np.cos(np.deg2rad(center_lat)))
    lat = center_lat + Y / D2KM
    interpolator = RegularGridInterpolator(
        (d["lat"], d["lon"]),
        wind_snapshot(d, t),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    speed = interpolator(np.stack([lat, lon], axis=-1))
    mesh = ax.pcolormesh(
        xe,
        yn,
        speed,
        cmap=WIND_CMAP,
        norm=mcolors.Normalize(0.0, 30.0),
        shading="gouraud",
        rasterized=True,
        zorder=0,
    )
    ax.contour(
        xe,
        yn,
        speed,
        levels=np.arange(5.0, 31.0, 5.0),
        colors="white",
        linewidths=0.45,
        alpha=0.55,
        zorder=1,
    )
    return mesh


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.6), constrained_layout=True)
    colors = (
        "#4E79A7", "#F28E2B", "#59A14F", "#E15759",
        "#B07AA1", "#76B7B2", "#EDC948", "#9C755F",
    )
    metric_rows = []
    results = {
        formation: simulate(formation, MODELS[formation])
        for formation in ("Ring", "Line", "Grid")
    }
    all_traj = np.concatenate(
        [results[key][0].reshape(-1, 2) for key in results],
        axis=0,
    )
    xmin, ymin = np.min(all_traj, axis=0)
    xmax, ymax = np.max(all_traj, axis=0)
    xpad = max(20.0, 0.08 * (xmax - xmin))
    ypad = max(20.0, 0.08 * (ymax - ymin))
    xlim = (xmin - xpad, xmax + xpad)
    ylim = (ymin - ypad, ymax + ypad)
    mesh = None
    for ax, formation in zip(axes, ("Ring", "Line", "Grid")):
        traj, _row, d, final_t = results[formation]
        mesh = draw_wind_background(ax, d, final_t, xlim, ylim)
        for i in range(traj.shape[1]):
            ax.plot(
                traj[:, i, 0], traj[:, i, 1],
                color="white", lw=2.4, alpha=0.55, zorder=4,
            )
            ax.plot(
                traj[:, i, 0], traj[:, i, 1],
                color=colors[i], lw=1.35, alpha=0.85,
            )
            ax.scatter(
                traj[0, i, 0], traj[0, i, 1],
                facecolors="none", edgecolors=colors[i], marker="o", s=36,
            )
            ax.scatter(
                traj[-1, i, 0], traj[-1, i, 1],
                color=colors[i], marker="^", s=36,
            )
        ax.scatter([0], [0], marker="*", s=130, color="#E63946", zorder=5)
        ax.axhline(0, color="0.8", lw=0.6)
        ax.axvline(0, color="0.8", lw=0.6)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("TC-relative east (km)")
        ax.set_title(f"{formation} formation")
        ax.grid(alpha=0.2)
        centroid = traj[:, :, :].mean(axis=1)
        radius = np.linalg.norm(traj - centroid[:, None, :], axis=2).mean()
        metric_rows.append(
            {
                "formation": formation,
                "mean_formation_radius_km": float(radius),
                "final_centroid_distance_km": float(np.linalg.norm(centroid[-1])),
            }
        )
        print(formation, metric_rows[-1], flush=True)
    for ax in axes:
        ax.set_ylabel("TC-relative north (km)")
    handles = [
        Line2D([0], [0], marker="o", color="0.4", linestyle="None",
               markerfacecolor="none", label="initial"),
        Line2D([0], [0], marker="^", color="0.4", linestyle="None",
               label="final"),
    ]
    axes[0].legend(handles=handles, frameon=False, loc="best", fontsize=8.5)
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.72, pad=0.018)
    cbar.set_label("10 m wind speed (m s$^{-1}$)")
    cbar.ax.tick_params(labelsize=9.5)
    fig.suptitle(
        "Eight-SUV surface formation comparison (CHAMPI 2015, 12 m/s)",
        fontsize=12,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "formation_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT / "formation_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(metric_rows).to_csv(
        RESULTS_ROOT / "formation" / "formation_representative.csv", index=False
    )
    print(OUT / "formation_comparison.pdf")


if __name__ == "__main__":
    main()
