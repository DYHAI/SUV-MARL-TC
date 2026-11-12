#!/usr/bin/env python3
"""Render a representative ring/line/grid/V formation comparison."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("DEPLOY_OCEAN_ONLY", "1")
os.environ.setdefault("DEPLOY_MIN_COAST_KM", "100")
os.environ.setdefault("SAFETY_MARGIN_KM", "100")
os.environ.setdefault("DYNAMICS_MODE", "constrained")
os.environ.setdefault("DYN_SURFACE_CURRENT", "0")
os.environ.setdefault("OBS_HEADING", "1")
os.environ.setdefault("DYN_SURFACE_ACC_MS2", "0.02")
os.environ.setdefault("DYN_SURFACE_DEC_MS2", "0.03")
os.environ.setdefault("DYN_SURFACE_YAW_DEG_S", "1.0")
os.environ.setdefault("DYN_SURFACE_MIN_TURN_RADIUS_M", "200")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator
from stable_baselines3 import PPO

from code.paths import PAPER_ROOT, REPO_ROOT, RESULTS_ROOT, RUNS_ROOT
from code.surface.eval_surface_robustness import deploy_fixed
from code.surface.fleet_segment_env import D2KM, FleetSegmentEnv, seg_idx


ROOT = REPO_ROOT
OUT = PAPER_ROOT / "figures"
METRICS_OUT = RESULTS_ROOT / "formation" / "representative_metrics.csv"
STORM = "KHANUN_2023"
STORM_T0 = 815
SEED = 917
MODELS = {
    "Ring": RUNS_ROOT / "official" / "models" / "surface_main"
    / "surface_n8_v12_s0.zip",
    "Line": RUNS_ROOT / "formation" / "models" / "surface_line_n8_v12_s0.zip",
    "Grid": RUNS_ROOT / "formation" / "models" / "surface_grid_n8_v12_s0.zip",
    "V": RUNS_ROOT / "formation" / "models" / "surface_v_n8_v12_s0.zip",
}
WIND_CMAP = plt.get_cmap("YlGnBu")


def simulate(formation: str, model_path: Path):
    seg = pd.read_csv(ROOT / "data" / "segments_500.csv")
    row = seg[(seg.storm == STORM) & (seg.t0 == STORM_T0)].iloc[0]
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
    if env.dynamics is not None:
        env.dynamics.reset(seed=SEED)
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
        linewidths=0.35,
        alpha=0.32,
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
    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 9.2), constrained_layout=True
    )
    axes = axes.reshape(-1)
    formation_colors = {
        "Ring": "#2B6CB0",
        "Line": "#C05621",
        "Grid": "#2F855A",
        "V": "#6B46C1",
    }
    metric_rows = []
    results = {
        formation: simulate(formation, MODELS[formation])
        for formation in ("Ring", "Line", "Grid", "V")
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
    for index, (ax, formation) in enumerate(
        zip(axes, ("Ring", "Line", "Grid", "V"))
    ):
        traj, _row, d, final_t = results[formation]
        mesh = draw_wind_background(ax, d, final_t, xlim, ylim)
        mesh.set_alpha(0.80)
        line_color = formation_colors[formation]
        for i in range(traj.shape[1]):
            ax.plot(
                traj[:, i, 0], traj[:, i, 1],
                color="white", lw=2.2, alpha=0.72, zorder=4,
            )
            ax.plot(
                traj[:, i, 0], traj[:, i, 1],
                color=line_color, lw=1.15, alpha=0.82, zorder=5,
            )
            ax.scatter(
                traj[0, i, 0], traj[0, i, 1],
                facecolors="white", edgecolors=line_color,
                marker="o", s=42, linewidth=1.15, zorder=7,
            )
            ax.scatter(
                traj[-1, i, 0], traj[-1, i, 1],
                facecolors=line_color, edgecolors="white",
                marker="^", s=46, linewidth=0.8, zorder=7,
            )
        ax.scatter(
            [0], [0], marker="*", s=175, color="#E63946",
            edgecolors="white", linewidths=0.8, zorder=9,
        )
        ax.axhline(0, color="white", lw=0.55, alpha=0.55, zorder=2)
        ax.axvline(0, color="white", lw=0.55, alpha=0.55, zorder=2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        if index < 2:
            ax.tick_params(labelbottom=False)
        ax.tick_params(direction="out", length=2.8, width=0.7, labelsize=9.5)
        ax.grid(color="white", alpha=0.16, lw=0.45)
        centroid = traj[:, :, :].mean(axis=1)
        radius = np.linalg.norm(traj - centroid[:, None, :], axis=2).mean()
        final_distance = float(np.linalg.norm(centroid[-1]))
        ax.set_title(
            f"{formation} formation\n"
            f"mean radius {radius:.1f} km, final offset {final_distance:.1f} km",
            loc="left", fontsize=11, fontweight="bold", pad=5,
        )
        metric_rows.append(
            {
                "formation": formation,
                "mean_formation_radius_km": float(radius),
                "final_centroid_distance_km": final_distance,
            }
        )
        print(formation, metric_rows[-1], flush=True)
    for ax in axes:
        ax.set_ylabel("TC-relative north (km)")
    handles = [
        Line2D(
            [0], [0], marker="o", color="0.25", linestyle="None",
            markerfacecolor="white", markeredgewidth=1.1,
            markersize=6.5, label="Initial position",
        ),
        Line2D(
            [0], [0], marker="^", color="0.25", linestyle="None",
            markerfacecolor="0.35", markeredgecolor="white",
            markeredgewidth=0.8, markersize=6.5, label="Final position",
        ),
        Line2D(
            [0], [0], marker="*", color="#E63946", linestyle="None",
            markeredgecolor="white", markeredgewidth=0.7,
            markersize=12, label="TC center",
        ),
    ]
    axes[-2].set_xlabel("TC-relative east (km)")
    axes[-1].set_xlabel("TC-relative east (km)")
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        fontsize=9.5,
        handletextpad=0.35,
        columnspacing=1.4,
    )
    cbar = fig.colorbar(
        mesh,
        ax=axes,
        orientation="horizontal",
        fraction=0.035,
        pad=0.035,
        aspect=55,
    )
    cbar.set_label("10 m wind speed (m s$^{-1}$)")
    cbar.set_ticks(np.arange(0.0, 31.0, 10.0))
    cbar.ax.tick_params(labelsize=9.5, length=2.5, width=0.7)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "FigB1_Formation_Comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT / "FigB1_Formation_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(METRICS_OUT, index=False)
    print(OUT / "FigB1_Formation_Comparison.pdf")


if __name__ == "__main__":
    main()
