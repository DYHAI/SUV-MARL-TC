"""Render representative surface and underwater trajectories for the paper."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("FLEET_VMAX", "20")
os.environ.setdefault("DEPLOY_OCEAN_ONLY", "1")
os.environ.setdefault("DEPLOY_MIN_COAST_KM", "100")
os.environ.setdefault("SAFETY_MARGIN_KM", "100")
os.environ.setdefault("SST_MODE", "none")
os.environ.setdefault("USE_MSLP", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from stable_baselines3 import PPO

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fleet_segment_env import (  # noqa: E402
    D2KM,
    FleetSegmentEnv,
    sample_deployment,
    safety_filter,
    seg_idx,
)
from surface_wind_env import land_ocean_mask  # noqa: E402
from underwater_fleet_flow_env import UnderwaterFleetFlowEnv, STEP_MIN  # noqa: E402


DOC = ROOT / "Latex_Yihui_MARL"
FIGDIR = DOC / "figures"
MODELDIR = ROOT / "ppo_fleet"
UW_MODELDIR = ROOT / "ppo_underwater_fleet"
SEGMENTS = ROOT / "data" / "segments_500.csv"
SURFACE_MATRIX_CASE = ("CHAMPI_2015_2015286", 1518)
SURFACE_EVOLUTION_CASE = ("CHAMPI_2015_2015286", 1518)
UNDERWATER_CASE = ("CHAMPI_2015_2015286", 1518)
UNDERWATER_DEPTHS = (0, 30, 50)


def display_storm(storm):
    """Return a publication label without the internal IBTrACS record ID."""
    parts = storm.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return f"{parts[0]} ({parts[1]})"
    return storm
FLEETS = (4, 8, 12, 16)
RADII = (100, 200, 400, 600, 800)
BETA = 0.8

COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
    "#bcbd22", "#7f7f7f", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
)
WIND_CMAP = LinearSegmentedColormap.from_list(
    "smooth_wind",
    ["#f7fbff", "#d7eff8", "#87cee8", "#39a8d0",
     "#66c679", "#f2df66", "#f5a142", "#e56738", "#b91f45", "#6a0d6b"],
    N=256,
)
WIND_CMAP.set_under("#ffffff")
def surface_model(n_auv: int) -> PPO:
    path = MODELDIR / (
        f"ppo_fleet_seg{n_auv}_ring_s005_d800_t10min_n500_dscan_"
        "mthr100_ab_windonly_noid.zip"
    )
    return PPO.load(path, device="cpu")


def surface_rollout(env: FleetSegmentEnv, model: PPO, row, radius: int, storm: str):
    env.d = env.data[storm]
    env.t0 = seg_idx(row.t0)
    env.t1 = seg_idx(row.t1)
    env.t = env.t0
    np.random.seed(17)
    env.lons, env.lats = sample_deployment(
        env.d,
        env.t0,
        env.n_auv,
        radius,
        formation="ring",
        ring_radius=100.0,
        deploy_mode="fixed",
        ocean_only=True,
        min_coast_km=100.0,
        fixed_km=radius,
    )
    env.vxs = np.zeros(env.n_auv)
    env.vys = np.zeros(env.n_auv)
    env.prev_actions = np.zeros((env.n_auv, 2), dtype=np.float32)
    obs = env._obs()
    action_prev = np.zeros((env.n_auv, 2), dtype=np.float32)
    trajectories = [[(env.lons[i], env.lats[i])] for i in range(env.n_auv)]
    eyes = [(float(env.d["track_lon"][env.t]), float(env.d["track_lat"][env.t]))]
    times = [0]
    while True:
        action, _ = model.predict(
            np.asarray(obs, dtype=np.float32).reshape(-1), deterministic=True
        )
        action = np.asarray(action, dtype=np.float32).reshape(env.n_auv, 2)
        action_prev = (1.0 - BETA) * action + BETA * action_prev
        action_use = np.clip(action_prev, -1.0, 1.0)
        norm = np.linalg.norm(action_use, axis=1, keepdims=True)
        action_use /= np.maximum(1.0, norm)
        action_use = safety_filter(
            action_use, env.lons, env.lats, env.dist_interp, 100.0
        )
        obs, _reward, _terminated, truncated, info = env.step(action_use)
        for i in range(env.n_auv):
            trajectories[i].append((env.lons[i], env.lats[i]))
        eyes.append((float(info["eye_lon"]), float(info["eye_lat"])))
        times.append((env.t - env.t0) * STEP_MIN / 60.0)
        if truncated:
            break
    return (
        [np.asarray(t) for t in trajectories],
        np.asarray(eyes),
        np.asarray(times),
    )


def wind_snapshot(d, t):
    h0 = int(d["h0"][t])
    h1 = int(d["h1"][t])
    fr = float(d["frac"][t])
    speed = (1.0 - fr) * d["spd"][h0] + fr * d["spd"][h1]
    return speed


def crop_extent(d, lon_values, lat_values, margin=2.5):
    lo0 = max(float(d["lon"].min()), float(np.min(lon_values) - margin))
    lo1 = min(float(d["lon"].max()), float(np.max(lon_values) + margin))
    la0 = max(float(d["lat"].min()), float(np.min(lat_values) - margin))
    la1 = min(float(d["lat"].max()), float(np.max(lat_values) + margin))
    return lo0, lo1, la0, la1


def add_auv(ax, lon, lat):
    ax.scatter(
        [lon], [lat], s=22, color="#d62728",
        edgecolor="white", linewidth=0.55, zorder=20,
    )


def draw_surface_background(ax, d, t, extent):
    lon, lat = d["lon"], d["lat"]
    speed = wind_snapshot(d, t)
    lo0, lo1, la0, la1 = extent
    jm = (lon >= lo0) & (lon <= lo1)
    im = (lat >= la0) & (lat <= la1)
    j0, j1 = np.where(jm)[0][[0, -1]]
    i0, i1 = np.where(im)[0][[0, -1]]
    lon_c, lat_c = lon[j0:j1 + 1], lat[i0:i1 + 1]
    spd_c = speed[i0:i1 + 1, j0:j1 + 1]
    lon_f = np.linspace(lon_c[0], lon_c[-1], 6 * len(lon_c))
    lat_f = np.linspace(lat_c[0], lat_c[-1], 6 * len(lat_c))
    interp = RegularGridInterpolator(
        (lat_c, lon_c), spd_c, method="cubic",
        bounds_error=False, fill_value=np.nan
    )
    spd_f = interp(np.stack(np.meshgrid(lat_f, lon_f, indexing="ij"), axis=-1))
    spd_f = np.clip(spd_f, 0.0, 30.0)
    mesh = ax.pcolormesh(
        lon_f, lat_f, spd_f, cmap=WIND_CMAP,
        norm=mcolors.Normalize(0, 30), shading="gouraud",
        rasterized=True, zorder=0,
    )
    ax.contour(
        lon_f, lat_f, spd_f, levels=np.arange(5, 31, 5),
        colors="white", linewidths=0.45, alpha=0.55, zorder=1,
    )
    return mesh


def make_surface_matrix(row, storm):
    models = {n: surface_model(n) for n in FLEETS}
    envs = {
        n: FleetSegmentEnv(
            storms=[storm],
            segments_df=pd.DataFrame([row]),
            use_uv=True,
            n_auv=n,
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
        for n in FLEETS
    }
    results = {}
    for n in FLEETS:
        for radius in RADII:
            results[(n, radius)] = surface_rollout(envs[n], models[n], row, radius, storm)

    fig, axes = plt.subplots(
        len(RADII), len(FLEETS), figsize=(15.0, 16.5),
        sharex="row", sharey="row", layout="constrained",
    )
    d = envs[4].data[storm]
    mesh = None
    for r, radius in enumerate(RADII):
        lon_values, lat_values = [], []
        for n in FLEETS:
            trajectories, eyes, _ = results[(n, radius)]
            for tr in trajectories:
                lon_values.extend(tr[:, 0])
                lat_values.extend(tr[:, 1])
            lon_values.extend(eyes[:, 0])
            lat_values.extend(eyes[:, 1])
        extent = crop_extent(d, np.asarray(lon_values), np.asarray(lat_values))
        for c, n in enumerate(FLEETS):
            ax = axes[r, c]
            trajectories, eyes, _ = results[(n, radius)]
            mesh = draw_surface_background(ax, d, envs[n].t1, extent)
            ax.plot(
                d["track_lon"][envs[n].t0:envs[n].t1 + 1],
                d["track_lat"][envs[n].t0:envs[n].t1 + 1],
                color="white", lw=1.5, ls="--", alpha=0.9, zorder=3,
            )
            for i, tr in enumerate(trajectories):
                color = COLORS[i]
                ax.plot(tr[:, 0], tr[:, 1], color=color, lw=1.15, alpha=0.9, zorder=5)
                ax.plot(tr[0, 0], tr[0, 1], "o", ms=3.0, color="white",
                        mec="#d62728", mew=0.8, zorder=8)
                add_auv(ax, tr[-1, 0], tr[-1, 1])
            ax.plot(eyes[-1, 0], eyes[-1, 1], "*", color="#FFD400", ms=12,
                    mec="black", mew=0.7, zorder=15)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_box_aspect(0.72)
            if r == 0:
                ax.set_title(f"{n}-SUV", fontsize=12, fontweight="bold")
            if c == 0:
                ax.text(
                    0.02, 0.97, f"$D_0$={radius} km",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=9, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.72, pad=1.2),
                )
            if r == len(RADII) - 1:
                ax.set_xlabel("Longitude (°E)")
            if c == 0:
                ax.set_ylabel("Latitude (°N)")
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.62, pad=0.018)
    cbar.set_label("10 m wind speed (m s$^{-1}$)")
    fig.suptitle(
        "Representative surface trajectories for 4/8/12/16 SUVs at five "
        r"deployment radii ($V_{\max}$=20 m/s)",
        fontsize=15, fontweight="bold",
    )
    out = FIGDIR / "surface_trajectory_matrix_20ms"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=280, bbox_inches="tight")
    plt.close(fig)
    return results


def make_surface_snapshots(row, storm):
    env = FleetSegmentEnv(
        storms=[storm], segments_df=pd.DataFrame([row]), use_uv=True,
        n_auv=4, smooth_penalty=0.05, spacing_deg=0.25, patch_n=1,
        deploy_radius=800.0, formation="ring", ring_radius=100.0,
        sst_mode="none", use_mslp=False, vmax=20.0,
    )
    d = env.data[storm]
    model = surface_model(4)
    trajectories, eyes, times = surface_rollout(
        env, model, row, 400, storm
    )
    idx = np.unique(np.linspace(0, len(times) - 1, 4).round().astype(int))
    while len(idx) < 4:
        idx = np.unique(np.linspace(0, len(times) - 1, len(idx) + 1).round().astype(int))
    lon_all = np.concatenate([tr[:, 0] for tr in trajectories] + [eyes[:, 0]])
    lat_all = np.concatenate([tr[:, 1] for tr in trajectories] + [eyes[:, 1]])
    extent = crop_extent(d, lon_all, lat_all)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), layout="constrained")
    mesh = None
    for ax, k in zip(axes.ravel(), idx):
        t = min(env.t0 + k, env.t1)
        mesh = draw_surface_background(ax, d, t, extent)
        ax.plot(
            d["track_lon"][env.t0:t + 1], d["track_lat"][env.t0:t + 1],
            color="white", lw=1.7, ls="--", alpha=0.9, zorder=3,
        )
        for i, tr in enumerate(trajectories):
            kk = min(k + 1, len(tr))
            ax.plot(tr[:kk, 0], tr[:kk, 1], color=COLORS[i], lw=1.35, zorder=5)
            add_auv(ax, tr[kk - 1, 0], tr[kk - 1, 1])
        ax.plot(eyes[k, 0], eyes[k, 1], "*", color="#FFD400", ms=15,
                mec="black", mew=0.7, zorder=15)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_box_aspect(0.72)
        ax.set_title(f"t = {times[k]:.1f} h", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.60, pad=0.02)
    cbar.set_label("10 m wind speed (m s$^{-1}$)")
    fig.suptitle(
        "Four-SUV surface tracking from a 400 km deployment radius",
        fontsize=14, fontweight="bold",
    )
    out = FIGDIR / "surface_trajectory_4suv_400km_20ms"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def underwater_rollout(env, policy, seed=101):
    np.random.seed(seed)
    obs, _ = env.reset()
    # Use a common, evenly spaced four-vehicle ring around the segment-end
    # anchor for all three strategies. This preserves the original end-anchor
    # timing while removing deployment randomness from the illustrative figure.
    clon = float(env.d["track_lon"][env.t])
    clat = float(env.d["track_lat"][env.t])
    theta0 = np.deg2rad(45.0)
    for i in range(env.n_auv):
        theta = theta0 + i * 2.0 * np.pi / env.n_auv
        env.lons[i] = clon + 200.0 * np.cos(theta) / (
            D2KM * np.cos(np.deg2rad(clat))
        )
        env.lats[i] = clat + 200.0 * np.sin(theta) / D2KM
    env.lons = np.clip(env.lons, env.d["lon"].min(), env.d["lon"].max())
    env.lats = np.clip(env.lats, env.d["lat"].min(), env.d["lat"].max())
    env.vxs = np.zeros(env.n_auv)
    env.vys = np.zeros(env.n_auv)
    env.flow_hist = [[] for _ in range(env.n_auv)]
    for i in range(env.n_auv):
        env.flow_hist[i].append(
            np.array(
                [
                    env._at2("spd", env.lats[i], env.lons[i]),
                    env._at2("u", env.lats[i], env.lons[i]),
                    env._at2("v", env.lats[i], env.lons[i]),
                ],
                dtype=np.float32,
            )
        )
    obs = env._obs()
    trajs = [[(env.lons[i], env.lats[i])] for i in range(env.n_auv)]
    speeds = []
    while True:
        action = policy(obs, env)
        obs, _reward, terminated, truncated, info = env.step(action)
        speeds.append(np.asarray(info["spd_at"], dtype=float))
        for i in range(env.n_auv):
            trajs[i].append((env.lons[i], env.lats[i]))
        if terminated or truncated:
            break
    return [np.asarray(t) for t in trajs], np.asarray(speeds), env


def draw_current_background(ax, d, speed, extent, vmax):
    lon, lat = d["lon"], d["lat"]
    lo0, lo1, la0, la1 = extent
    jm = (lon >= lo0) & (lon <= lo1)
    im = (lat >= la0) & (lat <= la1)
    j0, j1 = np.where(jm)[0][[0, -1]]
    i0, i1 = np.where(im)[0][[0, -1]]
    lon_c, lat_c = lon[j0:j1 + 1], lat[i0:i1 + 1]
    spd_c = speed[i0:i1 + 1, j0:j1 + 1]
    land_c = d["dist_to_land"][i0:i1 + 1, j0:j1 + 1]
    lon_f = np.linspace(lon_c[0], lon_c[-1], 6 * len(lon_c))
    lat_f = np.linspace(lat_c[0], lat_c[-1], 6 * len(lat_c))
    points = np.stack(np.meshgrid(lat_f, lon_f, indexing="ij"), axis=-1)
    spd_interp = RegularGridInterpolator(
        (lat_c, lon_c), spd_c, method="cubic",
        bounds_error=False, fill_value=np.nan
    )
    land_interp = RegularGridInterpolator(
        (lat_c, lon_c), land_c, bounds_error=False, fill_value=0.0
    )
    spd_f = spd_interp(points)
    spd_f = gaussian_filter(np.nan_to_num(spd_f, nan=0.0), sigma=0.9)
    ocean_f = land_interp(points) > 0.0
    spd_f = np.where(ocean_f, spd_f, np.nan)
    mesh = ax.pcolormesh(
        lon_f, lat_f, spd_f, cmap="viridis",
        norm=mcolors.Normalize(0, vmax), shading="gouraud",
        rasterized=True, zorder=0,
    )
    ax.contour(
        lon_f, lat_f, spd_f,
        levels=np.linspace(0.1 * vmax, 0.9 * vmax, 5),
        colors="white", linewidths=0.45, alpha=0.5, zorder=1,
    )
    return mesh


def make_underwater_comparison(row, storm):
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    depth_results = {}
    common_env = None
    common_extent = None
    for depth in UNDERWATER_DEPTHS:
        model = PPO.load(
            UW_MODELDIR
            / f"ppo_uwflow_500_d{depth}_v2.0_rC_20h_sb3_s0.zip",
            device="cpu",
        )

        def active(obs, env, model=model):
            action, _ = model.predict(
                np.asarray(obs, dtype=np.float32).reshape(-1), deterministic=True
            )
            return np.asarray(action, dtype=np.float32).reshape(env.n_auv, 2)

        policies = (
            ("Active IPPO", active),
            ("Random control",
             lambda o, e: np.random.uniform(-1, 1, (e.n_auv, 2))),
            ("Passive drift",
             lambda o, e: np.zeros((e.n_auv, 2), dtype=np.float32)),
        )
        rows = []
        for label, policy in policies:
            env = UnderwaterFleetFlowEnv(
                storms=[storm],
                segments_df=pd.DataFrame([row]),
                vmax=2.0,
                reward_mode="C",
                N=1,
                use_frame_stack=True,
                frame_k=5,
                start_radius=200.0,
                crop_margin=3.0,
                depth_m=depth,
                grid_deg=0.25,
                duration_h=20.0,
            )
            trajs, speeds, env = underwater_rollout(env, policy)
            rows.append((label, trajs, speeds, env))
            common_env = common_env or env
            lon_all = np.concatenate([tr[:, 0] for tr in trajs])
            lat_all = np.concatenate([tr[:, 1] for tr in trajs])
            local_extent = crop_extent(common_env.d, lon_all, lat_all, margin=0.8)
            if common_extent is None:
                common_extent = local_extent
            else:
                common_extent = (
                    min(common_extent[0], local_extent[0]),
                    max(common_extent[1], local_extent[1]),
                    min(common_extent[2], local_extent[2]),
                    max(common_extent[3], local_extent[3]),
                )
        depth_results[depth] = rows

    d = common_env.d
    fig = plt.figure(figsize=(15.5, 4.7 * len(UNDERWATER_DEPTHS)),
                     layout="constrained")
    grid = fig.add_gridspec(
        len(UNDERWATER_DEPTHS), 4,
        width_ratios=(1.0, 1.0, 1.0, 0.055),
        wspace=0.08, hspace=0.20,
    )
    for r, depth in enumerate(UNDERWATER_DEPTHS):
        vmax = max(
            float(np.nanpercentile(envx.M_mean, 99))
            for _, _, _, envx in depth_results[depth]
        )
        row_mesh = None
        for c, (label, trajs, speeds, envx) in enumerate(depth_results[depth]):
            ax = fig.add_subplot(grid[r, c])
            row_mesh = draw_current_background(
                ax, d, envx.M_mean, common_extent, vmax
            )
            ax.plot(d["track_lon"], d["track_lat"], color="white",
                    lw=1.2, ls="--", alpha=0.8, zorder=3)
            t_start = int(np.argmin(np.hypot(
                (d["track_lon"] - row["end_lon"])
                * D2KM * np.cos(np.deg2rad(row["end_lat"])),
                (d["track_lat"] - row["end_lat"]) * D2KM,
            )))
            center_lon = float(d["track_lon"][t_start])
            center_lat = float(d["track_lat"][t_start])
            theta = np.linspace(0.0, 2.0 * np.pi, 181)
            ring_lon = center_lon + 200.0 * np.cos(theta) / (
                D2KM * np.cos(np.deg2rad(center_lat))
            )
            ring_lat = center_lat + 200.0 * np.sin(theta) / D2KM
            ax.plot(ring_lon, ring_lat, color="white", lw=0.8,
                    ls=":", alpha=0.75, zorder=2)
            for i, tr in enumerate(trajs):
                ax.plot(tr[:, 0], tr[:, 1], color=COLORS[i], lw=1.5, zorder=5)
                ax.plot(tr[0, 0], tr[0, 1], "o", ms=4, color="white",
                        mec="#d62728", mew=1.0, zorder=8)
                add_auv(ax, tr[-1, 0], tr[-1, 1])
            path = np.mean([
                np.hypot(
                    np.diff(tr[:, 0]) * D2KM * np.cos(np.deg2rad(tr[:-1, 1])),
                    np.diff(tr[:, 1]) * D2KM,
                ).sum()
                for tr in trajs
            ])
            ax.set_xlim(common_extent[0], common_extent[1])
            ax.set_ylim(common_extent[2], common_extent[3])
            ax.set_box_aspect(0.72)
            ax.set_title(
                f"{label}\nmean |V|={speeds.mean():.2f} m/s, path={path:.1f} km",
                fontsize=14,
            )
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            if c == 0:
                ax.text(
                    -0.22, 0.5, f"{depth} m",
                    transform=ax.transAxes, rotation=90,
                    ha="center", va="center", fontsize=16, fontweight="bold",
                )
        cax = fig.add_subplot(grid[r, 3])
        cbar = fig.colorbar(row_mesh, cax=cax)
        cbar.set_label(f"{depth} m current speed (m s$^{{-1}}$)")
        cbar.ax.tick_params(labelsize=11)

    fig.suptitle(
        "Representative four-SUV subsurface strategies at 0, 30, and 50 m",
        fontsize=18, fontweight="bold",
    )
    out = FIGDIR / "underwater_strategy_trajectories_multidepth"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    segments = pd.read_csv(SEGMENTS)
    matrix_storm, matrix_t0 = SURFACE_MATRIX_CASE
    matrix_row = segments[
        (segments.storm == matrix_storm)
        & (segments.t0 == matrix_t0)
    ].iloc[0]
    evolution_storm, evolution_t0 = SURFACE_EVOLUTION_CASE
    evolution_row = segments[
        (segments.storm == evolution_storm)
        & (segments.t0 == evolution_t0)
    ].iloc[0]
    underwater_storm, underwater_t0 = UNDERWATER_CASE
    underwater_row = segments[
        (segments.storm == underwater_storm)
        & (segments.t0 == underwater_t0)
    ].iloc[0]
    FIGDIR.mkdir(parents=True, exist_ok=True)
    if os.environ.get("UNDERWATER_ONLY", "0") == "1":
        make_underwater_comparison(underwater_row, underwater_storm)
        return
    if os.environ.get("SURFACE_ONLY", "0") == "1":
        make_surface_snapshots(evolution_row, evolution_storm)
        return
    make_surface_matrix(matrix_row, matrix_storm)
    make_surface_snapshots(evolution_row, evolution_storm)
    make_underwater_comparison(underwater_row, underwater_storm)


if __name__ == "__main__":
    main()
