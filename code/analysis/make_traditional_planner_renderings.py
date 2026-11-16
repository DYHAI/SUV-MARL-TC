#!/usr/bin/env python3
"""Render matched surface and subsurface classical-planner comparisons."""

from __future__ import annotations

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from code.baselines.classical_planning_baselines import make_planner
from code.surface.eval_surface_robustness import deploy_fixed
from code.surface.fleet_segment_env import (
    D2KM,
    FleetSegmentEnv,
    coerce_formation_action,
    formation_safety_filter,
    policy_observation,
    rigid_formation_enabled,
    safety_filter,
    seg_idx as surface_seg_idx,
)
from code.underwater.underwater_fleet_flow_env import UnderwaterFleetFlowEnv
from code.underwater.underwater_coverage import (
    native_hycom_cells,
    native_hycom_coverage_area,
    native_hycom_weighted_coverage,
)
from code.underwater.underwater_flow_env import STEP_MIN as UNDERWATER_STEP_MIN

DOC = ROOT / "paper"
OUT = DOC / "figures" / "FigB1_Traditional_Planner_Rendering"
SURFACE_MODEL = Path(
    os.environ.get(
        "SURFACE_MODEL_DIR",
        ROOT
        / "constrained_dynamics_experiments"
        / "models"
        / "surface_primary_circle",
    )
) / "surface_n8_v12_s0.zip"
UNDERWATER_MODEL = (
    ROOT
    / "constrained_dynamics_experiments"
    / "models"
    / "underwater_primary"
    / "uw_d30_v1.5_s0.zip"
)
STORM = "MATMO_2025"
STORM_T0 = 561
SEED = 917
DEPLOY_RADIUS_KM = 400.0
DEPTH_M = 30.0
METHODS = (
    ("IPPO", "ippo"),
    ("Potential field", "potential"),
    ("Receding-horizon A*", "astar"),
    ("Lawnmower", "coverage"),
    ("Information greedy", "greedy"),
)
COLORS = {
    "IPPO": "#1f77b4",
    "Potential field": "#d95f02",
    "Receding-horizon A*": "#2ca02c",
    "Lawnmower": "#9467bd",
    "Information greedy": "#8c564b",
}


def representative_segment() -> pd.Series:
    seg = pd.read_csv(ROOT / "data" / "segments_primary_400.csv")
    candidates = seg[
        (seg.storm == STORM) & (seg.t0 == STORM_T0)
    ].reset_index(drop=True)
    if candidates.empty:
        raise ValueError(f"no representative segment for {STORM}")
    return candidates.iloc[0]


def surface_rollout(method: str, row: pd.Series, model: PPO):
    env = FleetSegmentEnv(
        storms=[STORM],
        segments_df=pd.DataFrame([row]),
        use_uv=True,
        n_auv=8,
        smooth_penalty=0.05,
        spacing_deg=0.25,
        patch_n=1,
        deploy_radius=800.0,
        formation="ring",
        ring_radius=100.0,
        sst_mode="none",
        use_mslp=False,
        vmax=12.0,
    )
    d = env.data[STORM]
    env.d = d
    env.t0 = surface_seg_idx(row["t0"])
    env.t1 = surface_seg_idx(row["t1"])
    env.t = env.t0
    env.vxs = np.zeros(env.n_auv, dtype=np.float32)
    env.vys = np.zeros(env.n_auv, dtype=np.float32)
    if env.dynamics is not None:
        env.dynamics.reset(seed=SEED)
    env.lons, env.lats = deploy_fixed(
        d, env.t0, env.n_auv, DEPLOY_RADIUS_KM, SEED, "ring"
    )
    env.prev_actions = np.zeros((env.n_auv, 2), dtype=np.float32)
    planner = None if method == "ippo" else make_planner(method, "surface")
    if planner is not None:
        planner.reset(env)
    rigid = rigid_formation_enabled()
    obs = policy_observation(env._obs(), rigid)
    executed = np.zeros((env.n_auv, 2), dtype=np.float32)
    lats, lons, errors = [env.lats.copy()], [env.lons.copy()], []
    while True:
        if method == "ippo":
            raw, _ = model.predict(obs, deterministic=True)
        else:
            planner_obs = env._obs().reshape(-1).astype(np.float32)
            raw = planner.act(planner_obs, env)
        if rigid:
            raw = coerce_formation_action(raw, env.n_auv)
        else:
            raw = np.asarray(raw, dtype=np.float32).reshape(env.n_auv, 2)
        executed = 0.2 * raw + 0.8 * executed
        action = executed / np.maximum(
            1.0, np.linalg.norm(executed, axis=1, keepdims=True)
        )
        if rigid:
            common = formation_safety_filter(
                action[0], env.lons, env.lats, env.dist_interp, 100.0
            )
            action = np.repeat(common[None, :], env.n_auv, axis=0)
        else:
            action = safety_filter(
                action, env.lons, env.lats, env.dist_interp, 100.0
            )
        obs, _reward, terminated, truncated, info = env.step(action)
        obs = policy_observation(obs, rigid)
        lats.append(env.lats.copy())
        lons.append(env.lons.copy())
        rel_x = (
            (env.lons - float(info["eye_lon"]))
            * D2KM
            * np.cos(np.deg2rad(float(info["eye_lat"])))
        )
        rel_y = (env.lats - float(info["eye_lat"])) * D2KM
        errors.append(float(np.hypot(np.mean(rel_x), np.mean(rel_y))))
        if terminated or truncated:
            break
    final_t = int(env.t)
    return (
        np.asarray(lats),
        np.asarray(lons),
        float(errors[-1]),
        d,
        final_t,
    )


def underwater_rollout(method: str, row: pd.Series, model: PPO):
    env = UnderwaterFleetFlowEnv(
        storms=[STORM],
        segments_df=pd.DataFrame([row]),
        vmax=1.5,
        reward_mode="C",
        N=1,
        use_frame_stack=True,
        frame_k=5,
        start_radius=200.0,
        crop_margin=3.0,
        depth_m=DEPTH_M,
        grid_deg=0.25,
        duration_h=20.0,
    )
    obs, _ = env.reset(seed=SEED)
    planner = None if method == "ippo" else make_planner(method, "subsurface")
    if planner is not None:
        planner.reset(env)
    lats, lons, current = [env.lats.copy()], [env.lons.copy()], []
    while True:
        if method == "ippo":
            raw, _ = model.predict(obs.reshape(-1), deterministic=True)
            action = np.asarray(raw, dtype=np.float32).reshape(env.n_auv, 2)
        else:
            action = np.asarray(planner.act(obs, env), dtype=np.float32).reshape(
                env.n_auv, 2
            )
        action /= np.maximum(1.0, np.linalg.norm(action, axis=1, keepdims=True))
        obs, _reward, terminated, truncated, info = env.step(action)
        lats.append(env.lats.copy())
        lons.append(env.lons.copy())
        current.append(np.asarray(info["spd_at"], dtype=np.float64))
        if terminated or truncated:
            break
    final_t = int(env.t)
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    mean_current = float(np.mean(current))
    sampled_lats = lats[:-1]
    sampled_lons = lons[:-1]
    current = np.asarray(current, dtype=np.float64)
    cells = native_hycom_cells(sampled_lats.ravel(), sampled_lons.ravel())
    coverage_area_km2 = native_hycom_coverage_area(cells)
    current_weighted_coverage = native_hycom_weighted_coverage(
        sampled_lats.ravel(),
        sampled_lons.ravel(),
        current.ravel(),
    )
    return (
        lats,
        lons,
        mean_current,
        current_weighted_coverage,
        env,
        final_t,
    )


def relative_km(lats, lons, eye_lon, eye_lat):
    x = (lons - eye_lon) * D2KM * np.cos(np.deg2rad(eye_lat))
    y = (lats - eye_lat) * D2KM
    return np.stack([x, y], axis=-1)


def field_at(env, t: int, key: str = "spd"):
    d = env.d
    h0, h1 = int(d["h0"][t]), int(d["h1"][t])
    frac = float(d["frac"][t])
    return (1.0 - frac) * d[key][h0] + frac * d[key][h1]


def sample_field(env, t: int, xlim, ylim):
    center_lon = float(env.d["track_lon"][t])
    center_lat = float(env.d["track_lat"][t])
    xe = np.linspace(xlim[0], xlim[1], 260)
    yn = np.linspace(ylim[0], ylim[1], 260)
    X, Y = np.meshgrid(xe, yn)
    lon = center_lon + X / (D2KM * np.cos(np.deg2rad(center_lat)))
    lat = center_lat + Y / D2KM
    field = field_at(env, t)
    interp = RegularGridInterpolator(
        (env.d["lat"], env.d["lon"]),
        field,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return xe, yn, interp(np.stack([lat, lon], axis=-1))


def draw_field(ax, env, t: int, xlim, ylim, label: str, norm):
    xe, yn, values = sample_field(env, t, xlim, ylim)
    mesh = ax.pcolormesh(
        xe,
        yn,
        values,
        shading="gouraud",
        cmap="YlGnBu",
        norm=norm,
        rasterized=True,
    )
    ax.contour(
        xe,
        yn,
        values,
        levels=5,
        colors="white",
        linewidths=0.35,
        alpha=0.45,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="white", alpha=0.14, linewidth=0.4)
    ax.set_xlabel(label)
    return mesh


def plot_row(
    axs,
    row_label: str,
    results,
    field_label: str,
    hide_upper_x: bool = False,
    panel_offset: int = 0,
    *,
    pad_fraction: float = 0.18,
    min_pad_km: float = 60.0,
    symmetric_about_tc: bool = False,
    min_half_span_km: float | None = None,
    min_width_km: float | None = None,
    min_height_km: float | None = None,
):
    all_xy = []
    for _name, xy, *_rest in results:
        all_xy.append(xy.reshape(-1, 2))
    all_xy = np.concatenate(all_xy, axis=0)
    xmin, ymin = np.nanmin(all_xy, axis=0)
    xmax, ymax = np.nanmax(all_xy, axis=0)
    # Every task panel must retain the TC core as spatial context, even when
    # paths are displaced far downstream by the environmental field.
    xmin, xmax = min(float(xmin), 0.0), max(float(xmax), 0.0)
    ymin, ymax = min(float(ymin), 0.0), max(float(ymax), 0.0)
    pad = max(
        float(min_pad_km),
        float(pad_fraction) * max(xmax - xmin, ymax - ymin),
    )
    if symmetric_about_tc:
        half_span = max(
            abs(xmin),
            abs(xmax),
            abs(ymin),
            abs(ymax),
        ) + pad
        if min_half_span_km is not None:
            half_span = max(half_span, float(min_half_span_km))
        xlim = (-half_span, half_span)
        ylim = (-half_span, half_span)
    else:
        xlim = (xmin - pad, xmax + pad)
        ylim = (ymin - pad, ymax + pad)
        if min_width_km is not None and xlim[1] - xlim[0] < min_width_km:
            center = 0.5 * (xlim[0] + xlim[1])
            half = 0.5 * float(min_width_km)
            xlim = (center - half, center + half)
        if min_height_km is not None and ylim[1] - ylim[0] < min_height_km:
            center = 0.5 * (ylim[0] + ylim[1])
            half = 0.5 * float(min_height_km)
            ylim = (center - half, center + half)
    field_values = [
        sample_field(result[3], result[4], xlim, ylim)[2]
        for result in results
    ]
    finite_values = np.concatenate(
        [values[np.isfinite(values)] for values in field_values]
    )
    norm = mcolors.Normalize(
        0.0,
        float(np.nanpercentile(finite_values, 99.0)),
    )
    mesh = None
    for index, (ax, result) in enumerate(zip(axs, results)):
        name, xy, score_text, env, t = result
        mesh = draw_field(
            ax,
            env,
            t,
            xlim,
            ylim,
            "TC-relative east (km)",
            norm,
        )
        for i in range(xy.shape[1]):
            ax.plot(
                xy[:, i, 0],
                xy[:, i, 1],
                color=COLORS[name],
                linewidth=1.35,
                alpha=0.94,
                solid_capstyle="round",
                zorder=4,
            )
            if xy.shape[0] >= 5:
                arrow_index = max(1, min(xy.shape[0] - 2, int(0.56 * xy.shape[0])))
                ax.annotate(
                    "",
                    xy=xy[arrow_index + 1, i],
                    xytext=xy[arrow_index, i],
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=COLORS[name],
                        lw=1.2,
                        mutation_scale=9,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=5,
                )
            ax.scatter(
                xy[0, i, 0],
                xy[0, i, 1],
                facecolor="white",
                edgecolor=COLORS[name],
                marker="o",
                s=40,
                linewidth=1.0,
                zorder=5,
            )
            ax.scatter(
                xy[-1, i, 0],
                xy[-1, i, 1],
                facecolor=COLORS[name],
                edgecolor="white",
                marker="^",
                s=44,
                linewidth=0.85,
                zorder=6,
            )
        ax.scatter([0], [0], marker="*", s=155, color="#d62728", zorder=7)
        ax.set_ylabel(
            f"{row_label}\nTC-relative north (km)" if index == 0 else ""
        )
        ax.text(
            0.025,
            0.975,
            f"({chr(ord('a') + panel_offset + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.6,
            fontweight="bold",
            color="#15324a",
            bbox=dict(
                boxstyle="round,pad=0.16",
                facecolor="white",
                edgecolor="none",
                alpha=0.82,
            ),
            zorder=10,
        )
        ax.set_title(
            f"{name}\n{score_text}",
            fontsize=7.4,
            fontweight="bold",
            linespacing=1.18,
            pad=3.5,
        )
        if hide_upper_x and index < 3:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("")
    return mesh


def main():
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
        }
    )
    row = representative_segment()
    surface_model = PPO.load(SURFACE_MODEL, device="cpu")
    underwater_model = PPO.load(UNDERWATER_MODEL, device="cpu")

    surface_results = []
    for name, method in METHODS:
        lats, lons, final, d, t = surface_rollout(
            method, row, surface_model
        )
        env = type("SurfaceField", (), {})()
        env.d = d
        xy = relative_km(lats, lons, d["track_lon"][t], d["track_lat"][t])
        surface_results.append(
            (
                name,
                xy,
                f"Final tracking error: {final:.1f} km",
                env,
                t,
            )
        )

    underwater_results = []
    for name, method in METHODS:
        lats, lons, mean_current, weighted_coverage, env, t = underwater_rollout(
            method, row, underwater_model
        )
        xy = relative_km(
            lats,
            lons,
            env.d["track_lon"][t],
            env.d["track_lat"][t],
        )
        underwater_results.append(
            (
                name,
                xy,
                f"Mean current: {mean_current:.3f} m/s\n"
                f"Current-weighted coverage:\n"
                f"{weighted_coverage / 1000.0:.2f} × $10^3$",
                env,
                t,
            )
        )

    # Render at approximately the final manuscript width.  A 3-by-2 grid for
    # each task nearly doubles the on-page map area relative to the previous
    # five-panel row while keeping all panels on one figure page.
    fig = plt.figure(
        figsize=(8.4, 9.2),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        6,
        3,
        height_ratios=(1.08, 1.08, 0.065, 1.08, 1.08, 0.065),
        hspace=0.08,
    )
    surface_axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    underwater_axes = [
        fig.add_subplot(grid[3, 0]),
        fig.add_subplot(grid[3, 1]),
        fig.add_subplot(grid[3, 2]),
        fig.add_subplot(grid[4, 0]),
        fig.add_subplot(grid[4, 1]),
    ]
    legend_surface = fig.add_subplot(grid[1, 2])
    legend_underwater = fig.add_subplot(grid[4, 2])
    for legend_ax in (legend_surface, legend_underwater):
        legend_ax.axis("off")
    surface_mesh = plot_row(
        surface_axes,
        "Surface, 12 m/s",
        surface_results,
        "TC-relative east (km)",
        hide_upper_x=True,
        panel_offset=0,
        pad_fraction=0.18,
        min_pad_km=60.0,
        symmetric_about_tc=True,
        min_half_span_km=500.0,
    )
    underwater_mesh = plot_row(
        underwater_axes,
        "Subsurface, 30 m, 1.5 m/s",
        underwater_results,
        "TC-relative east (km)",
        hide_upper_x=True,
        panel_offset=5,
        pad_fraction=0.18,
        min_pad_km=65.0,
        min_width_km=720.0,
        min_height_km=430.0,
    )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=5, label="Initial position"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="black",
               markersize=5, label="Final position"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#d62728",
               markersize=8, label="TC core"),
    ]
    legend_surface.legend(
        handles=handles,
        loc="center",
        frameon=False,
        title="Surface",
        title_fontsize=8.2,
        fontsize=7.2,
    )
    legend_underwater.legend(
        handles=handles,
        loc="center",
        frameon=False,
        title="Subsurface",
        title_fontsize=8.2,
        fontsize=7.2,
    )
    fig.colorbar(
        surface_mesh,
        cax=fig.add_subplot(grid[2, :]),
        orientation="horizontal",
        label="ERA5 10 m wind speed (m s$^{-1}$)",
    )
    fig.colorbar(
        underwater_mesh,
        cax=fig.add_subplot(grid[5, :]),
        orientation="horizontal",
        label="HYCOM current speed at 30 m (m s$^{-1}$)",
    )
    fig.suptitle(
        "Traditional planner comparison on MATMO (2025)",
        fontsize=9.5,
        fontweight="bold",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUT.with_suffix(".png"),
        dpi=420,
        bbox_inches="tight",
        facecolor="white",
    )
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
