#!/usr/bin/env python3
"""Draw the closed-loop SUV fleet-agent and TC environment interaction."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "paper" / "figures"
LANG = os.environ.get("RL_LOOP_LANG", "en").lower()
SHOW_MLP = os.environ.get("RL_LOOP_MLP", "0") == "1"
if LANG == "zh":
    FONT_PATH = os.environ.get("RL_LOOP_FONT", "/tmp/NotoSansCJKsc-Regular.otf")
    if Path(FONT_PATH).exists():
        font_manager.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=FONT_PATH
        ).get_name()


def tr(english: str, chinese: str) -> str:
    return chinese if LANG == "zh" else english


def add_auv(ax, image, xy, zoom=0.12):
    box = OffsetImage(image, zoom=zoom)
    artist = AnnotationBbox(
        box,
        xy,
        xycoords="axes fraction",
        frameon=False,
        box_alignment=(0.5, 0.5),
        zorder=8,
    )
    ax.add_artist(artist)
    return artist


def draw_cloud(ax, x, y, scale=1.0):
    parts = (
        (-0.042, -0.004, 0.080, 0.035),
        (-0.010, 0.010, 0.072, 0.050),
        (0.030, 0.002, 0.082, 0.040),
        (0.004, -0.008, 0.105, 0.032),
    )
    for dx, dy, width, height in parts:
        ax.add_patch(
            Ellipse(
                (x + dx * scale, y + dy * scale),
                width * scale,
                height * scale,
                facecolor="#f7f9fa",
                edgecolor="#c8d4da",
                linewidth=0.55,
                alpha=0.95,
                zorder=2,
            )
        )


def draw_environment(ax):
    card = FancyBboxPatch(
        (0.485, 0.05),
        0.50,
        0.88,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        edgecolor="#365466",
        facecolor="#f6f8fa",
        zorder=0,
    )
    ax.add_patch(card)

    sky = FancyBboxPatch(
        (0.497, 0.455),
        0.476,
        0.415,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.8,
        facecolor="white",
        edgecolor="#d5e0e5",
        zorder=1,
    )
    ocean = FancyBboxPatch(
        (0.497, 0.095),
        0.476,
        0.36,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.8,
        edgecolor="none",
        facecolor="#c9e4ea",
        zorder=1,
    )
    ax.add_patch(sky)
    ax.add_patch(ocean)

    draw_cloud(ax, 0.76, 0.77, scale=0.72)
    draw_cloud(ax, 0.88, 0.69, scale=0.64)
    draw_cloud(ax, 0.85, 0.57, scale=0.48)

    # Cyclone-like wind spirals in the atmosphere.
    center = np.array([0.66, 0.61])
    for radius, alpha in ((0.045, 0.95), (0.075, 0.75), (0.105, 0.52)):
        theta = np.linspace(0.0, 2.0 * np.pi, 180)
        x = center[0] + radius * np.cos(theta) * (0.62 + 0.38 * theta / (2 * np.pi))
        y = center[1] + radius * np.sin(theta) * (0.62 + 0.38 * theta / (2 * np.pi))
        ax.plot(x, y, color="#3d6f8b", lw=1.15, alpha=alpha, zorder=3)

    # Non-directional wave traces distinguish the ocean layer.
    for y0, phase in ((0.21, 0.0), (0.28, 0.8), (0.35, 1.6)):
        wave_x = np.linspace(0.525, 0.955, 180)
        wave_y = y0 + 0.006 * np.sin(22.0 * wave_x + phase)
        ax.plot(wave_x, wave_y, color="#176d8c", lw=1.0, alpha=0.58, zorder=3)

    ax.text(
        0.52,
        0.795,
        tr("Atmosphere\nERA5 10-m wind", "大气层\nERA5 10米风速"),
        fontsize=10,
        ha="left",
        va="center",
        color="#264f63",
        fontweight="bold",
    )
    ax.text(
        0.52,
        0.415,
        tr("Ocean\nHYCOM current", "海洋层\nHYCOM海流"),
        fontsize=10,
        ha="left",
        va="center",
        color="#175d78",
        fontweight="bold",
    )
def draw_agent(ax, auv_image, show_policy=True, y_shift=0.0):
    card = FancyBboxPatch(
        (0.03, 0.18 + y_shift),
        0.24,
        0.58,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        edgecolor="#365466",
        facecolor="#fffdf8",
        zorder=0,
    )
    ax.add_patch(card)
    ax.text(
        0.15,
        0.71 + y_shift,
        tr("SUV fleet agent", "SUV集群智能体"),
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#243746",
    )

    for xy in (
        (0.085, 0.54 + y_shift),
        (0.215, 0.54 + y_shift),
        (0.085, 0.31 + y_shift),
        (0.215, 0.31 + y_shift),
    ):
        add_auv(ax, auv_image, xy, zoom=0.285)

    ax.text(
        0.15,
        0.64 + y_shift,
        (
            tr("shared policy ", "共享策略 ") + r"$\pi_\theta$"
            if show_policy
            else tr("shared fleet observations", "共享集群观测")
        ),
        ha="center",
        va="center",
        fontsize=10 if show_policy else 9.4,
        color="#40545f",
        zorder=7,
    )


def draw_mlp(ax):
    """Draw an unboxed 3-4-3 MLP between the fleet and environment."""

    layer_x = (0.300, 0.375, 0.450)
    layer_sizes = (3, 4, 3)
    layer_centers = []
    for x, size in zip(layer_x, layer_sizes):
        ys = 0.49 + (np.arange(size) - (size - 1) / 2.0) * 0.095
        layer_centers.append(ys)

    colors = ("#72a9c8", "#508dad", "#d6814d")
    for left_x, left_ys, right_x, right_ys, color in zip(
        layer_x[:-1],
        layer_centers[:-1],
        layer_x[1:],
        layer_centers[1:],
        colors[:-1],
    ):
        for y0 in left_ys:
            for y1 in right_ys:
                ax.plot(
                    (left_x, right_x),
                    (y0, y1),
                    color=color,
                    linewidth=0.52,
                    alpha=0.42,
                    zorder=5,
                )

    for x, ys, color in zip(layer_x, layer_centers, colors):
        ax.scatter(
            np.full(len(ys), x),
            ys,
            s=68,
            facecolor=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=7,
        )


def draw_mlp_loop(ax):
    """Draw the action annotation and environment feedback around the MLP."""

    action_arrow = FancyArrowPatch(
        (0.27, 0.70),
        (0.49, 0.70),
        connectionstyle="arc3,rad=-0.45",
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color="#2b6fa8",
        zorder=6,
    )
    ax.add_patch(action_arrow)
    ax.text(
        0.38,
        0.84,
        tr("action ", "动作 ") + r"$\mathbf{a}_{t}$",
        ha="center",
        va="center",
        fontsize=10.3,
        color="#204f76",
        bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "none", "alpha": 1.0},
        zorder=10,
    )

    feedback_arrow = FancyArrowPatch(
        (0.49, 0.30),
        (0.27, 0.30),
        connectionstyle="arc3,rad=-0.45",
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color="#b65c2b",
        zorder=6,
    )
    ax.add_patch(feedback_arrow)
    ax.text(
        0.38,
        0.16,
        tr("next state ", "下一状态 ")
        + r"$\mathbf{s}_{t+1}$"
        + "\n"
        + tr("next observation ", "下一观测 ")
        + r"$\mathbf{o}_{t+1}$"
        + "\n"
        + tr("reward ", "奖励 ")
        + r"$r_{t}$",
        ha="center",
        va="center",
        fontsize=10.3,
        linespacing=1.08,
        color="#7a3a1d",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 1.0},
        zorder=10,
    )
    ax.text(
        0.375,
        0.705,
        tr("10 min interval\nclosed-loop decision", "10分钟间隔\n闭环决策"),
        ha="center",
        va="center",
        fontsize=8.5,
        color="#5b6670",
        style="italic",
        linespacing=1.08,
        zorder=10,
    )


def main() -> None:
    auv = plt.imread(FIGDIR / "auv_icon.png")
    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()

    draw_environment(ax)
    draw_agent(
        ax,
        auv,
        show_policy=True,
        y_shift=0.03 if SHOW_MLP else 0.0,
    )
    if SHOW_MLP:
        draw_mlp(ax)
        draw_mlp_loop(ax)
    add_auv(ax, np.fliplr(auv), (0.895, 0.47), zoom=0.22)
    add_auv(ax, np.fliplr(auv), (0.70, 0.47), zoom=0.22)
    add_auv(ax, np.fliplr(auv), (0.70, 0.28), zoom=0.22)

    approach_arrow = FancyArrowPatch(
        (0.865, 0.49),
        (0.73, 0.475),
        connectionstyle="arc3,rad=0.10",
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.55,
        color="#2b6fa8",
        zorder=7,
    )
    ax.add_patch(approach_arrow)
    ax.text(
        0.815,
        0.525,
        tr("Approach moving\nTC core", "接近移动\nTC核心区"),
        ha="center",
        va="center",
        fontsize=8.3,
        color="#204f76",
        fontweight="bold",
        linespacing=1.05,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "none", "alpha": 0.94},
        zorder=9,
    )

    dive_arrow = FancyArrowPatch(
        (0.70, 0.44),
        (0.70, 0.315),
        connectionstyle="arc3,rad=0",
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.35,
        linestyle=(0, (3.0, 2.5)),
        color="#175d78",
        zorder=7,
    )
    ax.add_patch(dive_arrow)
    ax.text(
        0.785,
        0.355,
        tr("Vertical dive\nto subsurface", "垂直下潜\n至水下观测"),
        ha="center",
        va="center",
        fontsize=8.1,
        color="#175d78",
        fontweight="bold",
        linespacing=1.05,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "none", "alpha": 0.94},
        zorder=9,
    )

    if not SHOW_MLP:
        action_arrow = FancyArrowPatch(
            (0.27, 0.58),
            (0.49, 0.58),
            connectionstyle="arc3,rad=-0.70",
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color="#2b6fa8",
            zorder=6,
        )
        feedback_arrow = FancyArrowPatch(
            (0.49, 0.32),
            (0.27, 0.32),
            connectionstyle="arc3,rad=-0.70",
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color="#b65c2b",
            zorder=6,
        )
        ax.add_patch(action_arrow)
        ax.add_patch(feedback_arrow)

        ax.text(
            0.38,
            0.77,
            tr("action ", "动作 ") + r"$\mathbf{a}_{t}$",
            ha="center",
            va="center",
            fontsize=10.3,
            color="#204f76",
            bbox={
                "boxstyle": "round,pad=0.45",
                "fc": "white",
                "ec": "none",
                "alpha": 1.0,
            },
            zorder=10,
        )
        ax.text(
            0.38,
            0.14,
            tr("next state ", "下一状态 ")
            + r"$\mathbf{s}_{t+1}$"
            "\n"
            + tr("next observation ", "下一观测 ")
            + r"$\mathbf{o}_{t+1}$"
            "\n"
            + tr("reward ", "奖励 ")
            + r"$r_{t}$",
            ha="center",
            va="center",
            fontsize=10.0,
            linespacing=1.12,
            color="#7a3a1d",
            bbox={
                "boxstyle": "round,pad=0.28",
                "fc": "white",
                "ec": "none",
                "alpha": 1.0,
            },
            zorder=10,
        )
        ax.text(
            0.38,
            0.50,
            tr("10 min interval\nclosed-loop decision", "10分钟间隔\n闭环决策"),
            ha="center",
            va="center",
            fontsize=9.2,
            color="#5b6670",
            style="italic",
        )

    FIGDIR.mkdir(parents=True, exist_ok=True)
    suffix = ("_MLP" if SHOW_MLP else "") + ("_zh" if LANG == "zh" else "")
    fig.savefig(
        FIGDIR / f"Fig2_RL_Agent_Environment_Loop{suffix}.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        FIGDIR / f"Fig2_RL_Agent_Environment_Loop{suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(FIGDIR / f"Fig2_RL_Agent_Environment_Loop{suffix}.pdf")


if __name__ == "__main__":
    main()
