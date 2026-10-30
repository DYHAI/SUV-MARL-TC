"""Plot completed underwater evaluations from a configurable summary."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
SOURCE = Path(
    os.environ.get(
        "UWFLOW_SUMMARY_PSV",
        ROOT.parent / "eval_uwflow_500_summary.psv",
    )
)
DEPTHS = (0, 10, 20, 30, 40, 50, 60)
SPEEDS = (2.0, 1.5)
METRICS = (
    ("path_mean", "path_std", "Motion path length per SUV (km)", (0, 175)),
    (
        "current_mean",
        "current_std",
        "Mean experienced current speed (m/s)",
        (0.0, 0.60),
    ),
)
METHODS = (
    ("Active IPPO", "#2b6fa8"),
    ("Random", "#d9822b"),
    ("Passive drift", "#3b9b6d"),
)


def aggregate_completed(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complete ten-seed final configurations; skip partial depths."""
    rows = []
    for depth in DEPTHS:
        for speed in SPEEDS:
            q = data[
                (data.depth == depth)
                & (data.vmax == speed)
                & (data.dur == 20.0)
            ]
            active = q[
                (q.algo == "sb3")
                & (q.reward == "C")
                & (q.train_seed >= 0)
            ]
            random = q[(q.algo == "random") & (q.reward == "random")]
            drift = q[(q.algo == "idle") & (q.reward == "idle")]
            if len(active) != 10 or random.empty or drift.empty:
                continue
            for method, part in (
                ("Active IPPO", active),
                ("Random", random.iloc[[0]]),
                ("Passive drift", drift.iloc[[0]]),
            ):
                rows.append(
                    {
                        "depth_m": depth,
                        "vmax_ms": speed,
                        "method": method,
                        "path_mean": part.path_km.mean(),
                        "path_std": (
                            part.path_km.std(ddof=1)
                            if len(part) > 1
                            else 0.0
                        ),
                        "current_mean": part.mean_V.mean(),
                        "current_std": (
                            part.mean_V.std(ddof=1)
                            if len(part) > 1
                            else 0.0
                        ),
                        "coverage_mean": part.cov_area_km2.mean(),
                        "coverage_std": (
                            part.cov_area_km2.std(ddof=1)
                            if len(part) > 1
                            else 0.0
                        ),
                        "mean_speed_ms": part.mean_speed_ms.mean(),
                        "propulsion_energy_proxy": part.mean_a2.mean(),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    data = pd.read_csv(SOURCE, sep="|")
    completed = aggregate_completed(data)
    available_depths = sorted(completed.depth_m.unique().astype(int).tolist())
    missing_depths = [depth for depth in DEPTHS if depth not in available_depths]
    print("completed depths:", available_depths)
    print("pending depths:", missing_depths)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(10.0, 6.4), sharex=True, layout="constrained"
    )
    width = 0.24
    x = np.arange(len(DEPTHS))

    for row, speed in enumerate(SPEEDS):
        for col, (mean_column, std_column, ylabel, ylim) in enumerate(METRICS):
            ax = axes[row, col]
            for k, (label, color) in enumerate(METHODS):
                part = completed[completed.method == label]
                heights = np.full(len(DEPTHS), np.nan)
                errors = np.full(len(DEPTHS), np.nan)
                for depth in available_depths:
                    match = part[
                        (part.depth_m == depth) & (part.vmax_ms == speed)
                    ]
                    if match.empty:
                        continue
                    index = DEPTHS.index(depth)
                    heights[index] = float(match.iloc[0][mean_column])
                    errors[index] = float(match.iloc[0][std_column])
                ax.bar(
                    x + (k - 1) * width,
                    heights,
                    width,
                    yerr=errors,
                    capsize=3,
                    color=color,
                    edgecolor="black",
                    linewidth=0.4,
                    label=label,
                )

            ax.set_xlim(-0.6, len(DEPTHS) - 0.4)
            ax.set_ylim(*ylim)
            ax.set_xticks(x, [f"{depth} m" for depth in DEPTHS])
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.text(
                0.01,
                0.98,
                rf"$V_{{\max}}$ = {speed:.1f} m/s",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                fontweight="bold",
            )
            ax.grid(axis="y", alpha=0.24, linestyle="--", linewidth=0.7)
            if row == 1:
                ax.set_xlabel("Ocean depth")

    handles = [Line2D([0], [0], color=color, lw=5) for _, color in METHODS]
    labels = [name for name, _ in METHODS]
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="outside lower center",
    )
    fig.suptitle(
        "Subsurface strategy comparison across 0--60 m (20 h)",
        fontsize=14,
        fontweight="bold",
    )

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGDIR / "underwater_methods_7depth_preview.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        FIGDIR / "underwater_methods_7depth_preview.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    completed.to_csv(
        ROOT / "underwater_methods_completed_depths.csv", index=False
    )


if __name__ == "__main__":
    main()
