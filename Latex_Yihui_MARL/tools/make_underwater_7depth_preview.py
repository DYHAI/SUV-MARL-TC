"""Create a no-data preview of the planned seven-depth Figure 2."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
DEPTHS = (0, 10, 20, 30, 40, 50, 60)
SPEEDS = (2.0, 1.5)
METRICS = (
    ("Motion path length per SUV (km)", (0, 175)),
    ("Mean experienced current speed (m/s)", (0.0, 0.60)),
)
METHODS = (
    ("Active IPPO", "#2b6fa8"),
    ("Random", "#d9822b"),
    ("Passive drift", "#3b9b6d"),
)


def main() -> None:
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

    for row, speed in enumerate(SPEEDS):
        for col, (ylabel, ylim) in enumerate(METRICS):
            ax = axes[row, col]
            ax.set_xlim(-0.6, len(DEPTHS) - 0.4)
            ax.set_ylim(*ylim)
            ax.set_xticks(np.arange(len(DEPTHS)), [f"{d} m" for d in DEPTHS])
            ax.set_ylabel(ylabel)
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
            ax.text(
                0.5,
                0.45,
                "Preview layout\nData generation in progress",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#666666",
                fontsize=11,
            )
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
        "Planned seven-depth subsurface observation comparison",
        fontsize=14,
        fontweight="bold",
    )

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "underwater_methods_7depth_preview.pdf", bbox_inches="tight")
    fig.savefig(
        FIGDIR / "underwater_methods_7depth_preview.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
