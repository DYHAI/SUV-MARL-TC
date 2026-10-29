"""Plot time-averaged and final TC-center distances for 12/20 m/s."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "Latex_Yihui_MARL"
LOGDIR = ROOT / "logs_surface_final_distance"
SUMMARY = DOC / "surface_vmax_10seed_summary.csv"
OUT = DOC / "figures" / "surface_vmax_distance_comparison.pdf"
OUT_PNG = DOC / "figures" / "surface_vmax_distance_comparison.png"
FLEETS = (4, 8, 12, 16)
SEEDS = range(10)
RADII = (100, 200, 400, 600, 800)


def collect_final() -> pd.DataFrame:
    rows = []
    for version in (12, 20):
        for fleet in FLEETS:
            for seed in SEEDS:
                for radius in RADII:
                    path = LOGDIR / f"v{version}_n{fleet}_s{seed}_d{radius}.log"
                    if not path.exists():
                        continue
                    match = re.search(r"^FINALROW\|([^\n]+)$", path.read_text(errors="replace"), re.M)
                    if not match:
                        continue
                    values = match.group(1).split("|")
                    rows.append(
                        {
                            "version_ms": version,
                            "fleet": fleet,
                            "seed": seed,
                            "distance_target_km": radius,
                            "final_distance": float(values[3]),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    time_avg = pd.read_csv(SUMMARY)
    final = collect_final()
    grouped = (
        final.groupby(["version_ms", "fleet", "distance_target_km"], as_index=False)
        .agg(
            final_distance=("final_distance", "mean"),
            final_distance_std=("final_distance", lambda x: x.std(ddof=1)),
            final_distance_n=("final_distance", "count"),
        )
    )
    if len(grouped) != 40:
        raise SystemExit(
            f"Incomplete final-distance evaluation: {len(grouped)}/40 conditions"
        )
    merged = time_avg.merge(
        grouped, on=["version_ms", "fleet", "distance_target_km"], how="inner"
    )
    merged.to_csv(DOC / "surface_distance_summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.8), sharex=True, layout="constrained"
    )
    metrics = (
        ("distance", "distance_std", "Time-averaged distance (km)"),
        ("final_distance", "final_distance_std", "Final distance (km)"),
    )
    colors = {
        4: "#0072B2",
        8: "#009E73",
        12: "#D55E00",
        16: "#CC79A7",
    }
    markers = {4: "o", 8: "s", 12: "^", 16: "D"}
    styles = {12: "--", 20: "-"}
    for col, (metric, std_col, ylabel) in enumerate(metrics):
        ax = axes[col]
        for version in (20, 12):
            for fleet in FLEETS:
                q = merged[
                    (merged.version_ms == version) & (merged.fleet == fleet)
                ].sort_values("distance_target_km")
                ax.errorbar(
                    q.distance_target_km,
                    q[metric],
                    yerr=q[std_col],
                    color=colors[fleet],
                    marker=markers[fleet],
                    linestyle=styles[version],
                    capsize=3,
                    lw=1.5,
                    elinewidth=0.8,
                    alpha=0.95,
                )
        ax.set_xlabel("Deployment radius (km)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(RADII)
        ax.grid(alpha=0.25)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=colors[n], marker=markers[n], linestyle="-")
        for n in FLEETS
    ]
    handles += [
        Line2D([0], [0], color="black", linestyle="-", label="$V_{max}$=20 m/s"),
        Line2D([0], [0], color="black", linestyle="--", label="$V_{max}$=12 m/s"),
    ]
    labels = [f"{n}-SUV" for n in FLEETS] + ["$V_{max}$=20 m/s", "$V_{max}$=12 m/s"]
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=6,
        loc="outside lower center",
    )
    fig.suptitle(
        "Surface tracking performance across deployment errors",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
