"""Plot final tracking error and final wind speed for 12/20 m/s."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "paper"
SUMMARY = (
    ROOT
    / "runs"
    / "results"
    / "primary_median"
    / "surface_main_summary.csv"
)
OUT = DOC / "figures" / "Fig4_Surface_Vmax_Distance_Comparison.pdf"
OUT_PNG = DOC / "figures" / "Fig4_Surface_Vmax_Distance_Comparison.png"
FLEETS = (4, 8, 12, 16)
RADII = (100, 200, 400, 600, 800)


def main() -> None:
    primary = pd.read_csv(SUMMARY)
    merged = primary.rename(
        columns={
            "vmax_ms": "version_ms",
            "n_auv": "fleet",
            "radius_km": "distance_target_km",
            "final_tracking_error_km_mean": "final_distance",
            "final_tracking_error_km_std": "final_distance_std",
            "final_wind_speed_ms_mean": "final_wind_speed",
            "final_wind_speed_ms_std": "final_wind_speed_std",
        }
    )
    required = {
        "version_ms",
        "fleet",
        "distance_target_km",
        "final_distance",
        "final_distance_std",
        "final_wind_speed",
        "final_wind_speed_std",
    }
    missing = required.difference(merged.columns)
    if missing:
        raise SystemExit(f"Missing surface summary columns: {sorted(missing)}")
    if len(merged) != 40:
        raise SystemExit(
            f"Incomplete surface summary: {len(merged)}/40 conditions"
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
        1, 2, figsize=(11.5, 4.8), sharex=True, layout="constrained"
    )
    metrics = (
        (
            "final_distance",
            "final_distance_std",
            "Final tracking error (km)",
        ),
        (
            "final_wind_speed",
            "final_wind_speed_std",
            "Final mean 10 m wind speed (m s$^{-1}$)",
        ),
    )
    colors = {
        4: "#0077BB",
        8: "#EE7733",
        12: "#009988",
        16: "#CC3311",
    }
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
                    marker="none",
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
        Line2D([0], [0], color=colors[n], linestyle="-")
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
        "Surface tracking performance across deployment error",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
