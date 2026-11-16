#!/usr/bin/env python3
"""Build the final subsurface IPPO-versus-planner figure and table."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "paper"
MEDIAN = ROOT / "results" / "primary"
FIG = DOC / "figures" / "FigC2_Underwater_Methods_7Depth"
TABLE = DOC / "tables" / "underwater_methods_10seed.tex"
SUMMARY = DOC / "figures" / "underwater_methods_10seed.csv"

DEPTHS = (0, 10, 20, 30, 40, 50, 60)
METHODS = (
    "IPPO",
    "Potential field",
    "Receding-horizon A*",
    "Lawnmower coverage",
    "Information greedy",
)
PLANNER_NAMES = {
    "PotentialFieldPlanner": "Potential field",
    "RecedingHorizonAStarPlanner": "Receding-horizon A*",
    "BoustrophedonCoveragePlanner": "Lawnmower coverage",
    "InformationGreedyPlanner": "Information greedy",
}
COLORS = {
    "IPPO": "#1F5A99",
    "Potential field": "#D55E00",
    "Receding-horizon A*": "#009E73",
    "Lawnmower coverage": "#CC79A7",
    "Information greedy": "#7A5195",
}
MARKERS = {
    "IPPO": "o",
    "Potential field": "s",
    "Receding-horizon A*": "^",
    "Lawnmower coverage": "D",
    "Information greedy": "v",
}
METRICS = (
    (
        "mean_current_speed_ms_mean",
        "mean_current_speed_ms_std",
        1.0,
        "Mean experienced current (m/s)",
    ),
    (
        "geometric_coverage_km2_mean",
        "geometric_coverage_km2_std",
        1.0e3,
        "Geometric coverage ($10^3$ km$^2$)",
    ),
    (
        "current_weighted_coverage_mean",
        "current_weighted_coverage_std",
        1.0e3,
        "Current-weighted coverage\n($10^3$ m s$^{-1}$ km$^2$)",
    ),
)


def load_summary() -> pd.DataFrame:
    ippo = pd.read_csv(MEDIAN / "underwater_ippo_summary.csv")
    classical = pd.read_csv(MEDIAN / "classical_underwater_summary.csv")
    classical["method"] = classical["planner"].map(PLANNER_NAMES)
    if classical["method"].isna().any():
        raise ValueError("unknown planner name in classical underwater summary")
    rows = []
    for _, row in ippo.iterrows():
        rows.append(
            {
                "depth_m": row.depth_m,
                "vmax_ms": row.vmax_ms,
                "method": "IPPO",
                "n_seed": int(row.n_training_seeds),
                "mean_current_speed_ms_mean": row.mean_current_speed_ms_mean,
                "mean_current_speed_ms_std": row.mean_current_speed_ms_std,
                "geometric_coverage_km2_mean": row.geometric_coverage_km2_mean,
                "geometric_coverage_km2_std": row.geometric_coverage_km2_std,
                "current_weighted_coverage_mean": row.current_weighted_coverage_mean,
                "current_weighted_coverage_std": row.current_weighted_coverage_std,
            }
        )
    for _, row in classical.iterrows():
        rows.append(
            {
                "depth_m": row.depth_m,
                "vmax_ms": row.vmax_ms,
                "method": row.method,
                "n_seed": int(row.n_training_seeds),
                "mean_current_speed_ms_mean": row.mean_current_speed_ms_mean,
                "mean_current_speed_ms_std": row.mean_current_speed_ms_std,
                "geometric_coverage_km2_mean": row.geometric_coverage_km2_mean,
                "geometric_coverage_km2_std": row.geometric_coverage_km2_std,
                "current_weighted_coverage_mean": row.current_weighted_coverage_mean,
                "current_weighted_coverage_std": row.current_weighted_coverage_std,
            }
        )
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(13.0, 7.0), sharex=True, layout="constrained"
    )
    for row, speed in enumerate((2.0, 1.5)):
        for col, (metric, std_col, scale, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            for method in METHODS:
                q = summary[
                    (summary.method == method)
                    & np.isclose(summary.vmax_ms, speed)
                ].sort_values("depth_m")
                ax.errorbar(
                    q.depth_m,
                    q[metric] / scale,
                    yerr=q[std_col].fillna(0.0) / scale,
                    color=COLORS[method],
                    marker=MARKERS[method],
                    markersize=4.5,
                    linewidth=1.6,
                    capsize=2.5,
                    label=method,
                )
            ax.set_xlabel("Observation depth (m)")
            ax.set_ylabel(ylabel)
            ax.set_xticks(DEPTHS)
            ax.grid(alpha=0.24)
            ax.text(
                0.98,
                0.97,
                rf"$V_{{\max}}={speed:.1f}$ m/s",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                    "pad": 1.5,
                },
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=5,
        loc="outside upper center",
    )
    fig.supxlabel(
        "IPPO: episode median within each seed, then arithmetic mean "
        r"$\pm$ sample SD across 10 seeds; planners: one episode median.",
        fontsize=8.5,
    )
    fig.savefig(FIG.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIG.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_table(summary: pd.DataFrame) -> None:
    names = ("IPPO", "PF", "A*", "Lawn", "Greedy")
    short = dict(zip(METHODS, names))
    lines = [
        r"\begin{table*}[p]",
        r"\centering",
        r"\caption{Subsurface observation comparison among IPPO and four "
        r"traditional planners over seven depths and two propulsion limits. "
        r"Values are episode medians averaged over ten IPPO training seeds, or "
        r"one deterministic planner evaluation. Higher values are better. "
        r"Geometric coverage and current-weighted coverage are reported in "
        r"$10^3$ km$^2$ and $10^3$ m s$^{-1}$ km$^2$, respectively.}",
        r"\label{tab:underwater_methods_10seed}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}rrlrrrrr@{}}",
        r"\toprule",
        r"Depth & $V_{\max}$ & Metric & IPPO & PF & A* & Lawn & Greedy \\",
        r"\midrule",
    ]
    metric_specs = (
        ("mean_current_speed_ms_mean", "Mean current (m/s)", 1.0, 3),
        ("geometric_coverage_km2_mean", "Geometric coverage", 1.0e3, 2),
        ("current_weighted_coverage_mean", "Current-weighted coverage", 1.0e3, 2),
    )
    for depth in DEPTHS:
        for speed in (1.5, 2.0):
            for idx, (metric, label, scale, digits) in enumerate(metric_specs):
                q = summary[
                    (summary.depth_m == depth)
                    & np.isclose(summary.vmax_ms, speed)
                ].set_index("method")
                values = [float(q.loc[method, metric]) / scale for method in METHODS]
                best = max(values)
                cells = []
                for value in values:
                    text = f"{value:.{digits}f}"
                    cells.append(
                        rf"\textbf{{{text}}}"
                        if np.isclose(value, best, rtol=0.0, atol=1e-9)
                        else text
                    )
                depth_cell = str(depth) if idx == 0 else ""
                speed_cell = f"{speed:.1f}" if idx == 0 else ""
                lines.append(
                    f"{depth_cell} & {speed_cell} & {label} & "
                    + " & ".join(cells)
                    + r" \\"
                )
            lines.append(r"\addlinespace[1pt]")
        if depth != DEPTHS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]
    TABLE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    DOC.joinpath("figures").mkdir(parents=True, exist_ok=True)
    DOC.joinpath("tables").mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    summary.to_csv(SUMMARY, index=False)
    make_figure(summary)
    make_table(summary)
    print(FIG.with_suffix(".pdf"))
    print(TABLE)


if __name__ == "__main__":
    main()
