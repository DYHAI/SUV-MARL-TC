#!/usr/bin/env python3
"""Build the by-radius and by-depth traditional-planner table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[1]
RESULTS = ROOT / "runs" / "results" / "classical_aggregate"
TABLES = ROOT / "paper" / "tables"

METHODS = (
    "IPPO",
    "Potential field",
    "Receding-horizon A*",
    "Lawnmower coverage",
    "Information greedy",
)
SHORT = {
    "IPPO": "IPPO",
    "Potential field": "PF",
    "Receding-horizon A*": "A*",
    "Lawnmower coverage": "Lawn",
    "Information greedy": "Greedy",
}


def cells(row: pd.Series, digits: int, winner: str) -> str:
    target = row.min() if winner == "min" else row.max()
    out = []
    for method in METHODS:
        value = float(row[method])
        text = f"{value:.{digits}f}"
        out.append(
            rf"\textbf{{{text}}}"
            if np.isclose(value, target, rtol=0.0, atol=1e-9)
            else text
        )
    return " & ".join(out)


def main() -> None:
    surface = pd.read_csv(RESULTS / "surface_all_methods_summary.csv")
    underwater = pd.read_csv(RESULTS / "underwater_all_methods_summary.csv")
    surface_final = surface.groupby(["radius_km", "method"])[
        "final_distance_km"
    ].mean().unstack("method")
    surface_average = surface.groupby(["radius_km", "method"])[
        "distance_avg_km"
    ].mean().unstack("method")
    by_depth = underwater[underwater.depth_m % 10 == 0]
    depth_metrics = {
        metric: by_depth.groupby(["depth_m", "method"])[metric]
        .mean()
        .unstack("method")
        for metric in ("mean_current_ms", "coverage_area_km2", "path_km")
    }

    lines = [
        r"\begin{center}",
        r"\captionof{table}{Distance and observation metrics for the traditional "
        r"planner comparison on the held-out TC segments. Panel A reports the "
        r"two surface distance metrics by deployment radius, averaged over the "
        r"four fleet sizes and both speed limits. Panel B reports mean "
        r"experienced current speed, geometric coverage, and path per SUV at each "
        r"depth, averaged over the two propulsion limits.}",
        r"\label{tab:planner_stratified}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrr@{}}",
        r"\toprule",
        r"\multicolumn{7}{l}{\textbf{Panel A: surface distance metrics}}\\",
        r"Radius & Metric & " + " & ".join(SHORT[m] for m in METHODS) + r"\\",
        r"\midrule",
    ]
    for radius in surface_final.index:
        lines.append(
            f"{int(radius)} km & Final tracking error (km) & "
            f"{cells(surface_final.loc[radius], 1, 'min')} \\\\"
        )
        lines.append(
            f" & Mean fleet-center error (km) & "
            f"{cells(surface_average.loc[radius], 1, 'min')} \\\\"
        )
        lines.append(r"\addlinespace[1.5pt]")

    lines += [
        r"\midrule",
        r"\multicolumn{7}{l}{\textbf{Panel B: subsurface metrics by depth}}\\",
        r"Depth & Metric & " + " & ".join(SHORT[m] for m in METHODS) + r"\\",
        r"\midrule",
    ]
    metric_rows = (
        ("Mean experienced current (m/s)", "mean_current_ms", 3, "max"),
        ("Geometric coverage (km$^2$)", "coverage_area_km2", 0, "max"),
        ("Path per SUV (km)", "path_km", 1, "min"),
    )
    depths = list(depth_metrics["mean_current_ms"].index)
    for depth in depths:
        for index, (label, metric, digits, winner) in enumerate(metric_rows):
            depth_label = f"{int(depth)} m" if index == 0 else ""
            lines.append(
                f"{depth_label} & {label} & "
                f"{cells(depth_metrics[metric].loc[depth], digits, winner)} \\\\"
            )
        if depth != depths[-1]:
            lines.append(r"\addlinespace[1.5pt]")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{center}", ""]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "planner_stratified.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(TABLES / "planner_stratified.tex")


if __name__ == "__main__":
    main()
