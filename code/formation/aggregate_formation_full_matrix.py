#!/usr/bin/env python3
"""Aggregate and tabulate the complete surface-formation comparison."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from code.paths import PAPER_ROOT, REPO_ROOT, RESULTS_ROOT, RUNS_ROOT

ROOT = REPO_ROOT
EXPERIMENT = Path(
    os.environ.get("OFFICIAL_RUNS_ROOT", str(RUNS_ROOT / "official"))
)
RAW = EXPERIMENT / "results" / "surface_dynamics_only" / "formation" / "logs"
TABLES = PAPER_ROOT / "tables"
SUMMARY = (
    RESULTS_ROOT
    / "formation"
    / "formation_full_matrix_summary.csv"
)
SPEEDS = (12, 20)
FORMATIONS = ("ring", "line", "grid", "v")
RADII = (100, 200, 400, 600, 800)
METRICS = (
    "final_distance_km",
    "distance_avg_km",
    "path_km",
    "formation_radius_km",
    "pairwise_spacing_km",
    "on_land_fraction",
)


def load_and_summarize() -> pd.DataFrame:
    rows: list[dict] = []
    for speed in SPEEDS:
        for formation in FORMATIONS:
            for radius in RADII:
                path = RAW / f"{formation}_v{speed}_d{radius}.log"
                if not path.is_file():
                    raise FileNotFoundError(path)
                text = path.read_text(errors="replace")
                row_match = re.search(
                    r"^ROW\|([^\n]+)$", text, flags=re.MULTILINE
                )
                final_match = re.search(
                    r"^FINALROW\|([^\n]+)$", text, flags=re.MULTILINE
                )
                formation_match = re.search(
                    r"^FORMATIONROW\|([^\n]+)$", text, flags=re.MULTILINE
                )
                if not (row_match and final_match and formation_match):
                    raise ValueError(f"{path}: incomplete evaluation record")
                row_values = [float(x) for x in row_match.group(1).split("|")]
                final_values = [
                    float(x) for x in final_match.group(1).split("|")
                ]
                formation_values = [
                    float(x) for x in formation_match.group(1).split("|")
                ]
                if int(row_values[-1]) != 768:
                    raise ValueError(f"{path}: expected 768 evaluations")
                row = {
                    "speed_ms": speed,
                    "training_radius_km": 800,
                    "evaluation_radius_km": radius,
                    "formation": formation,
                    "n_segments": int(row_values[-1]),
                    "model": formation,
                    "final_distance_km": final_values[3],
                    "distance_avg_km": row_values[3],
                    "path_km": row_values[5],
                    "on_land_fraction": row_values[10],
                    "formation_radius_km": formation_values[3],
                    "pairwise_spacing_km": formation_values[4],
                }
                rows.append(row)
    return pd.DataFrame(rows)


def _best_cell(values: dict[str, float], formation: str, *, lower: bool) -> str:
    target = min(values.values()) if lower else max(values.values())
    value = values[formation]
    text = f"{value:.1f}"
    if np.isclose(value, target, rtol=0.0, atol=1e-9):
        return rf"\textbf{{{text}}}"
    return text


def write_table(summary: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Surface initial-formation comparison for eight SUVs. "
        r"Each policy was trained with formation-center offsets sampled inside "
        r"an 800 km radius and evaluated without retraining at fixed offsets "
        r"of 100, 200, 400, 600, and 800 km on the same 768 held-out TC "
        r"segments. Error is the final SUV-to-best-track-center distance and "
        r"path is the per-vehicle trajectory length. Bold entries are the best "
        r"value in each speed/radius row; error and path are assessed "
        r"separately. All distances are in km.}",
        r"\label{tab:formation_comparison}",
        r"\small",
        r"\setlength{\tabcolsep}{2.7pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ccccccccccc@{}}",
        r"\toprule",
        r"Speed & Train radius & Eval radius & \multicolumn{2}{c}{Ring} & "
        r"\multicolumn{2}{c}{Line} & \multicolumn{2}{c}{Grid} & "
        r"\multicolumn{2}{c}{V} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}"
        r"\cmidrule(lr){10-11}",
        r"(m/s) & (km) & (km) & Error & Path & Error & Path & "
        r"Error & Path & Error & Path \\",
        r"\midrule",
    ]
    for speed in SPEEDS:
        for radius in RADII:
            block = summary[
                (summary["speed_ms"] == speed)
                & (summary["evaluation_radius_km"] == radius)
            ].set_index("formation")
            errors = {f: float(block.loc[f, "final_distance_km"]) for f in FORMATIONS}
            paths = {f: float(block.loc[f, "path_km"]) for f in FORMATIONS}
            cells = [
                str(speed),
                "800",
                str(radius),
            ]
            for formation in FORMATIONS:
                cells.append(_best_cell(errors, formation, lower=True))
                cells.append(_best_cell(paths, formation, lower=True))
            lines.append(" & ".join(cells) + r" \\")
        if speed != SPEEDS[-1]:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table*}",
        "",
    ]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "formation_comparison.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    summary = load_and_summarize()
    summary.to_csv(SUMMARY, index=False)
    write_table(summary)
    print(summary.to_string(index=False))
    print(SUMMARY)
    print(TABLES / "formation_comparison.tex")


if __name__ == "__main__":
    main()
