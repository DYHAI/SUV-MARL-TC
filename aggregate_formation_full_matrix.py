#!/usr/bin/env python3
"""Aggregate and tabulate the complete surface-formation comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "robustness_eval" / "formation_full_matrix"
TABLES = ROOT / "Latex_Yihui_MARL" / "tables"
SUMMARY = ROOT / "robustness_eval" / "formation_full_matrix_summary.csv"
SPEEDS = (12, 20)
FORMATIONS = ("ring", "line", "grid")
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
                path = RAW / f"formation_{formation}_v{speed}_d{radius}.csv"
                if not path.is_file():
                    raise FileNotFoundError(path)
                frame = pd.read_csv(path)
                if len(frame) != 768:
                    raise ValueError(f"{path}: expected 768 rows, got {len(frame)}")
                if set(frame["condition"]) != {"clean"}:
                    raise ValueError(f"{path}: expected only clean evaluations")
                if not np.allclose(frame["vmax_ms"], speed):
                    raise ValueError(f"{path}: speed metadata mismatch")
                if not np.allclose(frame["radius_km"], radius):
                    raise ValueError(f"{path}: radius metadata mismatch")
                row = {
                    "speed_ms": speed,
                    "training_radius_km": 800,
                    "evaluation_radius_km": radius,
                    "formation": formation,
                    "n_segments": len(frame),
                    "model": str(frame["model"].iloc[0]),
                }
                row.update({metric: float(frame[metric].mean()) for metric in METRICS})
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
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ccccccccc@{}}",
        r"\toprule",
        r"Speed & Train radius & Eval radius & \multicolumn{2}{c}{Ring} & "
        r"\multicolumn{2}{c}{Line} & \multicolumn{2}{c}{Grid} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"(m/s) & (km) & (km) & Error & Path & Error & Path & Error & Path \\",
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
