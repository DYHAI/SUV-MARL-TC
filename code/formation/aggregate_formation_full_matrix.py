#!/usr/bin/env python3
"""Aggregate and tabulate the complete surface-formation comparison."""

from __future__ import annotations

import re
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "runs"
RAW = EXPERIMENT / "results" / "surface_dynamics_only" / "formation" / "logs"
if os.environ.get("SURFACE_CENTER_RESULT_ROOT"):
    RAW = (
        Path(os.environ["SURFACE_CENTER_RESULT_ROOT"])
        / "formation"
        / "logs"
    )
TABLES = ROOT / "paper" / "tables"
SUMMARY = (
    EXPERIMENT
    / "results"
    / "surface_dynamics_only"
    / "formation"
    / "formation_full_matrix_summary.csv"
)
if os.environ.get("SURFACE_CENTER_RESULT_ROOT"):
    SUMMARY = (
        Path(os.environ["SURFACE_CENTER_RESULT_ROOT"])
        / "summary"
        / "formation_full_matrix_summary.csv"
    )
SPEEDS = (12, 20)
FORMATIONS = ("ring", "line", "grid", "v")
RADII = (100, 200, 400, 600, 800)
REQUIRE_RIGID = os.environ.get("RIGID_FORMATION", "0") == "1"
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
    expected_evaluations: int | None = None
    for speed in SPEEDS:
        for formation in FORMATIONS:
            for radius in RADII:
                path = RAW / f"{formation}_v{speed}_d{radius}.log"
                if not path.is_file():
                    raise FileNotFoundError(path)
                segment_path = (
                    path.parent.parent
                    / "segments"
                    / f"{formation}_v{speed}_d{radius}_seg.csv"
                )
                if not segment_path.is_file():
                    raise FileNotFoundError(segment_path)
                segment_values = pd.read_csv(segment_path)
                if "final_wind_speed_ms" not in segment_values:
                    raise ValueError(
                        f"{segment_path}: missing final_wind_speed_ms"
                    )
                text = path.read_text(errors="replace")
                row_match = re.search(
                    r"^ROW\|([^\n]+)$", text, flags=re.MULTILINE
                )
                final_match = re.search(
                    r"^FINALROW\|([^\n]+)$", text, flags=re.MULTILINE
                )
                centre_match = re.search(
                    r"^CENTERROW\|([^\n]+)$", text, flags=re.MULTILINE
                )
                formation_match = re.search(
                    r"^FORMATIONROW\|([^\n]+)$", text, flags=re.MULTILINE
                )
                rigid_match = re.search(
                    r"^RIGIDROW\|([^\n]+)$", text, flags=re.MULTILINE
                )
                constraint_match = re.search(
                    r"^CONSTRAINTROW\|([^\n]+)$", text, flags=re.MULTILINE
                )
                if not (
                    row_match
                    and final_match
                    and centre_match
                    and formation_match
                    and constraint_match
                    and (rigid_match or not REQUIRE_RIGID)
                ):
                    raise ValueError(f"{path}: incomplete evaluation record")
                row_values = [float(x) for x in row_match.group(1).split("|")]
                final_values = [
                    float(x) for x in final_match.group(1).split("|")
                ]
                centre_values = [
                    float(x) for x in centre_match.group(1).split("|")
                ]
                formation_values = [
                    float(x) for x in formation_match.group(1).split("|")
                ]
                rigid_values = (
                    [float(x) for x in rigid_match.group(1).split("|")]
                    if rigid_match
                    else [800.0, float(radius), 8.0, 0.0]
                )
                constraint_values = [
                    float(x)
                    for x in constraint_match.group(1).split("|")
                ]
                n_evaluations = int(row_values[-1])
                if expected_evaluations is None:
                    expected_evaluations = n_evaluations
                elif n_evaluations != expected_evaluations:
                    raise ValueError(
                        f"{path}: found {n_evaluations} evaluations, "
                        f"expected {expected_evaluations}"
                    )
                if rigid_values[3] > 1e-6:
                    raise ValueError(f"{path}: rigid formation drift")
                if constraint_values[3] > speed + 1e-6:
                    raise ValueError(f"{path}: speed limit violated")
                if constraint_values[4] > 0.03 + 1e-6:
                    raise ValueError(f"{path}: acceleration limit violated")
                if constraint_values[5] > 1.0 + 1e-6:
                    raise ValueError(f"{path}: yaw-rate limit violated")
                row = {
                    "speed_ms": speed,
                    "training_radius_km": 800,
                    "evaluation_radius_km": radius,
                    "formation": formation,
                    "n_segments": n_evaluations,
                    "model": formation,
                    "final_distance_km": float(
                        segment_values["final_centre_dist"].median()
                    ),
                    "distance_avg_km": centre_values[3],
                    "agent_final_distance_km": final_values[3],
                    "agent_distance_avg_km": row_values[3],
                    "path_km": row_values[5],
                    "final_wind_speed_ms": float(
                        segment_values["final_wind_speed_ms"].median()
                    ),
                    "on_land_fraction": row_values[10],
                    "formation_radius_km": formation_values[3],
                    "pairwise_spacing_km": formation_values[4],
                    "max_formation_offset_drift_km": rigid_values[3],
                    "max_prop_speed_ms": constraint_values[3],
                    "max_accel_mps2": constraint_values[4],
                    "max_yaw_rate_dps": constraint_values[5],
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
    n_segments = int(summary["n_segments"].iloc[0])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Surface initial-formation comparison for eight SUVs. "
        r"All four layouts use a mean radius of 100 km. "
        r"Each policy was trained with formation-center offsets sampled inside "
        r"an 800 km radius and evaluated without retraining at fixed offsets "
        rf"of 100, 200, 400, 600, and 800 km on the same {n_segments} held-out TC "
        r"segments. Final tracking error is the fleet-center-to-TC-core "
        r"distance at the end of the episode, and final wind speed is the "
        r"fleet-mean ERA5 10 m wind speed at the final SUV positions. Bold "
        r"entries are the best value in each speed/radius row, with smaller "
        r"final tracking error and larger final wind speed preferred. Final "
        r"tracking error and mean radius are in km, and final wind speed is "
        r"in m\,s$^{-1}$.}",
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
        r"(m/s) & (km) & (km) & Final tracking error & Final wind speed & "
        r"Final tracking error & Final wind speed & "
        r"Final tracking error & Final wind speed & "
        r"Final tracking error & Final wind speed \\",
        r"\midrule",
    ]
    for speed in SPEEDS:
        for radius in RADII:
            block = summary[
                (summary["speed_ms"] == speed)
                & (summary["evaluation_radius_km"] == radius)
            ].set_index("formation")
            errors = {f: float(block.loc[f, "final_distance_km"]) for f in FORMATIONS}
            winds = {
                f: float(block.loc[f, "final_wind_speed_ms"])
                for f in FORMATIONS
            }
            cells = [
                str(speed),
                "800",
                str(radius),
            ]
            for formation in FORMATIONS:
                cells.append(_best_cell(errors, formation, lower=True))
                wind_value = winds[formation]
                wind_text = f"{wind_value:.2f}"
                if np.isclose(
                    wind_value,
                    max(winds.values()),
                    rtol=0.0,
                    atol=1e-9,
                ):
                    wind_text = rf"\textbf{{{wind_text}}}"
                cells.append(wind_text)
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
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY, index=False)
    write_table(summary)
    print(summary.to_string(index=False))
    print(SUMMARY)
    print(TABLES / "formation_comparison.tex")


if __name__ == "__main__":
    main()
