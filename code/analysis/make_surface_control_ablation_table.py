"""Generate the low-pass and on-land-penalty ablation table."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "tables" / "surface_control_ablations.tex"
SUMMARY = (
    ROOT
    / "runs"
    / "results"
    / "primary_median"
    / "surface_filter_summary.csv"
)

def read_land_fraction(path: Path) -> float:
    text = path.read_text(errors="replace")
    match = re.search(r"上岸占比\(AUV-step\)\s+([0-9.]+)\s*%", text)
    if not match:
        raise ValueError(f"on-land fraction not found in {path}")
    return float(match.group(1))


def read_lowpass_metrics(path: Path) -> tuple[float, float, float, float]:
    text = path.read_text(errors="replace")
    final_distance = re.search(r"最终中心距离\s+([0-9.]+)\s+km", text)
    path_km = re.search(r"航迹长度\s+([0-9.]+)\s+km", text)
    mean_speed = re.search(r"平均航速\s+([0-9.]+)\s+m/s", text)
    max_speed = re.search(r"最大速度\s+([0-9.]+)\s+m/s", text)
    if not (final_distance and path_km and mean_speed and max_speed):
        raise ValueError(f"metrics not found in {path}")
    return (
        float(final_distance.group(1)),
        float(path_km.group(1)),
        float(mean_speed.group(1)),
        float(max_speed.group(1)),
    )


def main() -> None:
    if not SUMMARY.exists():
        raise SystemExit(
            f"waiting for primary median filter summary: {SUMMARY}"
        )
    summary = pd.read_csv(SUMMARY)
    required = {
        "n_auv",
        "filter_beta",
        "n_training_seeds",
        "final_tracking_error_km_mean",
        "path_km_mean",
        "mean_speed_ms_mean",
        "max_speed_ms_mean",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise SystemExit(f"missing median filter columns: {sorted(missing)}")
    if set(summary.n_training_seeds.astype(int)) != {10}:
        raise SystemExit("primary filter summary must contain ten training seeds")
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Execution-layer low-pass filtering for 4/8/12/16 surface "
        r"SUVs on 262 held-out segments from 40 unseen TCs in the 400-TC "
        r"dataset. Each trained actor is evaluated with the filter off and on "
        r"($\beta=0.8$) on identical validation episodes. Values are medians "
        r"over episodes within each seed, averaged over ten seeds.}",
        r"\label{tab:surface_control_ablations}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{0.9}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lr rrrr@{}}",
        r"\toprule",
        r"Fleet & Filter & Final tracking error (km) & Path per SUV (km) & "
        r"Mean speed (m/s) & Maximum speed (m/s) \\",
        r"\midrule",
    ]
    for fleet in (4, 8, 12, 16):
        part = summary[summary.n_auv.astype(int) == fleet]
        values = {
            float(row.filter_beta): row
            for _, row in part.iterrows()
        }
        best = {
            metric: min(values, key=lambda beta: values[beta][metric])
            for metric in (
                "final_tracking_error_km_mean",
                "path_km_mean",
                "mean_speed_ms_mean",
                "max_speed_ms_mean",
            )
        }
        for beta, label in ((0.0, "Off"), (0.8, "On")):
            row = values[beta]
            cells = []
            for metric, digits in (
                ("final_tracking_error_km_mean", 1),
                ("path_km_mean", 1),
                ("mean_speed_ms_mean", 2),
                ("max_speed_ms_mean", 2),
            ):
                text = f"{row[metric]:.{digits}f}"
                cells.append(
                    r"\textbf{" + text + "}"
                    if best[metric] == beta
                    else text
                )
            lines.append(f"{fleet} & {label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\par\smallskip",
        r"\parbox{\textwidth}{\footnotesize Final tracking error is the "
        r"fleet-center-to-TC-core distance at the final time step. Off and on "
        r"change only the execution-layer filter on the same trained actor.}",
        r"\end{table*}",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(OUT)


if __name__ == "__main__":
    main()
