"""Summarize IBTrACS translation speeds against the SUV speed envelope."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from code.paths import PAPER_ROOT, REPO_ROOT, RESULTS_ROOT

ROOT = REPO_ROOT
SPLIT = ROOT / "data" / "typhoon_split_500.json"
TRACKS = ROOT / "data" / "selected_typhoons"
OUT = RESULTS_ROOT / "analysis"
TABLE = PAPER_ROOT / "tables" / "tc_speed_envelope.tex"
KNOT_TO_MS = 0.5144444444444445


def main() -> None:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    storms = split["train"] + split["test"]
    point_speeds: list[np.ndarray] = []
    event_medians: list[float] = []
    event_p95: list[float] = []
    for storm in storms:
        path = TRACKS / storm / "track.csv"
        frame = pd.read_csv(
            path, usecols=["STORM_SPEED", "TRACK_TYPE"], low_memory=False
        )
        main = frame[
            frame["TRACK_TYPE"].astype(str).str.lower().eq("main")
        ]
        speed = pd.to_numeric(main["STORM_SPEED"], errors="coerce").dropna()
        speed_ms = speed.to_numpy(dtype=float) * KNOT_TO_MS
        if speed_ms.size == 0:
            continue
        point_speeds.append(speed_ms)
        event_medians.append(float(np.median(speed_ms)))
        event_p95.append(float(np.percentile(speed_ms, 95)))

    speeds = np.concatenate(point_speeds)
    summary = {
        "n_tc": len(storms),
        "n_track_points": int(speeds.size),
        "mean_ms": float(speeds.mean()),
        "median_ms": float(np.median(speeds)),
        "p90_ms": float(np.percentile(speeds, 90)),
        "p95_ms": float(np.percentile(speeds, 95)),
        "p99_ms": float(np.percentile(speeds, 99)),
        "fraction_le_1_5ms": float((speeds <= 1.5).mean()),
        "fraction_le_2_0ms": float((speeds <= 2.0).mean()),
        "fraction_le_12ms": float((speeds <= 12.0).mean()),
        "fraction_le_20ms": float((speeds <= 20.0).mean()),
        "event_median_min_ms": float(np.min(event_medians)),
        "event_median_max_ms": float(np.max(event_medians)),
        "event_p95_min_ms": float(np.min(event_p95)),
        "event_p95_max_ms": float(np.max(event_p95)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(OUT / "tc_speed_envelope_summary.csv", index=False)

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                r"\caption{IBTrACS TC translation speeds for the 500-TC dataset.",
                r"The percentages denote the fraction of main-track samples not exceeding",
                r"each platform speed. These values describe the speed envelope only;",
                r"tracking also depends on bearing error and remaining segment duration.}",
                r"\label{tab:tc_speed_envelope}",
                r"\small",
                r"\begin{tabular}{@{}lrrrrrr@{}}",
                r"\toprule",
                r"Quantity & P50 & P90 & P95 & P99 & Samples & TCs \\",
                r"\midrule",
                (
                    f"TC translation speed (m s$^{{-1}}$) & "
                    f"{summary['median_ms']:.2f} & {summary['p90_ms']:.2f} & "
                    f"{summary['p95_ms']:.2f} & {summary['p99_ms']:.2f} & "
                    f"{summary['n_track_points']:,} & {summary['n_tc']} \\\\"
                ),
                r"\midrule",
                r"Envelope & \multicolumn{2}{c}{Samples within speed} & "
                r"\multicolumn{2}{c}{Samples above speed} & \multicolumn{2}{c}{Interpretation} \\",
                (
                    f"1.5 m/s submerged & \\multicolumn{{2}}{{c}}"
                    f"{{{(speeds <= 1.5).mean() * 100:.1f}\\%}} & "
                    f"\\multicolumn{{2}}{{c}}{{{(speeds > 1.5).mean() * 100:.1f}\\%}} & "
                    r"\multicolumn{2}{l}{local sampling, not TC chasing} \\"
                ),
                (
                    f"2.0 m/s submerged & \\multicolumn{{2}}{{c}}"
                    f"{{{(speeds <= 2.0).mean() * 100:.1f}\\%}} & "
                    f"\\multicolumn{{2}}{{c}}{{{(speeds > 2.0).mean() * 100:.1f}\\%}} & "
                    r"\multicolumn{2}{l}{local sampling, not TC chasing} \\"
                ),
                (
                    f"12 m/s surface & \\multicolumn{{2}}{{c}}"
                    f"{{{(speeds <= 12.0).mean() * 100:.1f}\\%}} & "
                    f"\\multicolumn{{2}}{{c}}{{{(speeds > 12.0).mean() * 100:.1f}\\%}} & "
                    r"\multicolumn{2}{l}{current-platform scenario} \\"
                ),
                (
                    f"20 m/s surface & \\multicolumn{{2}}{{c}}"
                    f"{{{(speeds <= 20.0).mean() * 100:.1f}\\%}} & "
                    f"\\multicolumn{{2}}{{c}}{{{(speeds > 20.0).mean() * 100:.1f}\\%}} & "
                    r"\multicolumn{2}{l}{future-upgrade scenario} \\"
                ),
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(OUT / "tc_speed_envelope_summary.csv")
    print(TABLE)


if __name__ == "__main__":
    main()
