"""Summarize segment-mean TC translation speeds against SUV speed limits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SEGMENTS = ROOT / "data" / "segments_primary_400.csv"
OUT = ROOT / "results" / "surface"
TABLE = ROOT / "paper" / "tables" / "tc_speed_envelope.tex"


def main() -> None:
    segments = pd.read_csv(SEGMENTS)
    duration_h = (segments["t1"] - segments["t0"]) * 10.0 / 60.0
    speeds = (segments["arc_km"] / duration_h / 3.6).to_numpy(dtype=float)
    summary = {
        "n_tc": int(segments["storm"].nunique()),
        "n_segments": int(speeds.size),
        "mean_ms": float(speeds.mean()),
        "median_ms": float(np.median(speeds)),
        "p90_ms": float(np.percentile(speeds, 90)),
        "p95_ms": float(np.percentile(speeds, 95)),
        "p99_ms": float(np.percentile(speeds, 99)),
        "fraction_le_1_5ms": float((speeds <= 1.5).mean()),
        "fraction_le_2_0ms": float((speeds <= 2.0).mean()),
        "fraction_le_12ms": float((speeds <= 12.0).mean()),
        "fraction_le_20ms": float((speeds <= 20.0).mean()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(OUT / "tc_speed_envelope_summary.csv", index=False)

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                r"\caption{IBTrACS TC translation speeds aggregated over the",
                r"3,392 open-ocean segments in the 400-TC dataset.",
                r"Segment speed is segment arc length divided by elapsed time.",
                r"The percentages denote the fraction of segments not exceeding each platform speed;",
                r"tracking also depends on bearing error and remaining segment duration.}",
                r"\label{tab:tc_speed_envelope}",
                r"\small",
                r"\begin{tabular}{@{}lrrrrrr@{}}",
                r"\toprule",
                r"Quantity & P50 & P90 & P95 & P99 & Segments & TCs \\",
                r"\midrule",
                (
                    f"Segment-mean TC translation speed (m s$^{{-1}}$) & "
                    f"{summary['median_ms']:.2f} & {summary['p90_ms']:.2f} & "
                    f"{summary['p95_ms']:.2f} & {summary['p99_ms']:.2f} & "
                    f"{summary['n_segments']:,} & {summary['n_tc']} \\\\"
                ),
                r"\midrule",
                r"Envelope & \multicolumn{2}{c}{Segments within speed} & "
                r"\multicolumn{2}{c}{Segments above speed} & \multicolumn{2}{c}{Interpretation} \\",
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
