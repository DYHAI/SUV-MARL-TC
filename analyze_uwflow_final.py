#!/usr/bin/env python3
"""Produce the final seven-depth underwater comparison statistics."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "eval_uwflow_500_summary.psv"
DOC = ROOT / "Latex_Yihui_MARL"
DEPTHS = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
SPEEDS = (1.5, 2.0)
METRICS = (
    "mean_V",
    "cov_area_km2",
    "info_cov_pct",
    "core_hit_pct",
    "recon_rmse",
    "path_km",
    "mean_speed_ms",
    "mean_a2",
)


def main() -> None:
    data = pd.read_csv(SRC, sep="|")
    output = []
    for depth in DEPTHS:
        for speed in SPEEDS:
            group = data[
                (data.depth == depth)
                & (data.vmax == speed)
                & (data.dur == 20.0)
            ]
            active = group[
                (group.algo == "sb3")
                & (group.reward == "C")
                & (group.train_seed >= 0)
            ].sort_values("train_seed")
            random = group[
                (group.algo == "random") & (group.reward == "random")
            ]
            drift = group[
                (group.algo == "idle") & (group.reward == "idle")
            ]
            if len(active) != 10 or len(random) != 1 or len(drift) != 1:
                raise ValueError(
                    f"incomplete data at depth={depth:g}, vmax={speed:g}: "
                    f"active={len(active)}, random={len(random)}, drift={len(drift)}"
                )

            drift_mean = drift.iloc[0].mean_V
            for method, part in (
                ("Active IPPO", active),
                ("Random", random),
                ("Passive drift", drift),
            ):
                row = {
                    "depth_m": depth,
                    "vmax_ms": speed,
                    "method": method,
                    "n_seed": len(part),
                    "active_seed_wins_vs_drift": (
                        int((active.mean_V > drift_mean).sum())
                        if method == "Active IPPO"
                        else np.nan
                    ),
                    "active_seed_significant_improvements": (
                        int(
                            (
                                (active.gain_V_pct > 0)
                                & (active.p_V < 0.05)
                            ).sum()
                        )
                        if method == "Active IPPO"
                        else np.nan
                    ),
                    "active_seed_significant_deficits": (
                        int(
                            (
                                (active.gain_V_pct < 0)
                                & (active.p_V < 0.05)
                            ).sum()
                        )
                        if method == "Active IPPO"
                        else np.nan
                    ),
                }
                for metric in METRICS:
                    row[metric] = float(part[metric].mean())
                    row[f"{metric}_std"] = (
                        float(part[metric].std(ddof=1))
                        if len(part) > 1
                        else 0.0
                    )
                output.append(row)

    result = pd.DataFrame(output)
    result.to_csv(DOC / "underwater_final_analysis.csv", index=False)

    report = []
    for depth in DEPTHS:
        for speed in SPEEDS:
            q = result[(result.depth_m == depth) & (result.vmax_ms == speed)]
            a = q[q.method == "Active IPPO"].iloc[0]
            r = q[q.method == "Random"].iloc[0]
            d = q[q.method == "Passive drift"].iloc[0]
            report.append(
                {
                    "depth_m": depth,
                    "vmax_ms": speed,
                    "active_current": a.mean_V,
                    "random_current": r.mean_V,
                    "drift_current": d.mean_V,
                    "active_current_gain_pct": 100.0
                    * (a.mean_V / d.mean_V - 1.0),
                    "active_path_ratio": a.path_km / d.path_km,
                    "active_energy_ratio": (
                        a.mean_a2 / d.mean_a2 if d.mean_a2 > 0 else np.nan
                    ),
                    "active_coverage_ratio": a.cov_area_km2 / d.cov_area_km2,
                    "random_coverage_ratio": r.cov_area_km2 / d.cov_area_km2,
                    "active_speed_ratio": a.mean_speed_ms / d.mean_speed_ms,
                    "active_gain_vs_random_pct": 100.0
                    * (a.mean_V / r.mean_V - 1.0),
                    "active_seed_wins_vs_drift": int(
                        a.active_seed_wins_vs_drift
                    ),
                    "active_significant_improvements": int(
                        a.active_seed_significant_improvements
                    ),
                    "active_significant_deficits": int(
                        a.active_seed_significant_deficits
                    ),
                }
            )
    compact = pd.DataFrame(report)
    compact.to_csv(DOC / "underwater_final_comparison.csv", index=False)

    lines = [
        "# Final seven-depth underwater comparison",
        "",
        "All policies are paired on the same 500 held-out episodes "
        "(50 TCs x 10 deployment seeds). Active values are means over ten "
        "independent training seeds. Random and passive drift are untrained "
        "paired baselines.",
        "",
        "| Depth | Vmax | Current A/R/D (m/s) | Active gain vs drift | "
        "Coverage A/R/D (km2) | Path A/D | Energy A/R/D | Seed wins | "
        "Significant + / - |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in compact.iterrows():
        depth = int(row.depth_m)
        speed = float(row.vmax_ms)
        q = result[
            (result.depth_m == row.depth_m) & (result.vmax_ms == row.vmax_ms)
        ].set_index("method")
        active = q.loc["Active IPPO"]
        random = q.loc["Random"]
        drift = q.loc["Passive drift"]
        lines.append(
            f"| {depth} m | {speed:.1f} m/s | "
            f"{active.mean_V:.3f}/{random.mean_V:.3f}/{drift.mean_V:.3f} | "
            f"{row.active_current_gain_pct:+.2f}% | "
            f"{active.cov_area_km2:.0f}/{random.cov_area_km2:.0f}/"
            f"{drift.cov_area_km2:.0f} | "
            f"{row.active_path_ratio:.2f}x | "
            f"{active.mean_a2:.3f}/{random.mean_a2:.3f}/"
            f"{drift.mean_a2:.3f} | "
            f"{int(row.active_seed_wins_vs_drift)}/10 | "
            f"{int(row.active_significant_improvements)}/"
            f"{int(row.active_significant_deficits)} |"
        )
    lines += [
        "",
        "## Summary",
        "",
        f"- Active mean-current gain vs drift: "
        f"{compact.active_current_gain_pct.min():+.2f}% to "
        f"{compact.active_current_gain_pct.max():+.2f}%.",
        f"- Active ground path / drift ground path: "
        f"{compact.active_path_ratio.min():.2f}x to "
        f"{compact.active_path_ratio.max():.2f}x.",
        f"- Active coverage / drift coverage: "
        f"{compact.active_coverage_ratio.min():.2f}x to "
        f"{compact.active_coverage_ratio.max():.2f}x.",
        f"- Active mean-current gain vs random control: "
        f"{compact.active_gain_vs_random_pct.min():+.2f}% to "
        f"{compact.active_gain_vs_random_pct.max():+.2f}%.",
        f"- Positive active-current gain in "
        f"{int((compact.active_current_gain_pct > 0).sum())}/14 configurations.",
        f"- Active wins in "
        f"{int(compact.active_seed_wins_vs_drift.sum())}/140 "
        "seed-configuration comparisons.",
    ]
    (DOC / "underwater_final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    pd.set_option("display.width", 220)
    print(compact.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nactive-current improvement > 0:",
          int((compact.active_current_gain_pct > 0).sum()), "/", len(compact))
    print("active-current improvement > 5%:",
          int((compact.active_current_gain_pct > 5).sum()), "/", len(compact))
    print("active path longer than drift:",
          int((compact.active_path_ratio > 1).sum()), "/", len(compact))


if __name__ == "__main__":
    main()
