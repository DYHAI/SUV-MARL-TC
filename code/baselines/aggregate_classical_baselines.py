#!/usr/bin/env python3
"""Aggregate and plot the causal traditional-planning baselines."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
DOC = ROOT / "Latex_Yihui_MARL"
EXPERIMENT = ROOT / "constrained_dynamics_experiments"
OUT = EXPERIMENT / "results" / "classical_aggregate"
FIG = OUT / "figures"
UNDERWATER_RESULTS = EXPERIMENT / "results" / "underwater"
SURFACE_CLASSICAL = EXPERIMENT / "results" / "surface_dynamics_only" / "classical"
UNDERWATER_CLASSICAL = EXPERIMENT / "results" / "classical_underwater"

DEPTHS = (0, 10, 20, 30, 40, 50, 60)
SPEEDS_UW = (2.0, 1.5)
FLEETS = (4, 8, 12, 16)
SPEEDS_SURFACE = (20.0, 12.0)
RADII = (100, 200, 400, 600, 800)

METHODS = {
    "PotentialFieldPlanner": "Potential field",
    "RecedingHorizonAStarPlanner": "Receding-horizon A*",
    "BoustrophedonCoveragePlanner": "Lawnmower coverage",
    "InformationGreedyPlanner": "Information greedy",
}
STYLE = {
    "IPPO": dict(color="#245c8a", marker="o"),
    "Potential field": dict(color="#d47b20", marker="s"),
    "Receding-horizon A*": dict(color="#3d9664", marker="^"),
    "Lawnmower coverage": dict(color="#7b52ab", marker="D"),
    "Information greedy": dict(color="#b23a48", marker="P"),
}


def learned_episode_metrics(depth_m: float, vmax_ms: float) -> pd.DataFrame:
    """Load IPPO evaluations and aggregate training seeds per deployment episode."""
    eval_path = UNDERWATER_RESULTS / (
        f"eval_info_info_d{depth_m:g}_start200_20h_v{vmax_ms:g}_500.csv"
    )
    learned = pd.read_csv(eval_path)
    learned = learned[learned.policy.str.startswith("uw_")].copy()
    learned = learned.rename(
        columns={
            "mean_V": "mean_current_ms",
            "cov_area_km2": "coverage_area_km2",
        }
    )
    learned["current_weighted_coverage"] = (
        learned["mean_current_ms"] * learned["coverage_area_km2"]
    )
    metrics = (
        "mean_current_ms",
        "coverage_area_km2",
        "path_km",
        "current_weighted_coverage",
    )
    return (
        learned.groupby(["storm", "seed"], as_index=False)[list(metrics)]
        .mean()
        .rename(columns={"seed": "deployment_index"})
    )


def aggregate_surface() -> pd.DataFrame:
    paths = sorted(SURFACE_CLASSICAL.glob("surface_*_test_n*_v*_s1.csv"))
    if len(paths) != 32:
        raise SystemExit(f"expected 32 surface planner files, found {len(paths)}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["method"] = frame["planner"].map(METHODS)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    keys = ["method", "n_auv", "vmax_ms", "radius_km"]
    metrics = [
        "distance_avg_km",
        "final_distance_km",
        "path_km",
        "mean_speed_ms",
        "on_land_fraction",
        "duration_h",
    ]
    grouped = raw.groupby(keys, as_index=False).agg(
        distance_avg_km=("distance_avg_km", "mean"),
        distance_avg_km_std=("distance_avg_km", "std"),
        final_distance_km=("final_distance_km", "mean"),
        final_distance_km_std=("final_distance_km", "std"),
        path_km=("path_km", "mean"),
        path_km_std=("path_km", "std"),
        mean_speed_ms=("mean_speed_ms", "mean"),
        mean_speed_ms_std=("mean_speed_ms", "std"),
        on_land_fraction=("on_land_fraction", "mean"),
        on_land_fraction_std=("on_land_fraction", "std"),
        duration_h=("duration_h", "mean"),
        duration_h_std=("duration_h", "std"),
        n_episode=("planner", "size"),
    )
    grouped.to_csv(OUT / "surface_classical_summary.csv", index=False)

    # Add the ten-seed MARL means in the same long-form schema.
    dynamic = pd.read_csv(DOC / "surface_distance_summary.csv")
    marl = dynamic.rename(
        columns={
            "fleet": "n_auv",
            "version_ms": "vmax_ms",
            "distance_target_km": "radius_km",
            "distance": "distance_avg_km",
            "final_distance": "final_distance_km",
            "path": "path_km",
            "mean_speed": "mean_speed_ms",
            "on_land_fraction": "on_land_fraction",
        }
    )
    marl["method"] = "IPPO"
    keep = [
        "method", "n_auv", "vmax_ms", "radius_km",
        "distance_avg_km", "final_distance_km", "path_km",
        "mean_speed_ms", "on_land_fraction",
    ]
    marl = marl[keep]
    marl["n_episode"] = 7680
    combined = pd.concat([marl, grouped[keep + ["duration_h", "n_episode"]]], ignore_index=True)
    combined.to_csv(OUT / "surface_all_methods_summary.csv", index=False)
    return raw, combined


def aggregate_underwater() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(UNDERWATER_CLASSICAL.glob("underwater_*_test_d*_v*_s10.csv"))
    if len(paths) != 56:
        raise SystemExit(f"expected 56 underwater planner files, found {len(paths)}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["method"] = frame["planner"].map(METHODS)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw["current_weighted_coverage"] = (
        raw["mean_current_ms"] * raw["coverage_area_km2"]
    )
    summary = (
        raw.groupby(["method", "depth_m", "vmax_ms"], as_index=False)
        .agg(
            mean_current_ms=("mean_current_ms", "mean"),
            mean_current_std=("mean_current_ms", "std"),
            mean_current_sem=("mean_current_ms", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            coverage_area_km2=("coverage_area_km2", "mean"),
            coverage_area_std=("coverage_area_km2", "std"),
            path_km=("path_km", "mean"),
            path_km_std=("path_km", "std"),
            mean_ground_speed_ms=("mean_ground_speed_ms", "mean"),
            current_weighted_coverage=("current_weighted_coverage", "mean"),
            current_weighted_coverage_std=("current_weighted_coverage", "std"),
            n_episode=("storm", "size"),
        )
    )
    summary.to_csv(OUT / "underwater_classical_summary.csv", index=False)

    final = pd.read_csv(DOC / "underwater_final_analysis.csv")
    marl = final[final.method == "Active IPPO"].copy()
    marl = marl.rename(
        columns={
            "mean_current_speed": "mean_current_ms",
            "mean_current_speed_std": "mean_current_std",
            "coverage_area_km2_std": "coverage_area_std",
        }
    )
    marl["method"] = "IPPO"
    marl["n_episode"] = 500
    marl = marl[
        [
            "method", "depth_m", "vmax_ms", "mean_current_ms",
            "mean_current_std", "coverage_area_km2", "coverage_area_std",
            "path_km", "path_km_std", "n_episode",
            "current_weighted_coverage", "current_weighted_coverage_std",
        ]
    ]
    combined = pd.concat([marl, summary], ignore_index=True)
    combined.to_csv(OUT / "underwater_all_methods_summary.csv", index=False)
    return raw, combined


def underwater_paired_tests(raw: pd.DataFrame) -> None:
    rows = []
    metrics = (
        "mean_current_ms",
        "coverage_area_km2",
        "path_km",
        "current_weighted_coverage",
    )
    for (method, depth, speed), base in raw.groupby(
        ["method", "depth_m", "vmax_ms"]
    ):
        learned = learned_episode_metrics(float(depth), float(speed))
        left = base[
            ["storm", "deployment_index", *metrics]
        ].copy()
        paired = left.merge(
            learned,
            on=["storm", "deployment_index"],
            suffixes=("_classical", "_marl"),
        )
        if len(paired) != 500:
            raise SystemExit(
                f"incomplete paired data for {method}, {depth:g} m, {speed:g}: "
                f"{len(paired)} rows"
            )
        for metric in metrics:
            a = paired[f"{metric}_marl"].to_numpy(float)
            b = paired[f"{metric}_classical"].to_numpy(float)
            diff = a - b
            test = stats.ttest_rel(a, b)
            wil = stats.wilcoxon(a, b, zero_method="zsplit")
            rows.append(
                {
                    "depth_m": depth,
                    "vmax_ms": speed,
                    "baseline": method,
                    "metric": metric,
                    "marl_mean": a.mean(),
                    "baseline_mean": b.mean(),
                    "marl_minus_baseline": diff.mean(),
                    "relative_change_pct": 100.0 * diff.mean() / b.mean(),
                    "paired_t": test.statistic,
                    "p_value_t": test.pvalue,
                    "wilcoxon_statistic": wil.statistic,
                    "p_value_wilcoxon": wil.pvalue,
                    "n_pair": len(paired),
                }
            )
    result = pd.DataFrame(rows)
    order = np.argsort(result.p_value_t.to_numpy())
    adjusted = np.empty(len(result))
    running = 0.0
    m = len(result)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * float(result.iloc[idx].p_value_t))
        running = max(running, value)
        adjusted[idx] = running
    result["p_value_holm"] = adjusted
    result.to_csv(OUT / "underwater_pairwise_tests.csv", index=False)


def plot_surface(summary: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(
        len(SPEEDS_SURFACE), len(FLEETS), figsize=(13.5, 6.2),
        sharex=True, sharey=True, constrained_layout=True,
    )
    for row, speed in enumerate(SPEEDS_SURFACE):
        for col, fleet in enumerate(FLEETS):
            ax = axes[row, col]
            for method in (
                "IPPO", "Potential field", "Receding-horizon A*",
                "Lawnmower coverage", "Information greedy",
            ):
                q = summary[
                    (summary.method == method)
                    & (summary.n_auv == fleet)
                    & np.isclose(summary.vmax_ms, speed)
                ].sort_values("radius_km")
                ax.plot(
                    q.radius_km, q.final_distance_km,
                    label=method, linewidth=1.8, markersize=4,
                    **STYLE[method],
                )
            ax.set_title(f"{fleet} SUVs")
            ax.set_xticks(RADII)
            ax.grid(alpha=0.25)
            if col == 0:
                ax.set_ylabel(f"$V_{{max}}$={speed:g} m/s\nFinal error (km)")
            if row == len(SPEEDS_SURFACE) - 1:
                ax.set_xlabel("Deployment radius (km)")
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Surface tracking: MARL versus traditional planners", fontweight="bold")
    fig.savefig(FIG / "classical_surface_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIG / "classical_surface_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def surface_pairwise_tests(summary: pd.DataFrame) -> None:
    rows = []
    for baseline in (
        "Potential field", "Receding-horizon A*",
        "Lawnmower coverage", "Information greedy",
    ):
        for metric in (
            "final_distance_km",
            "distance_avg_km",
            "path_km",
            "mean_speed_ms",
            "on_land_fraction",
        ):
            piv = summary.pivot_table(
                index=["n_auv", "vmax_ms", "radius_km"],
                columns="method",
                values=metric,
            )
            a = piv["IPPO"].to_numpy(float)
            b = piv[baseline].to_numpy(float)
            paired = stats.ttest_rel(a, b)
            wil = stats.wilcoxon(a, b, zero_method="zsplit")
            rows.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "marl_mean": float(a.mean()),
                    "baseline_mean": float(b.mean()),
                    "marl_minus_baseline": float((a - b).mean()),
                    "marl_better_cases": int((a < b).sum()),
                    "n_config": len(a),
                    "paired_t": paired.statistic,
                    "p_value_t": paired.pvalue,
                    "wilcoxon_statistic": wil.statistic,
                    "p_value_wilcoxon": wil.pvalue,
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "surface_pairwise_tests.csv", index=False)


def plot_underwater(summary: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    metrics = [
        ("mean_current_ms", "Mean experienced current (m/s)"),
        ("coverage_area_km2", "Geometric coverage ($10^3$ km$^2$)"),
        ("path_km", "Path per SUV (km)"),
        (
            "current_weighted_coverage",
            "Current-weighted coverage ($10^6$ m s$^{-1}$ km$^2$)",
        ),
    ]
    fig, axes = plt.subplots(
        len(SPEEDS_UW), len(metrics), figsize=(14.2, 6.7),
        sharex=True, constrained_layout=True,
    )
    for row, speed in enumerate(SPEEDS_UW):
        for col, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            for method in (
                "IPPO", "Potential field", "Receding-horizon A*",
                "Lawnmower coverage", "Information greedy",
            ):
                q = summary[
                    (summary.method == method)
                    & np.isclose(summary.vmax_ms, speed)
                ].sort_values("depth_m")
                y = q[metric].to_numpy(float)
                if metric == "coverage_area_km2":
                    y = y / 1000.0
                elif metric == "current_weighted_coverage":
                    y = y / 1.0e6
                ax.plot(
                    q.depth_m, y, label=method, linewidth=1.8,
                    markersize=5, **STYLE[method],
                )
            ax.set_xticks(DEPTHS)
            ax.set_xlabel("Observation depth (m)")
            ax.set_ylabel(label if col == 0 else "")
            ax.set_title(label)
            ax.grid(alpha=0.25)
            ax.text(
                0.02, 0.96, rf"$V_{{\max}}$={speed:g} m/s",
                transform=ax.transAxes, va="top", fontweight="bold",
            )
    axes[0, 0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Subsurface observation: MARL versus traditional planners", fontweight="bold")
    fig.savefig(FIG / "classical_underwater_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIG / "classical_underwater_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary_table(surface: pd.DataFrame, underwater: pd.DataFrame) -> None:
    method_order = (
        "IPPO",
        "Potential field",
        "Receding-horizon A*",
        "Lawnmower coverage",
        "Information greedy",
    )
    s = (
        surface.groupby("method", as_index=False)
        .agg(
            final_error=("final_distance_km", "mean"),
            mean_distance=("distance_avg_km", "mean"),
            path=("path_km", "mean"),
        )
        .set_index("method")
        .loc[list(method_order)]
        .reset_index()
    )
    u = (
        underwater.groupby("method", as_index=False)
        .agg(
            current=("mean_current_ms", "mean"),
            coverage=("coverage_area_km2", "mean"),
            path=("path_km", "mean"),
            weighted_coverage=("current_weighted_coverage", "mean"),
        )
        .set_index("method")
        .loc[list(method_order)]
        .reset_index()
    )
    labels = {
        "IPPO": "Parameter-sharing IPPO",
        "Potential field": "Potential field",
        "Receding-horizon A*": "Receding-horizon A*",
        "Lawnmower coverage": "Lawnmower coverage",
        "Information greedy": "Information greedy",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comparison with four traditional planning baselines on the "
        r"held-out TC segments from the same 500-TC train/test split. Surface "
        r"values are means over the 4/8/12/16-SUV fleets, both speed limits, "
        r"and all five deployment radii. Subsurface values are means over all "
        r"seven depths and both propulsion limits. Smaller errors and paths "
        r"are better; higher current and coverage are better.}",
        r"\label{tab:classical_planners}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\multicolumn{5}{l}{\textbf{Panel A: surface tracking}}\\",
        r"Method & Final error (km) & Mean tracking error (km) & "
        r"\multicolumn{2}{c}{Path per SUV (km)}\\",
        r"\midrule",
    ]
    for _, row in s.iterrows():
        values = (row.final_error, row.mean_distance, row.path)
        best = [abs(row[k] - s[k].min()) < 1e-9 for k in ("final_error", "mean_distance", "path")]
        cells = [f"\\textbf{{{v:.1f}}}" if mark else f"{v:.1f}" for v, mark in zip(values, best)]
        lines.append(
            f"{labels[row.method]} & " + " & ".join(cells[:2])
            + rf" & \multicolumn{{2}}{{c}}{{{cells[2]}}} \\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{5}{l}{\textbf{Panel B: subsurface observation}}\\",
        r"Method & Mean current (m/s) & Geometric coverage (km$^2$) & "
        r"Path (km) & Current-weighted coverage "
        r"($10^3$ m\,s$^{-1}$\,km$^2$)\\",
        r"\midrule",
    ]
    for _, row in u.iterrows():
        weighted = row.weighted_coverage / 1.0e3
        values = (row.current, row.coverage, row.path, weighted)
        best = [
            abs(row[k] - u[k].max()) < 1e-9
            for k in ("current", "coverage", "weighted_coverage")
        ]
        best.append(abs(row.path - u.path.min()) < 1e-9)
        cells = [
            f"\\textbf{{{row.current:.3f}}}" if best[0] else f"{row.current:.3f}",
            f"\\textbf{{{row.coverage:.0f}}}" if best[1] else f"{row.coverage:.0f}",
        ]
        cells.append(
            f"\\textbf{{{values[2]:.1f}}}" if best[3] else f"{values[2]:.1f}"
        )
        cells.append(
            f"\\textbf{{{values[3]:.3f}}}" if best[2] else f"{values[3]:.3f}"
        )
        lines.append(f"{labels[row.method]} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
        "",
    ]
    (DOC / "tables" / "classical_planner_comparison.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    _, surface = aggregate_surface()
    surface_pairwise_tests(surface)
    underwater_raw, underwater = aggregate_underwater()
    underwater_paired_tests(underwater_raw)
    plot_surface(surface)
    plot_underwater(underwater)
    write_summary_table(surface, underwater)
    print(OUT / "surface_all_methods_summary.csv")
    print(OUT / "underwater_all_methods_summary.csv")
    print(FIG / "classical_surface_comparison.pdf")
    print(FIG / "classical_underwater_comparison.pdf")


if __name__ == "__main__":
    main()
