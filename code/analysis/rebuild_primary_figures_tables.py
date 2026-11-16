#!/usr/bin/env python3
"""Rebuild paper-facing results from canonical surface-median and underwater-mean summaries.

Only summaries produced by ``aggregate_primary_median.py`` are accepted.
Legacy 500-TC and episode-arithmetic-mean files are intentionally ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "paper"
MEDIAN = ROOT / "results" / "primary"
OUT = DOC / "generated_primary"


def fmt(value: float, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def method_average(
    frame: pd.DataFrame,
    metric_map: dict[str, tuple[str, int, str]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for group, part in frame.groupby("planner"):
        out[str(group)] = {
            label: float(part[column].mean())
            for label, (column, _digits, _winner) in metric_map.items()
        }
    return out


def write_classical_table() -> Path | None:
    surface_path = MEDIAN / "classical_surface_by_method.csv"
    underwater_path = MEDIAN / "classical_underwater_summary.csv"
    surface_main = MEDIAN / "surface_main_summary.csv"
    underwater_main = MEDIAN / "underwater_ippo_summary.csv"
    if not surface_path.exists() or not underwater_path.exists():
        return None
    surface = pd.read_csv(surface_path).set_index("planner")
    underwater = pd.read_csv(underwater_path)
    uw_method = (
        underwater.groupby("planner", as_index=False)[
            [
                "mean_current_speed_ms_mean",
                "geometric_coverage_km2_mean",
                "current_weighted_coverage_mean",
            ]
        ]
        .mean()
        .set_index("planner")
    )

    display = {
        "PotentialFieldPlanner": "Potential field",
        "RecedingHorizonAStarPlanner": "Receding-horizon A*",
        "BoustrophedonCoveragePlanner": "Lawnmower coverage",
        "InformationGreedyPlanner": "Information greedy",
    }
    order = (
        "IPPO",
        "Potential field",
        "Receding-horizon A*",
        "Lawnmower coverage",
        "Information greedy",
    )
    rows: dict[str, dict[str, float]] = {}
    if surface_main.exists():
        main = pd.read_csv(surface_main)
        rows["IPPO"] = {
            "final": main["final_tracking_error_km_mean"].mean(),
            "wind": main["final_wind_speed_ms_mean"].mean(),
            "path": main["path_km_mean"].mean(),
        }
    if underwater_main.exists():
        main_uw = pd.read_csv(underwater_main)
        rows.setdefault("IPPO", {})
        rows["IPPO"].update(
            {
                "current": main_uw["mean_current_speed_ms_mean"].mean(),
                "coverage": main_uw["geometric_coverage_km2_mean"].mean(),
                "weighted": main_uw["current_weighted_coverage_mean"].mean(),
            }
        )
    for method, row in surface.iterrows():
        label = display[str(method)]
        rows.setdefault(label, {}).update(
            {
                "final": row["final_tracking_error_km_mean"],
                "wind": row["final_wind_speed_ms_mean"],
                "path": row["path_km_mean"],
            }
        )
    for method, row in uw_method.iterrows():
        label = display[str(method)]
        rows.setdefault(label, {}).update(
            {
                "current": row["mean_current_speed_ms_mean"],
                "coverage": row["geometric_coverage_km2_mean"],
                "weighted": row["current_weighted_coverage_mean"],
            }
        )
    if not all(
        {"final", "wind", "current", "coverage", "weighted"} <= set(row)
        for row in rows.values()
    ):
        return None

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Method comparison using the primary 400-TC split. Surface errors use episode medians; subsurface current and coverage metrics use episode means.}",
        r"\label{tab:classical_planners}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"\multicolumn{4}{l}{\textbf{Panel A: surface tracking}}\\",
        r"Method & Final tracking error (km) & Final wind speed (m/s) & "
        r"Path per SUV (km) \\",
        r"\midrule",
    ]
    surface_metrics = ("final", "wind", "path")
    surface_best = {
        metric: (
            min(order, key=lambda name: rows[name][metric])
            if metric != "wind"
            else max(order, key=lambda name: rows[name][metric])
        )
        for metric in surface_metrics
    }
    for method in order:
        row = rows[method]
        values = []
        for metric in surface_metrics:
            value = fmt(row[metric], 2 if metric == "wind" else 1)
            if method == surface_best[metric]:
                value = rf"\textbf{{{value}}}"
            values.append(value)
        lines.append(
            f"{method} & " + " & ".join(values) + r" \\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{4}{l}{\textbf{Panel B: subsurface observation}}\\",
        r"Method & Mean current (m/s) & Geometric coverage (km$^2$) & "
        r"Current-weighted coverage ($10^3$ m\,s$^{-1}$\,km$^2$) \\",
        r"\midrule",
    ]
    underwater_metrics = ("current", "coverage", "weighted")
    underwater_best = {
        metric: max(order, key=lambda name: rows[name][metric])
        for metric in underwater_metrics
    }
    for method in order:
        row = rows[method]
        values = []
        for metric in underwater_metrics:
            digits = 3 if metric != "coverage" else 0
            displayed = row[metric] / 1.0e3 if metric == "weighted" else row[metric]
            value = fmt(displayed, digits)
            if method == underwater_best[metric]:
                value = rf"\textbf{{{value}}}"
            values.append(value)
        lines.append(f"{method} & " + " & ".join(values) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
        "",
    ]
    path = OUT / "classical_planner_comparison_primary.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_formation_table() -> Path | None:
    path = MEDIAN / "surface_formation_summary.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    lines = [
        r"\centering",
        r"\captionof{table}{Surface initial-formation comparison for eight SUVs. "
        r"All four layouts use a mean radius of 100 km. This is a matched "
        r"single-seed sensitivity study using training seed 0, with "
        r"formation-center offsets sampled inside an 800 km radius and "
        r"evaluation without retraining at fixed offsets of 100, 200, 400, "
        r"600, and 800 km on the same 262 held-out TC segments. Final "
        r"tracking error is the fleet-center-to-TC-core distance at the end "
        r"of the episode, and final wind speed is the fleet-mean 10 m wind speed "
        r"at the final positions. "
        r"Entries are medians over validation episodes. Bold entries are the "
        r"best value in each speed/radius row, with smaller final tracking "
        r"error and larger final wind speed preferred. "
        r"Final tracking error and mean radius are in km, and final wind "
        r"speed is in m\,s$^{-1}$.}",
        r"\label{tab:formation_comparison}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{1.1pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ccccccccccc@{}}",
        r"\toprule",
        r"Speed & Train upper bound & Eval radius & \multicolumn{2}{c}{Ring} & "
        r"\multicolumn{2}{c}{Line} & \multicolumn{2}{c}{Grid} & "
        r"\multicolumn{2}{c}{V} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}"
        r"\cmidrule(lr){10-11}",
        r"(m/s) & (km) & (km) & \makecell{Final tracking\\error} & "
        r"\makecell{Final wind\\speed} & "
        r"\makecell{Final tracking\\error} & \makecell{Final wind\\speed} & "
        r"\makecell{Final tracking\\error} & \makecell{Final wind\\speed} & "
        r"\makecell{Final tracking\\error} & \makecell{Final wind\\speed} \\",
        r"\midrule",
    ]
    formations = ("ring", "line", "grid", "v")
    speeds = (12, 20)
    radii = (100, 200, 400, 600, 800)
    for speed_index, speed in enumerate(speeds):
        for radius in radii:
            part = frame[
                (frame["vmax_ms"] == speed)
                & (frame["radius_km"] == radius)
            ].set_index("formation")
            best_final = part["final_tracking_error_km_mean"].idxmin()
            best_wind = part["final_wind_speed_ms_mean"].idxmax()
            cells = [str(speed), "800", str(radius)]
            for formation in formations:
                final = fmt(
                    part.loc[formation, "final_tracking_error_km_mean"], 1
                )
                wind = fmt(
                    part.loc[formation, "final_wind_speed_ms_mean"], 2
                )
                if formation == best_final:
                    final = rf"\textbf{{{final}}}"
                if formation == best_wind:
                    wind = rf"\textbf{{{wind}}}"
                cells.extend((final, wind))
            lines.append(" & ".join(cells) + r" \\")
        if speed_index + 1 < len(speeds):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular*}", ""]
    out = OUT / "surface_formation_primary.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_underwater_table() -> Path | None:
    path = MEDIAN / "underwater_ippo_summary.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{IPPO subsurface metrics using episode means averaged over ten seeds. Geometric coverage is computed on unique native HYCOM cells of \(0.04^\circ\) latitude by \(0.08^\circ\) longitude. Current-weighted coverage is reported in \(10^3\) m\,s$^{-1}$\,km$^2$.}",
        r"\label{tab:underwater_ippo_main}",
        r"\small",
        r"\begin{tabular}{@{}rrrrr@{}}",
        r"\toprule",
        r"Depth (m) & $V_{\max}$ (m/s) & Current (m/s) & Geometric coverage (km$^2$) & Current-weighted coverage \\",
        r"\midrule",
    ]
    for _, row in frame.sort_values(["depth_m", "vmax_ms"]).iterrows():
        lines.append(
            f"{fmt(row.depth_m, 0)} & {fmt(row.vmax_ms, 1)} & "
            f"{fmt(row.mean_current_speed_ms_mean, 3)} & "
            f"{fmt(row.geometric_coverage_km2_mean, 0)} & "
            f"{fmt(row.current_weighted_coverage_mean / 1.0e3, 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    out = OUT / "underwater_ippo_primary.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_filter_table() -> Path | None:
    path = MEDIAN / "surface_filter_summary.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Execution-layer action low-pass filtering for the 12 m/s "
        r"surface fleets at a fixed 400 km deployment offset. For each fleet "
        r"size and each of ten training seeds, the same frozen policy is "
        r"evaluated without filtering (\(\beta=0\)) and with filtering "
        r"(\(\beta=0.8\)) on the same 262 open-ocean validation episodes. "
        r"Entries are medians over validation episodes within each training "
        r"seed, then arithmetic means over the ten training seeds.}",
        r"\label{tab:surface_control_ablations}",
        r"\small",
        r"\begin{tabular}{@{}rrrrrrr@{}}",
        r"\toprule",
        r"Fleet & $\beta$ & Final error (km) & Final wind speed (m/s) & "
        r"Path (km) & Mean speed (m/s) & Maximum speed (m/s) \\",
        r"\midrule",
    ]
    metric_specs = (
        ("final_tracking_error_km_mean", 1, True),
        ("final_wind_speed_ms_mean", 2, False),
        ("path_km_mean", 1, True),
        ("mean_speed_ms_mean", 2, True),
        ("max_speed_ms_mean", 2, True),
    )
    for n_auv, part in frame.sort_values(["n_auv", "filter_beta"]).groupby(
        "n_auv", sort=True
    ):
        best = {
            column: (
                part[column].min() if lower_is_better else part[column].max()
            )
            for column, _digits, lower_is_better in metric_specs
        }
        for _, row in part.iterrows():
            cells = [str(int(n_auv)), f"{row.filter_beta:g}"]
            for column, digits, _lower_is_better in metric_specs:
                value = fmt(row[column], digits)
                if abs(float(row[column]) - float(best[column])) < 1e-12:
                    value = rf"\textbf{{{value}}}"
                cells.append(value)
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    out = OUT / "surface_filter_primary.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_planner_stratified() -> Path | None:
    surface_classical_path = MEDIAN / "classical_surface_summary.csv"
    surface_ippo_path = MEDIAN / "surface_main_summary.csv"
    underwater_classical_path = MEDIAN / "classical_underwater_summary.csv"
    underwater_ippo_path = MEDIAN / "underwater_ippo_summary.csv"
    required = (
        surface_classical_path,
        surface_ippo_path,
        underwater_classical_path,
        underwater_ippo_path,
    )
    if not all(path.exists() for path in required):
        return None
    surface_classical = pd.read_csv(surface_classical_path)
    surface_ippo = pd.read_csv(surface_ippo_path)
    underwater_classical = pd.read_csv(underwater_classical_path)
    underwater_ippo = pd.read_csv(underwater_ippo_path)

    planner_short = {
        "PotentialFieldPlanner": "PF",
        "RecedingHorizonAStarPlanner": "A*",
        "BoustrophedonCoveragePlanner": "Lawn",
        "InformationGreedyPlanner": "Greedy",
    }
    names = ("IPPO", "PF", "A*", "Lawn", "Greedy")

    def cells(
        rows: dict[str, dict[str, float]],
        metric: str,
        digits: int,
        *,
        lower_is_better: bool = False,
        scale: float = 1.0,
    ) -> str:
        values = {name: float(rows[name][metric]) / scale for name in names}
        best = min(values.values()) if lower_is_better else max(values.values())
        out = []
        for name in names:
            value = values[name]
            text = fmt(value, digits)
            out.append(
                rf"\textbf{{{text}}}"
                if abs(value - best) < 1e-12
                else text
            )
        return " & ".join(out)

    lines = [
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\begin{longtable}{@{\extracolsep{\fill}}lllrrrrr@{}}",
        r"\caption{Planner-stratified primary results reported separately for the two "
        r"surface speed limits and the two subsurface propulsion limits. Surface entries "
        r"use episode medians averaged across IPPO seeds, or one episode median for "
        r"deterministic planners. Subsurface entries use episode means. Geometric "
        r"coverage counts unique native HYCOM cells of \(0.04^\circ\) latitude by "
        r"\(0.08^\circ\) longitude. Current-weighted coverage is reported in "
        r"\(10^3\) m\,s$^{-1}$\,km$^2$.}",
        r"\label{tab:planner_stratified}\\",
        r"\toprule",
        r"\multicolumn{8}{l}{\textbf{Panel A: surface tracking}}\\",
        r"Speed (m/s) & Radius & Metric & " + " & ".join(names) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{Planner-stratified primary results (continued).}\\",
        r"\toprule",
        r"Speed (m/s) & Radius/depth & Metric & " + " & ".join(names) + r" \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for speed in (12.0, 20.0):
        for radius in (100, 200, 400, 600, 800):
            rows: dict[str, dict[str, float]] = {}
            q = surface_ippo[
                (surface_ippo.vmax_ms == speed)
                & (surface_ippo.radius_km == radius)
            ]
            rows["IPPO"] = {
                "final": q.final_tracking_error_km_mean.mean(),
                "wind": q.final_wind_speed_ms_mean.mean(),
            }
            qc = surface_classical[
                (surface_classical.vmax_ms == speed)
                & (surface_classical.radius_km == radius)
            ]
            for planner, part in qc.groupby("planner"):
                rows[planner_short[str(planner)]] = {
                    "final": part.final_tracking_error_km_mean.mean(),
                    "wind": part.final_wind_speed_ms_mean.mean(),
                }
            lines.append(
                f"{speed:g} & {radius} km & Final tracking error (km) & "
                + cells(rows, "final", 1, lower_is_better=True)
                + r" \\"
            )
            lines.append(
                f"{speed:g} & {radius} km & Final mean wind speed (m/s) & "
                + cells(rows, "wind", 2)
                + r" \\"
            )
            if radius != 800:
                lines.append(r"\addlinespace[1.0pt]")

    lines += [
        r"\midrule",
        r"\multicolumn{8}{l}{\textbf{Panel B: subsurface observation}}\\",
        r"\midrule",
    ]
    for speed in (1.5, 2.0):
        for depth in (0, 10, 20, 30, 40, 50, 60):
            rows = {}
            qi = underwater_ippo[
                (underwater_ippo.vmax_ms == speed)
                & (underwater_ippo.depth_m == depth)
            ]
            rows["IPPO"] = {
                "current": qi.mean_current_speed_ms_mean.mean(),
                "coverage": qi.geometric_coverage_km2_mean.mean(),
                "weighted": qi.current_weighted_coverage_mean.mean(),
            }
            qc = underwater_classical[
                (underwater_classical.vmax_ms == speed)
                & (underwater_classical.depth_m == depth)
            ]
            for planner, part in qc.groupby("planner"):
                rows[planner_short[str(planner)]] = {
                    "current": part.mean_current_speed_ms_mean.mean(),
                    "coverage": part.geometric_coverage_km2_mean.mean(),
                    "weighted": part.current_weighted_coverage_mean.mean(),
                }
            for metric, label, digits, scale in (
                ("current", "Mean experienced current (m/s)", 3, 1.0),
                ("coverage", "Geometric coverage (km$^2$)", 0, 1.0),
                ("weighted", "Current-weighted coverage", 3, 1.0e3),
            ):
                lines.append(
                    f"{speed:.1f} & {depth} m & {label} & "
                    + cells(rows, metric, digits, scale=scale)
                    + r" \\"
                )
            if depth != 60:
                lines.append(r"\addlinespace[1.0pt]")
        if speed == 1.5:
            lines.append(r"\midrule")

    lines += [
        r"\end{longtable}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\normalsize",
    ]
    out = OUT / "planner_stratified_primary.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_algorithm_table() -> Path | None:
    surface_path = MEDIAN / "surface_algorithm_summary.csv"
    underwater_path = MEDIAN / "underwater_algorithms_summary.csv"
    if not surface_path.exists() or not underwater_path.exists():
        return None
    surface = (
        pd.read_csv(surface_path)
        .groupby("algorithm", as_index=False)[
            [
                "final_tracking_error_km_mean",
                "path_km_mean",
            ]
        ]
        .mean()
        .set_index("algorithm")
    )
    underwater = (
        pd.read_csv(underwater_path)
        .groupby("algorithm", as_index=False)[
            [
                "mean_current_speed_ms_mean",
                "geometric_coverage_km2_mean",
                "current_weighted_coverage_mean",
            ]
        ]
        .mean()
        .set_index("algorithm")
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Representative algorithm comparison under the primary 400-TC setup. Surface metrics use seed medians; subsurface metrics use seed means.}",
        r"\small",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Algorithm & Surface final error & Surface path & Underwater current & Underwater coverage & Weighted coverage \\",
        r"\midrule",
    ]
    for algorithm in ("IPPO", "SAC", "TD3"):
        s = surface.loc[algorithm]
        u = underwater.loc[algorithm]
        lines.append(
            f"{algorithm} & {fmt(s.final_tracking_error_km_mean, 1)} & "
            f"{fmt(s.path_km_mean, 1)} & "
            f"{fmt(u.mean_current_speed_ms_mean, 3)} & "
            f"{fmt(u.geometric_coverage_km2_mean, 0)} & "
            f"{fmt(u.current_weighted_coverage_mean, 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    out = OUT / "algorithm_comparison_primary.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "classical": write_classical_table(),
        "formation": write_formation_table(),
        "underwater": write_underwater_table(),
        "filter": write_filter_table(),
        "planner_stratified": write_planner_stratified(),
        "algorithm": write_algorithm_table(),
    }
    report = {
        key: str(path) if path is not None else "waiting_for_complete_primary_median_summary"
        for key, path in outputs.items()
    }
    (OUT / "build_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
