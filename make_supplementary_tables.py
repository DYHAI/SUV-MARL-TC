#!/usr/bin/env python3
"""Build manuscript tables from completed supplementary experiment outputs."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "Latex_Yihui_MARL" / "tables"
ROBUST = ROOT / "robustness_eval"
ALG = ROOT / "ppo_algorithm_comparison"


def _algorithm(model: str) -> str:
    lower = model.lower()
    if "_sac_" in lower:
        return "SAC"
    if "_td3_" in lower:
        return "TD3"
    return "IPPO"


def _seed(model: str) -> int:
    match = re.search(r"_s(\d+)\.zip$", model)
    return int(match.group(1)) if match else 0


def _log_wall_seconds(path: Path) -> float:
    """Return active wall time from the timestamped progress lines."""
    stamps = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"\[(\d\d:\d\d:\d\d)\]", line)
        if match:
            stamps.append(dt.datetime.strptime(match.group(1), "%H:%M:%S"))
    if len(stamps) < 2:
        raise ValueError(f"not enough timestamped progress lines: {path}")
    seconds = (stamps[-1] - stamps[0]).total_seconds()
    return seconds + 86400.0 if seconds < 0 else seconds


def algorithm_stage(path: Path, metrics: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["algorithm"] = raw["model"].map(_algorithm)
    raw["seed"] = raw["model"].map(_seed)
    per_model = raw.groupby(
        ["algorithm", "seed", "model"], as_index=False
    )[metrics].mean()
    named = {}
    for metric in metrics:
        named[f"{metric}_mean"] = (metric, "mean")
        named[f"{metric}_std"] = (metric, "std")
    return per_model.groupby("algorithm", as_index=False).agg(**named)


def write_algorithm_table() -> None:
    surface = algorithm_stage(
        ROBUST / "algorithm_surface_eval.csv",
        ["final_distance_km", "distance_avg_km", "path_km"],
    )
    underwater = algorithm_stage(
        ROBUST / "algorithm_underwater_eval.csv",
        ["mean_current_ms", "coverage_area_km2", "path_km", "command_proxy"],
    )
    underwater["current_weighted_coverage"] = (
        underwater["mean_current_ms_mean"]
        * underwater["coverage_area_km2_mean"]
    )
    underwater["current_weighted_coverage_std"] = np.nan
    train_files = sorted(ALG.glob("*_training.csv"))
    training = pd.concat([pd.read_csv(p) for p in train_files], ignore_index=True)
    training["algorithm"] = training["algorithm"].str.upper()
    if "max_process_tree_rss_gib" not in training:
        training["max_process_tree_rss_gib"] = np.nan
    ippo_runtime = []
    for task, pattern in (
        ("surface", "logs_vmax12_seed10/train_8au_s{seed}.log"),
        (
            "underwater",
            "logs_uwflow_7depth/"
            "train_ppo_uwflow_500_d30_v1.5_rC_20h_sb3_s{seed}.log",
        ),
    ):
        seconds = [
            _log_wall_seconds(ROOT / pattern.format(seed=seed))
            for seed in (0, 1)
        ]
        ippo_runtime.append(
            {
                "task": task,
                "algorithm": "IPPO",
                "wall_mean": float(np.mean(seconds)),
                "wall_std": float(np.std(seconds, ddof=1)),
                "rss_mean": np.nan,
                "gpu_mean": np.nan,
                "seeds": 2,
            }
        )
    runtime = pd.concat(
        [
            training.groupby(["task", "algorithm"], as_index=False).agg(
                wall_mean=("elapsed_s", "mean"),
                wall_std=("elapsed_s", "std"),
                rss_mean=("max_process_tree_rss_gib", "mean"),
                gpu_mean=("gpu_max_allocated_gib", "mean"),
                seeds=("seed", "nunique"),
            ),
            pd.DataFrame(ippo_runtime),
        ],
        ignore_index=True,
    ).sort_values(["task", "algorithm"])
    runtime.to_csv(ROBUST / "algorithm_runtime.csv", index=False)
    order = ("IPPO", "SAC", "TD3")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Algorithm comparison of parameter-sharing IPPO, SAC, and TD3 "
        r"on representative surface and subsurface tasks. Each algorithm uses "
        r"two independent training seeds and $2\times10^6$ environment steps "
        r"per run. Values are mean $\pm$ sample standard deviation across "
        r"training seeds.}",
        r"\label{tab:algorithm_comparison}",
        r"\footnotesize",
        r"\begin{minipage}[t]{0.49\textwidth}",
        r"\centering\textbf{(a) Surface: 8 SUVs, 12 m/s, 400 km}\\",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule Algorithm & Final error (km) & Avg. distance (km) & Path (km)\\",
        r"\midrule",
    ]
    for alg in order:
        row = surface[surface.algorithm == alg].iloc[0]
        lines.append(
            f"{alg} & {row.final_distance_km_mean:.1f}$\\pm${row.final_distance_km_std:.1f} & "
            f"{row.distance_avg_km_mean:.1f}$\\pm${row.distance_avg_km_std:.1f} & "
            f"{row.path_km_mean:.1f}$\\pm${row.path_km_std:.1f} \\\\"
        )
    lines += [
        r"\bottomrule\end{tabular}",
        r"\end{minipage}\hfill",
        r"\begin{minipage}[t]{0.49\textwidth}",
        r"\centering\textbf{(b) Subsurface: 30 m, 1.5 m/s}\\",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule Algorithm & Current (m/s) & Coverage (km$^2$) & Path (km)\\",
        r"\midrule",
    ]
    for alg in order:
        row = underwater[underwater.algorithm == alg].iloc[0]
        lines.append(
            f"{alg} & {row.mean_current_ms_mean:.3f}$\\pm${row.mean_current_ms_std:.3f} & "
            f"{row.coverage_area_km2_mean:.0f}$\\pm${row.coverage_area_km2_std:.0f} & "
            f"{row.path_km_mean:.1f}$\\pm${row.path_km_std:.1f} \\\\"
        )
    lines += [
        r"\bottomrule\end{tabular}",
        r"\end{minipage}",
        r"\par\smallskip",
        r"\parbox{0.96\textwidth}{\footnotesize\textbf{Training resources "
        r"(two representative seeds):} "
    ]
    resource_parts = []
    for _, row in runtime.iterrows():
        text = f"{row.task}/{row.algorithm}: {row.wall_mean / 60.0:.1f} min"
        if pd.notna(row.rss_mean):
            text += f", {row.rss_mean:.1f} GiB peak process tree"
        resource_parts.append(text)
    lines[-1] += "; ".join(resource_parts) + ".}"
    lines += [r"\end{table*}", ""]
    (TABLES / "algorithm_comparison.tex").write_text("\n".join(lines), encoding="utf-8")
    surface.to_csv(ROBUST / "algorithm_surface_summary.csv", index=False)
    underwater.to_csv(ROBUST / "algorithm_underwater_summary.csv", index=False)


def write_formation_table() -> None:
    # Keep the manuscript generator synchronized with the complete
    # speed x formation x deployment-radius matrix.
    from aggregate_formation_full_matrix import load_and_summarize, write_table

    summary = load_and_summarize()
    summary.to_csv(ROBUST / "formation_full_matrix_summary.csv", index=False)
    write_table(summary)


def write_ablation_table() -> None:
    ablation = pd.read_csv(ROBUST / "underwater_reward_ablation.csv")

    def reward_mode(model: str) -> str:
        if "_rA_" in model:
            return "Current only (A)"
        if "_rB_" in model:
            return "Current + effort + coast (B)"
        if "_rC_" in model:
            return "B + alignment (C)"
        return "Other"

    ablation["reward"] = ablation["model"].map(reward_mode)
    ablation["seed"] = ablation["model"].map(_seed)
    per_model = ablation.groupby(
        ["reward", "seed", "model"], as_index=False
    )[
        ["mean_current_ms", "coverage_area_km2", "path_km", "command_proxy"]
    ].mean()
    reward = per_model.groupby("reward", as_index=False).agg(
        mean_current_ms_mean=("mean_current_ms", "mean"),
        mean_current_ms_std=("mean_current_ms", "std"),
        coverage_area_km2_mean=("coverage_area_km2", "mean"),
        coverage_area_km2_std=("coverage_area_km2", "std"),
        path_km_mean=("path_km", "mean"),
        path_km_std=("path_km", "std"),
        command_proxy_mean=("command_proxy", "mean"),
        command_proxy_std=("command_proxy", "std"),
    )
    safety_on = pd.read_csv(ROBUST / "surface_safety_on.csv")
    safety_off = pd.read_csv(ROBUST / "surface_safety_off.csv")
    safety = pd.DataFrame(
        [
            {
                "variant": "Safety on",
                "final_error": safety_on.final_distance_km.mean(),
                "path": safety_on.path_km.mean(),
                "land": safety_on.on_land_fraction.mean(),
            },
            {
                "variant": "Safety off",
                "final_error": safety_off.final_distance_km.mean(),
                "path": safety_off.path_km.mean(),
                "land": safety_off.on_land_fraction.mean(),
            },
        ]
    )
    reward.to_csv(ROBUST / "underwater_reward_ablation_summary.csv", index=False)
    safety.to_csv(ROBUST / "surface_safety_summary.csv", index=False)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Component ablations. Panel A varies underwater reward terms "
        r"with three training seeds. Panel B removes the surface coastline safety "
        r"filter without retraining; the action-filter ablation is reported "
        r"separately in Table~\ref{tab:surface_control_ablations}.}",
        r"\label{tab:component_ablations}",
        r"\small",
        r"\textbf{(a) Underwater reward terms}\\",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule Reward & Current (m/s) & Coverage (km$^2$) & Path (km) & Effort\\",
        r"\midrule",
    ]
    for label in ("Current only (A)", "Current + effort + coast (B)", "B + alignment (C)"):
        row = reward[reward.reward == label].iloc[0]
        lines.append(
            f"{label} & {row.mean_current_ms_mean:.3f}$\\pm${row.mean_current_ms_std:.3f} & "
            f"{row.coverage_area_km2_mean:.0f}$\\pm${row.coverage_area_km2_std:.0f} & "
            f"{row.path_km_mean:.1f}$\\pm${row.path_km_std:.1f} & "
            f"{row.command_proxy_mean:.3f}$\\pm${row.command_proxy_std:.3f} \\\\"
        )
    lines += [
        r"\bottomrule\end{tabular}",
        r"\par\medskip",
        r"\textbf{(b) Surface coastline safety layer}\\",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule Variant & Final error (km) & Path (km) & Land fraction\\",
        r"\midrule",
    ]
    for _, row in safety.iterrows():
        lines.append(
            f"{row.variant} & {row.final_error:.1f} & {row.path:.1f} & "
            f"{row.land * 100:.3f}\\% \\\\"
        )
    lines += [r"\bottomrule\end{tabular}", r"\end{table*}", ""]
    (TABLES / "ablations.tex").write_text("\n".join(lines), encoding="utf-8")


def write_stratified_planner_table() -> None:
    surface = pd.read_csv(ROOT / "classical_eval" / "surface_all_methods_summary.csv")
    underwater = pd.read_csv(ROOT / "classical_eval" / "underwater_all_methods_summary.csv")
    methods = (
        "IPPO",
        "Potential field",
        "Receding-horizon A*",
        "Lawnmower coverage",
        "Information greedy",
    )
    short = {
        "IPPO": "IPPO",
        "Potential field": "PF",
        "Receding-horizon A*": "A*",
        "Lawnmower coverage": "Lawn",
        "Information greedy": "Greedy",
    }
    surface_final = (
        surface.groupby(["radius_km", "method"])["final_distance_km"]
        .mean()
        .unstack("method")
    )
    surface_average = (
        surface.groupby(["radius_km", "method"])["distance_avg_km"]
        .mean()
        .unstack("method")
    )
    by_depth = underwater[underwater.depth_m % 10 == 0]
    depth_metrics = {
        metric: by_depth.groupby(["depth_m", "method"])[metric]
        .mean()
        .unstack("method")
        for metric in ("mean_current_ms", "coverage_area_km2", "path_km")
    }

    def metric_cells(row: pd.Series, digits: int, winner: str) -> str:
        target = row.min() if winner == "min" else row.max()
        cells = []
        for method in methods:
            value = float(row[method])
            text = f"{value:.{digits}f}"
            cells.append(
                rf"\textbf{{{text}}}"
                if np.isclose(value, target, rtol=0.0, atol=1e-9)
                else text
            )
        return " & ".join(cells)

    lines = [
        r"\begin{center}",
        r"\captionof{table}{Distance and observation metrics for the traditional "
        r"planner comparison on the held-out TC segments. Panel A reports the "
        r"two surface distance metrics by deployment radius, averaged over the "
        r"four fleet sizes and both speed limits. Panel B reports mean "
        r"experienced current speed, nominal coverage, and path per SUV at each "
        r"depth, averaged over the two propulsion limits.}",
        r"\label{tab:planner_stratified}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrr@{}}",
        r"\toprule",
        r"\multicolumn{7}{l}{\textbf{Panel A: surface distance metrics}}\\",
        r"Radius & Metric & " + " & ".join(short[m] for m in methods) + r"\\",
        r"\midrule",
    ]
    for radius in surface_final.index:
        final_cells = metric_cells(surface_final.loc[radius], 1, "min")
        average_cells = metric_cells(surface_average.loc[radius], 1, "min")
        lines.append(f"{int(radius)} km & Final error (km) & {final_cells} \\\\")
        lines.append(f" & Time-averaged distance (km) & {average_cells} \\\\")
        lines.append(r"\addlinespace[1.5pt]")
    lines += [
        r"\midrule",
        r"\multicolumn{7}{l}{\textbf{Panel B: subsurface metrics by depth}}\\",
        r"Depth & Metric & " + " & ".join(short[m] for m in methods) + r"\\",
        r"\midrule",
    ]
    metric_rows = (
        ("Mean experienced current (m/s)", "mean_current_ms", 3),
        ("Nominal coverage (km$^2$)", "coverage_area_km2", 0),
        ("Path per SUV (km)", "path_km", 1),
    )
    for depth in depth_metrics["mean_current_ms"].index:
        for index, (label, metric, digits) in enumerate(metric_rows):
            cells = metric_cells(
                depth_metrics[metric].loc[depth],
                digits,
                "min" if metric == "path_km" else "max",
            )
            depth_label = f"{int(depth)} m" if index == 0 else ""
            lines.append(f"{depth_label} & {label} & {cells} \\\\")
        if depth != depth_metrics["mean_current_ms"].index[-1]:
            lines.append(r"\addlinespace[1.5pt]")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{center}", ""]
    (TABLES / "planner_stratified.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    write_algorithm_table()
    write_formation_table()
    write_ablation_table()
    write_stratified_planner_table()
    print("supplementary tables written")


if __name__ == "__main__":
    main()
