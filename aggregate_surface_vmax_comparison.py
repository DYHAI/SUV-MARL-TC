"""Aggregate ten-seed surface experiments at 12 and 20 m/s.

Each ROW record is first computed over all 768 held-out segments. The ten
seed-level records are then summarized with the sample standard deviation
(ddof=1), so the reported uncertainty is training-seed variability rather than
the much smaller segment-level SEM.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "Latex_Yihui_MARL" / "figures"
TABLEDIR = ROOT / "Latex_Yihui_MARL" / "tables"
FLEETS = (4, 8, 12, 16)
SEEDS = tuple(range(10))
RADII = (100, 200, 400, 600, 800)
FIELDS = {
    "d_nominal": 0,
    "d_actual": 1,
    "n_auv": 2,
    "distance": 3,
    "path": 5,
    "mean_speed": 7,
    "duration_h": 9,
    "on_land_fraction": 10,
    "touched_episode_fraction": 11,
}


def row_path(version: int, n_auv: int, seed: int, distance: int) -> Path:
    if version == 12:
        return (
            ROOT
            / "logs_vmax12_seed10"
            / f"eval_n{n_auv}_s{seed}_d{distance}.log"
        )
    if n_auv == 16:
        return (
            ROOT
            / "logs_seed16"
            / f"eval_n16_s{seed}_d{distance}.log"
        )
    if seed == 0:
        return (
            ROOT
            / "logs_xfer"
            / f"xfer_r800_windonly_n{n_auv}_d{distance}.log"
        )
    return (
        ROOT
        / "logs_seed_extra"
        / f"eval_n{n_auv}_s{seed}_d{distance}.log"
    )


def parse_row(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    match = re.search(r"^ROW\|([^\n]+)$", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"ROW record not found: {path}")
    values = [float(x) for x in match.group(1).split("|")]
    if len(values) != 16:
        raise ValueError(f"Unexpected ROW field count in {path}: {len(values)}")
    return {key: values[index] for key, index in FIELDS.items()}


def collect() -> tuple[pd.DataFrame, list[Path]]:
    rows: list[dict[str, float]] = []
    missing: list[Path] = []
    for version in (12, 20):
        for n_auv in FLEETS:
            for seed in SEEDS:
                for distance in RADII:
                    path = row_path(version, n_auv, seed, distance)
                    if not path.exists():
                        missing.append(path)
                        continue
                    try:
                        values = parse_row(path)
                    except ValueError:
                        missing.append(path)
                        continue
                    rows.append(
                        {
                            "version_ms": version,
                            "fleet": n_auv,
                            "seed": seed,
                            "distance_target_km": distance,
                            **values,
                        }
                    )
    return pd.DataFrame(rows), missing


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["version_ms", "fleet", "distance_target_km"]
    metrics = ["distance", "path", "mean_speed", "on_land_fraction"]
    mean = raw.groupby(keys, as_index=False)[metrics].mean()
    std = raw.groupby(keys, as_index=False)[metrics].std(ddof=1)
    count = raw.groupby(keys, as_index=False)[metrics].count()
    out = mean.merge(std, on=keys, suffixes=("", "_std"))
    out = out.merge(count, on=keys, suffixes=("", "_n"))
    return out


def paired_tests(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n_auv in FLEETS:
        for distance in RADII:
            a = raw[
                (raw.version_ms == 12)
                & (raw.fleet == n_auv)
                & (raw.distance_target_km == distance)
            ].sort_values("seed")
            b = raw[
                (raw.version_ms == 20)
                & (raw.fleet == n_auv)
                & (raw.distance_target_km == distance)
            ].sort_values("seed")
            merged = a.merge(b, on="seed", suffixes=("_12", "_20"))
            if len(merged) != 10:
                continue
            for metric in ("distance", "path", "mean_speed", "on_land_fraction"):
                x12 = merged[f"{metric}_12"].to_numpy(float)
                x20 = merged[f"{metric}_20"].to_numpy(float)
                test = stats.ttest_rel(x12, x20)
                diff = x12 - x20
                rows.append(
                    {
                        "fleet": n_auv,
                        "distance_target_km": distance,
                        "metric": metric,
                        "mean_12": x12.mean(),
                        "mean_20": x20.mean(),
                        "delta_12_minus_20": diff.mean(),
                        "delta_std": diff.std(ddof=1),
                        "t_stat": test.statistic,
                        "p_value": test.pvalue,
                        "n_seed": len(merged),
                    }
                )
    return pd.DataFrame(rows)


def fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def make_table(summary: pd.DataFrame, metrics: list[str], labels: list[str],
               digits: list[int], caption: str, label: str, path: Path) -> None:
    lines = [
        r"\begin{table*}[p]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Metric & $V_{\max}$ (m/s) & 100 km & 200 km & 400 km & 600 km & 800 km \\",
        r"\midrule",
    ]
    for metric_index, (metric, label_text, nd) in enumerate(
        zip(metrics, labels, digits)
    ):
        label_text = label_text.replace("%", r"\%")
        value_scale = 100.0 if metric == "on_land_fraction" else 1.0
        if metric_index:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textbf{{{label_text}}}}} \\")
        for version in (20, 12):
            for fleet in FLEETS:
                cells = []
                for distance in RADII:
                    row = summary[
                        (summary.version_ms == version)
                        & (summary.fleet == fleet)
                        & (summary.distance_target_km == distance)
                    ].iloc[0]
                    cells.append(
                        rf"{fmt(row[metric] * value_scale, nd)}$\pm$"
                        rf"{fmt(row[metric + '_std'] * value_scale, nd)}"
                    )
                lines.append(
                    rf"{fleet}-SUV & {version} & " + " & ".join(cells) + r" \\"
                )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\parbox{\textwidth}{\footnotesize Values are means $\pm$ sample "
        r"standard deviations across ten independently trained seeds. Each "
        r"seed-level value is first averaged over the 768 held-out segments.}",
        r"\end{table*}",
        "",
    ]
    path.write_text("\n".join(lines))


def plot_metrics(summary: pd.DataFrame, path_pdf: Path, path_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    )
    fig, axes = plt.subplots(
        4, 4, figsize=(13.0, 11.0), sharex=True, constrained_layout=True
    )
    metrics = [
        ("distance", "Tracking distance (km)", 1.0),
        ("path", "Path length per SUV (km)", 1.0),
        ("mean_speed", "Mean speed (m s$^{-1}$)", 1.0),
        ("on_land_fraction", "On-land fraction (%)", 100.0),
    ]
    style = {
        20: dict(color="#226aa6", marker="o"),
        12: dict(color="#d9791f", marker="s"),
    }
    for row, fleet in enumerate(FLEETS):
        for col, (metric, ylabel, scale) in enumerate(metrics):
            ax = axes[row, col]
            for version in (20, 12):
                d = summary[
                    (summary.fleet == fleet)
                    & (summary.version_ms == version)
                ].sort_values("distance_target_km")
                ax.errorbar(
                    d.distance_target_km,
                    d[metric] * scale,
                    yerr=d[f"{metric}_std"] * scale,
                    capsize=3,
                    lw=1.6,
                    elinewidth=0.9,
                    label=rf"$V_{{\max}}$={version} m/s",
                    **style[version],
                )
            ax.set_title(rf"$N$={fleet}")
            if row == 3:
                ax.set_xlabel("Deployment radius (km)")
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.set_xticks(RADII)
            if row == 0 and col == 0:
                ax.legend(frameon=False, loc="best")
    fig.suptitle(
        "Ten-seed surface comparison: 12 versus 20 m/s",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_numeric_results(
    summary: pd.DataFrame, tests: pd.DataFrame, path: Path
) -> None:
    paired = tests[tests.metric == "distance"].copy()
    better = int((paired.delta_12_minus_20 > 0).sum())
    sig = paired[paired.p_value < 0.05]
    sig_20 = int((sig.delta_12_minus_20 > 0).sum())
    sig_12 = int((sig.delta_12_minus_20 < 0).sum())
    by_radius = (
        summary.groupby(["version_ms", "distance_target_km"])[
            ["distance", "path", "mean_speed", "on_land_fraction"]
        ]
        .mean()
        .reset_index()
    )
    lines = []
    pct_lines = []
    for distance in RADII:
        q = by_radius[by_radius.distance_target_km == distance].set_index(
            "version_ms"
        )
        delta_d = q.loc[12, "distance"] - q.loc[20, "distance"]
        delta_p = q.loc[12, "path"] - q.loc[20, "path"]
        delta_v = q.loc[12, "mean_speed"] - q.loc[20, "mean_speed"]
        pct_d = 100.0 * delta_d / q.loc[20, "distance"]
        lines.append(
            f"{distance}~km: "
            f"$\\Delta d={delta_d:+.1f}$~km, "
            f"$\\Delta L={delta_p:+.1f}$~km, "
            f"$\\Delta \\bar v={delta_v:+.2f}$~m/s"
        )
        pct_lines.append(f"{distance}~km: {pct_d:+.1f}\\%")
    median_delta = float(
        paired.delta_12_minus_20.median()
    )
    text = [
        "Across the twenty matched fleet-size/radius comparisons, the 20 m/s "
        f"policy has the smaller mean tracking distance in {better} cases. "
        f"The median paired difference "
        f"$(d_{{12}}-d_{{20}})$ is {median_delta:+.1f} km. "
        f"Paired tests identify {sig_20} cases in which 20 m/s is significantly "
        f"better and {sig_12} cases in which 12 m/s is significantly better "
        r"at $p<0.05$.",
        "Averaged over the four fleet sizes, the differences "
        "(12 m/s minus 20 m/s) at the five radii are: "
        + "; ".join(lines)
        + ". Here, positive tracking distance and negative path length mean "
        "that the 12 m/s policy is farther from the center and travels less; "
        "negative mean-speed differences show the direct propulsion penalty.",
        "The relative tracking-distance penalty of the 12 m/s policy is "
        + "; ".join(pct_lines)
        + ". The absolute penalty therefore grows with the deployment radius, "
        "consistent with the finite time available to remove forecast-induced "
        "displacement. The shorter 12 m/s paths reduce propulsion demand, but "
        "that energy saving comes at the cost of poorer TC-center tracking.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(text) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="write partial diagnostics instead of requiring all 400 evaluations",
    )
    args = parser.parse_args()

    raw, missing = collect()
    expected = 2 * len(FLEETS) * len(SEEDS) * len(RADII)
    print(f"Parsed {len(raw)}/{expected} seed-level evaluations")
    if missing:
        print(f"Missing {len(missing)} records")
        for path in missing[:20]:
            print(f"  {path.relative_to(ROOT)}")
        if not args.allow_partial:
            raise SystemExit(2)
    if raw.empty:
        return

    OUTDIR.mkdir(parents=True, exist_ok=True)
    TABLEDIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(ROOT / "Latex_Yihui_MARL" / "surface_vmax_10seed_raw.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(ROOT / "Latex_Yihui_MARL" / "surface_vmax_10seed_summary.csv", index=False)
    tests = paired_tests(raw)
    tests.to_csv(ROOT / "Latex_Yihui_MARL" / "surface_vmax_10seed_paired_tests.csv", index=False)

    if len(summary) != 2 * len(FLEETS) * len(RADII):
        print("Partial data: tables and final figure were not generated.")
        return

    make_table(
        summary,
        ["distance", "path"],
        ["Tracking distance (km)", "Path length per SUV (km)"],
        [1, 1],
        "Ten-seed surface tracking and trajectory metrics for 12 and 20 m/s.",
        "tab:surface_vmax_dynamic",
        TABLEDIR / "surface_vmax_dynamic.tex",
    )
    make_table(
        summary,
        ["mean_speed", "on_land_fraction"],
        ["Mean ground speed (m s$^{-1}$)", "On-land AUV-step fraction (%)"],
        [2, 3],
        "Ten-seed surface speed and coastline exposure for 12 and 20 m/s.",
        "tab:surface_vmax_operational",
        TABLEDIR / "surface_vmax_operational.tex",
    )
    plot_metrics(
        summary,
        OUTDIR / "surface_vmax_10seed_metrics.pdf",
        OUTDIR / "surface_vmax_10seed_metrics.png",
    )
    write_numeric_results(
        summary,
        tests,
        ROOT / "Latex_Yihui_MARL" / "sections" / "surface_vmax_results.tex",
    )
    print(ROOT / "Latex_Yihui_MARL" / "surface_vmax_10seed_summary.csv")
    print(TABLEDIR / "surface_vmax_dynamic.tex")
    print(TABLEDIR / "surface_vmax_operational.tex")
    print(OUTDIR / "surface_vmax_10seed_metrics.pdf")


if __name__ == "__main__":
    main()
