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

from code.paths import PAPER_ROOT, REPO_ROOT, RESULTS_ROOT, RUNS_ROOT

ROOT = REPO_ROOT
LOG_ROOT = RUNS_ROOT / "surface" / "logs"
OUTDIR = PAPER_ROOT / "figures"
TABLEDIR = PAPER_ROOT / "tables"
RESULTDIR = RESULTS_ROOT / "surface"
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
        return LOG_ROOT / "vmax12_seed10" / f"eval_n{n_auv}_s{seed}_d{distance}.log"
    if n_auv == 16:
        return LOG_ROOT / "seed16" / f"eval_n16_s{seed}_d{distance}.log"
    if seed == 0:
        return LOG_ROOT / "xfer" / f"xfer_r800_windonly_n{n_auv}_d{distance}.log"
    return LOG_ROOT / "seed_extra" / f"eval_n{n_auv}_s{seed}_d{distance}.log"


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
        2, 2, figsize=(10.0, 6.5), sharex=True, layout="constrained"
    )
    metrics = [
        ("distance", "Tracking distance (km)"),
        ("path", "Path length per SUV (km)"),
    ]
    style = {
        4: dict(color="#0072B2", marker="o"),
        8: dict(color="#009E73", marker="s"),
        12: dict(color="#D55E00", marker="^"),
        16: dict(color="#CC79A7", marker="D"),
    }
    for row, version in enumerate((20, 12)):
        for col, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for fleet in FLEETS:
                d = summary[
                    (summary.fleet == fleet)
                    & (summary.version_ms == version)
                ].sort_values("distance_target_km")
                ax.errorbar(
                    d.distance_target_km,
                    d[metric],
                    yerr=d[f"{metric}_std"],
                    capsize=3,
                    lw=1.6,
                    elinewidth=0.9,
                    label=rf"{fleet}-SUV",
                    **style[fleet],
                )
            if row == 1:
                ax.set_xlabel("Deployment radius (km)")
            ax.set_ylabel(ylabel)
            ax.text(
                0.01, 0.98, rf"$V_{{\max}}$ = {version} m/s",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10, fontweight="bold",
            )
            ax.grid(alpha=0.25)
            ax.set_xticks(RADII)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, ncol=4,
        loc="outside lower center",
    )
    fig.suptitle(
        "Surface tracking and path-length comparison",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    RESULTDIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RESULTDIR / "surface_vmax_10seed_raw.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(RESULTDIR / "surface_vmax_10seed_summary.csv", index=False)
    tests = paired_tests(raw)
    tests.to_csv(RESULTDIR / "surface_vmax_10seed_paired_tests.csv", index=False)

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
    print(RESULTDIR / "surface_vmax_10seed_summary.csv")
    print(TABLEDIR / "surface_vmax_dynamic.tex")
    print(TABLEDIR / "surface_vmax_operational.tex")
    print(OUTDIR / "surface_vmax_10seed_metrics.pdf")


if __name__ == "__main__":
    main()
