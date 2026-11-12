"""Ten-seed underwater comparison: active policy, random control, passive drift."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from code.paths import PAPER_ROOT, REPO_ROOT, RESULTS_ROOT

ROOT = REPO_ROOT
SRC = RESULTS_ROOT / "underwater" / "eval_uwflow_500_summary.psv"
FIGDIR = PAPER_ROOT / "figures"
TABLEDIR = PAPER_ROOT / "tables"
DEPTHS = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
SPEEDS = (2.0, 1.5)


def aggregate(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for depth in DEPTHS:
        for speed in SPEEDS:
            q = d[
                (d.depth == depth)
                & (d.vmax == speed)
                & (d.dur == 20.0)
            ]
            active = q[
                (q.algo == "sb3")
                & (q.reward == "C")
                & (q.train_seed >= 0)
            ]
            random = q[(q.algo == "random") & (q.reward == "random")]
            drift = q[(q.algo == "idle") & (q.reward == "idle")]
            if len(active) != 10 or random.empty or drift.empty:
                raise ValueError(
                    f"incomplete underwater data for depth={depth}, vmax={speed}"
                )
            for method, part, nseed in (
                ("Active IPPO", active, len(active)),
                ("Random", random.iloc[[0]], 1),
                ("Passive drift", drift.iloc[[0]], 1),
            ):
                rows.append(
                    {
                        "depth_m": depth,
                        "vmax_ms": speed,
                        "method": method,
                        "n_seed": nseed,
                        "mean_current_speed": part.mean_V.mean(),
                        "mean_current_speed_std": (
                            part.mean_V.std(ddof=1) if nseed > 1 else 0.0
                        ),
                        "coverage_area_km2": part.cov_area_km2.mean(),
                        "coverage_area_km2_std": (
                            part.cov_area_km2.std(ddof=1) if nseed > 1 else 0.0
                        ),
                        "path_km": part.path_km.mean(),
                        "path_km_std": (
                            part.path_km.std(ddof=1) if nseed > 1 else 0.0
                        ),
                        "mean_speed_ms": part.mean_speed_ms.mean(),
                        "mean_speed_ms_std": (
                            part.mean_speed_ms.std(ddof=1) if nseed > 1 else 0.0
                        ),
                        "action_magnitude": part.mean_a2.mean(),
                        "action_magnitude_std": (
                            part.mean_a2.std(ddof=1) if nseed > 1 else 0.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def add_current_weighted_coverage(s: pd.DataFrame) -> pd.DataFrame:
    reference = RESULTS_ROOT / "underwater" / "underwater_final_analysis.csv"
    if not reference.exists():
        raise FileNotFoundError(
            "The final current-weighted coverage table is required: "
            f"{reference}"
        )
    ref = pd.read_csv(reference)
    keys = ["depth_m", "vmax_ms", "method"]
    weighted = ref[
        keys
        + [
            "current_weighted_coverage",
            "current_weighted_coverage_std",
        ]
    ]
    out = s.drop(
        columns=[
            "current_weighted_coverage",
            "current_weighted_coverage_std",
        ],
        errors="ignore",
    ).merge(weighted, on=keys, how="left", validate="one_to_one")
    required = ["current_weighted_coverage", "current_weighted_coverage_std"]
    if out[required].isna().any().any():
        raise ValueError("Current-weighted coverage is incomplete for a method.")
    return out


def make_table(s: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[p]",
        r"\centering",
        r"\caption{Ten-seed comparison of active control, random control, and "
        r"passive drift in the 20 h subsurface task. Values are means $\pm$ "
        r"sample standard deviations across ten trained active-policy seeds. "
        r"Random control and passive drift share the same 500 evaluation "
        r"episodes and have no training-seed variability. Normalized action "
        r"magnitude is the mean squared norm of the unprojected action.}",
        r"\label{tab:underwater_methods_10seed}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}rrlrrrrr@{}}",
        r"\toprule",
        r"Depth (m) & $V_{\max}$ (m/s) & Method & "
        r"Mean current (m/s) & Nominal coverage (km$^2$) & "
        r"Current-weighted coverage ($10^3$ m s$^{-1}$ km$^2$) & "
        r"Mean ground speed (m/s) & Normalized action magnitude \\",
        r"\midrule",
    ]
    for depth in DEPTHS:
        for speed in SPEEDS:
            q = s[(s.depth_m == depth) & (s.vmax_ms == speed)]
            for _, row in q.iterrows():
                def field(name: str, digits: int, scale: float = 1.0) -> str:
                    if int(row.n_seed) == 1:
                        return f"{row[name] / scale:.{digits}f}"
                    return (
                        f"{row[name] / scale:.{digits}f}$\\pm$"
                        f"{row[name + '_std'] / scale:.{digits}f}"
                    )

                lines.append(
                    f"{depth:g} & {speed:.1f} & {row.method} & "
                    f"{field('mean_current_speed', 3)} & "
                    f"{field('coverage_area_km2', 0)} & "
                    f"{field('current_weighted_coverage', 3, 1.0e3)} & "
                    f"{field('mean_speed_ms', 2)} & "
                    f"{field('action_magnitude', 3)} \\\\"
                )
            if not (depth == DEPTHS[-1] and speed == SPEEDS[-1]):
                lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
        "",
    ]
    (TABLEDIR / "underwater_methods_10seed.tex").write_text("\n".join(lines))


def make_figure(s: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5})
    fig, axes = plt.subplots(
        2, 3, figsize=(12.0, 6.4)
    )
    fig.subplots_adjust(
        left=0.07, right=0.99, top=0.82, bottom=0.10,
        hspace=0.36, wspace=0.25,
    )
    metrics = [
        (
            "mean_current_speed",
            1.0,
            "Experienced current speed",
            "Mean current speed (m/s)",
        ),
        (
            "coverage_area_km2",
            1.0e3,
            "Nominal spatial coverage",
            r"Nominal coverage ($10^3$ km$^2$)",
        ),
        (
            "current_weighted_coverage",
            1.0e3,
            "Current-weighted coverage",
            r"$\mathcal{C}_w$ ($10^3$ m s$^{-1}$ km$^2$)",
        ),
    ]
    colors = {
        "Active IPPO": "#264653",
        "Random": "#E76F51",
        "Passive drift": "#2A9D8F",
    }
    width = 0.24
    x = np.arange(len(DEPTHS))
    for r, speed in enumerate(SPEEDS):
        for c, (metric, scale, title, ylabel) in enumerate(metrics):
            ax = axes[r, c]
            for k, method in enumerate(("Active IPPO", "Random", "Passive drift")):
                q = (
                    s[(s.vmax_ms == speed) & (s.method == method)]
                    .set_index("depth_m")
                    .loc[list(DEPTHS)]
                )
                ax.bar(
                    x + (k - 1) * width,
                    q[metric] / scale,
                    width,
                    yerr=q[f"{metric}_std"] / scale,
                    capsize=3,
                    color=colors[method],
                    edgecolor="black",
                    linewidth=0.4,
                    label=method,
                )
            ax.set_xticks(x, [f"{int(d)} m" for d in DEPTHS])
            ax.tick_params(axis="x", labelsize=10.5)
            ax.set_xlabel("Observation depth (m)", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            if r == 0:
                ax.set_title(title)
            ax.text(
                0.98, 0.96, rf"$V_{{\max}}$ = {speed:.1f} m/s",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, fontweight="bold",
            )
            ax.grid(axis="y", alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        fontsize=12,
    )
    fig.suptitle(
        "Subsurface strategy comparison across 0-60 m",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.savefig(
        FIGDIR / "FigC2_Underwater_Methods_7Depth.pdf", bbox_inches="tight"
    )
    fig.savefig(
        FIGDIR / "FigC2_Underwater_Methods_7Depth.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    d = pd.read_csv(SRC, sep="|")
    s = add_current_weighted_coverage(aggregate(d))
    TABLEDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_ROOT / "underwater" / "underwater_methods_10seed.csv"
    s.to_csv(csv_path, index=False)
    make_table(s)
    make_figure(s)
    print(csv_path)
    print(FIGDIR / "FigC2_Underwater_Methods_7Depth.pdf")


if __name__ == "__main__":
    main()
