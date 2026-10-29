"""Ten-seed underwater comparison: active policy, random control, passive drift."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "eval_uwflow_500_summary.psv"
FIGDIR = ROOT / "Latex_Yihui_MARL" / "figures"
TABLEDIR = ROOT / "Latex_Yihui_MARL" / "tables"
DEPTHS = (0.0, 30.0, 50.0)
SPEEDS = (1.5, 2.0)


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
                    }
                )
    return pd.DataFrame(rows)


def make_table(s: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[p]",
        r"\centering",
        r"\caption{Ten-seed comparison of active control, random control, and "
        r"passive drift in the 20 h subsurface task. Values are means $\pm$ "
        r"sample standard deviations across ten trained active-policy seeds. "
        r"Random control and passive drift share the same 500 evaluation "
        r"episodes and have no training-seed variability.}",
        r"\label{tab:underwater_methods_10seed}",
        r"\small",
        r"\begin{tabular}{@{}rrlrrrr@{}}",
        r"\toprule",
        r"Depth (m) & $V_{\max}$ (m/s) & Method & "
        r"Mean current (m/s) & Coverage (km$^2$) & "
        r"Path (km) & Mean ground speed (m/s) \\",
        r"\midrule",
    ]
    for depth in DEPTHS:
        for speed in SPEEDS:
            q = s[(s.depth_m == depth) & (s.vmax_ms == speed)]
            for _, row in q.iterrows():
                def field(name: str, digits: int) -> str:
                    if int(row.n_seed) == 1:
                        return f"{row[name]:.{digits}f}"
                    return (
                        f"{row[name]:.{digits}f}$\\pm$"
                        f"{row[name + '_std']:.{digits}f}"
                    )

                lines.append(
                    f"{depth:g} & {speed:.1f} & {row.method} & "
                    f"{field('mean_current_speed', 3)} & "
                    f"{field('coverage_area_km2', 0)} & "
                    f"{field('path_km', 1)} & "
                    f"{field('mean_speed_ms', 2)} \\\\"
                )
            if not (depth == DEPTHS[-1] and speed == SPEEDS[-1]):
                lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TABLEDIR / "underwater_methods_10seed.tex").write_text("\n".join(lines))


def make_figure(s: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(
        2, 4, figsize=(14.0, 7.2), constrained_layout=True
    )
    metrics = [
        ("mean_current_speed", "Mean current speed (m/s)"),
        ("coverage_area_km2", "Coverage area (km$^2$)"),
        ("path_km", "Path length per SUV (km)"),
        ("mean_speed_ms", "Mean ground speed (m/s)"),
    ]
    colors = {
        "Active IPPO": "#2b6fa8",
        "Random": "#d9822b",
        "Passive drift": "#3b9b6d",
    }
    width = 0.24
    x = np.arange(len(DEPTHS))
    for r, speed in enumerate(SPEEDS):
        for c, (metric, ylabel) in enumerate(metrics):
            ax = axes[r, c]
            for k, method in enumerate(("Active IPPO", "Random", "Passive drift")):
                q = (
                    s[(s.vmax_ms == speed) & (s.method == method)]
                    .set_index("depth_m")
                    .loc[list(DEPTHS)]
                )
                ax.bar(
                    x + (k - 1) * width,
                    q[metric],
                    width,
                    yerr=q[f"{metric}_std"],
                    capsize=3,
                    color=colors[method],
                    edgecolor="black",
                    linewidth=0.4,
                    label=method,
                )
            ax.set_xticks(x, [f"{int(d)} m" for d in DEPTHS])
            if r == 1:
                ax.set_xlabel("Ocean depth")
            if c == 0:
                ax.set_ylabel(rf"$V_{{\max}}$={speed:.1f} m/s\n{ylabel}")
            else:
                ax.set_title(ylabel)
            ax.grid(axis="y", alpha=0.22)
    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle(
        "Ten-seed subsurface observation strategy comparison (20 h)",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(
        FIGDIR / "underwater_methods_10seed.pdf", bbox_inches="tight"
    )
    fig.savefig(
        FIGDIR / "underwater_methods_10seed.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    d = pd.read_csv(SRC, sep="|")
    s = aggregate(d)
    TABLEDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    csv_path = ROOT / "Latex_Yihui_MARL" / "underwater_methods_10seed.csv"
    s.to_csv(csv_path, index=False)
    make_figure(s)
    print(csv_path)
    print(FIGDIR / "underwater_methods_10seed.pdf")


if __name__ == "__main__":
    main()
