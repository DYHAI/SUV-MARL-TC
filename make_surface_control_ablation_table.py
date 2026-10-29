"""Generate the low-pass and on-land-penalty ablation table."""

import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Latex_Yihui_MARL" / "tables" / "surface_control_ablations.tex"

def read_land_fraction(path: Path) -> float:
    text = path.read_text(errors="replace")
    match = re.search(r"上岸占比\(AUV-step\)\s+([0-9.]+)\s*%", text)
    if not match:
        raise ValueError(f"on-land fraction not found in {path}")
    return float(match.group(1))


def read_lowpass_metrics(path: Path) -> tuple[float, float, float, float]:
    text = path.read_text(errors="replace")
    final_distance = re.search(r"最终中心距离\s+([0-9.]+)\s+km", text)
    path_km = re.search(r"航迹长度\s+([0-9.]+)\s+km", text)
    mean_speed = re.search(r"平均航速\s+([0-9.]+)\s+m/s", text)
    max_speed = re.search(r"最大速度\s+([0-9.]+)\s+m/s", text)
    if not (final_distance and path_km and mean_speed and max_speed):
        raise ValueError(f"metrics not found in {path}")
    return (
        float(final_distance.group(1)),
        float(path_km.group(1)),
        float(mean_speed.group(1)),
        float(max_speed.group(1)),
    )


def main() -> None:
    no_penalty = read_land_fraction(ROOT / "eval_land000.txt")
    with_penalty = read_land_fraction(ROOT / "eval_landpen.txt")
    logdir = ROOT / "logs_filter_ablation"
    lowpass = {}
    for fleet in (4, 8, 12, 16):
        lowpass[fleet] = {
            "off": read_lowpass_metrics(logdir / f"exact_n{fleet}_b00.log"),
            "on": read_lowpass_metrics(logdir / f"exact_n{fleet}_b08.log"),
        }
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Effect of execution-layer low-pass filtering for 4/8/12/16 "
        r"SUVs on the 500-TC benchmark. The same trained policy is evaluated "
        r"with the filter off and on ($\beta=0.8$) over 768 held-out segments.}",
        r"\label{tab:surface_control_ablations}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{0.9}",
        r"\resizebox{0.94\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Fleet & Filter & Final mean distance (km) & Path per SUV (km) & "
        r"Mean speed (m/s) & Maximum speed (m/s) \\",
        r"\midrule",
    ]
    for fleet in (4, 8, 12, 16):
        states = ("off", "on")
        values = {
            metric: {
                state: lowpass[fleet][state][index]
                for state in states
            }
            for index, metric in enumerate(("final", "path", "mean", "maximum"))
        }
        best = {
            metric: min(states, key=lambda state: values[metric][state])
            for metric in values
        }
        for state, label in (("off", "Off"), ("on", "On")):
            final, path, mean, maximum = lowpass[fleet][state]
            cells = []
            for metric, value, digits in (
                ("final", final, 1),
                ("path", path, 1),
                ("mean", mean, 2),
                ("maximum", maximum, 2),
            ):
                text = f"{value:.{digits}f}"
                cells.append(r"\textbf{" + text + "}" if best[metric] == state else text)
            lines.append(f"{fleet} & {label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\par\smallskip",
        r"\parbox{0.94\textwidth}{\footnotesize Final mean distance is the "
        r"fleet average of the individual SUV-to-best-track-center distances "
        r"at the final time step. Off/on changes only the "
        r"execution-layer filter on the same trained actor.}",
        r"\end{table*}",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(["No reward", "$P_{land}=20$"], [no_penalty, with_penalty],
           color=["#d9822b", "#3b9b6d"], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("On-land AUV-step fraction (%)")
    ax.set_ylim(0, max(no_penalty, with_penalty) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate([no_penalty, with_penalty]):
        ax.text(idx, value + 0.1, f"{value:.2f}%", ha="center", va="bottom")
    figdir = ROOT / "Latex_Yihui_MARL" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "land_penalty_ablation.pdf", bbox_inches="tight")
    fig.savefig(figdir / "land_penalty_ablation.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
