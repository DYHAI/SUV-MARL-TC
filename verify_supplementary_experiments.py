#!/usr/bin/env python3
"""Verify files and row counts for the reviewer-response experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
FAILS: list[str] = []


def require(path: Path, label: str | None = None) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        FAILS.append(f"missing or empty: {label or path}")


def main() -> int:
    speed = ROOT / "analysis" / "tc_speed_envelope_summary.csv"
    require(speed)
    if speed.is_file():
        frame = pd.read_csv(speed)
        if len(frame) != 1 or int(frame.iloc[0].n_tc) != 500:
            FAILS.append("TC speed summary does not cover 500 TCs")

    surface = sorted((ROOT / "classical_eval").glob("surface_*_test_n*_v*_s1.csv"))
    underwater = sorted(
        (ROOT / "classical_eval").glob("underwater_*_test_d*_v*_s10.csv")
    )
    if len(surface) != 32:
        FAILS.append(f"traditional surface matrix: {len(surface)}/32")
    if len(underwater) != 56:
        FAILS.append(f"traditional underwater matrix: {len(underwater)}/56")
    for path in surface:
        data = pd.read_csv(path)
        if len(data) != 3840:
            FAILS.append(f"{path.name}: {len(data)}/3840 rows")
    for path in underwater:
        data = pd.read_csv(path)
        if len(data) != 500:
            FAILS.append(f"{path.name}: {len(data)}/500 rows")

    expected_algorithms = {
        "surface_sac_n8_v12_s0",
        "surface_sac_n8_v12_s1",
        "surface_td3_n8_v12_s0",
        "surface_td3_n8_v12_s1",
        "underwater_sac_n4_v1.5_d30_s0",
        "underwater_sac_n4_v1.5_d30_s1",
        "underwater_td3_n4_v1.5_d30_s0",
        "underwater_td3_n4_v1.5_d30_s1",
    }
    for name in sorted(expected_algorithms):
        require(ROOT / "ppo_algorithm_comparison" / f"{name}.zip", name)
    require(ROOT / "robustness_eval" / "algorithm_surface_eval.csv")
    require(ROOT / "robustness_eval" / "algorithm_underwater_eval.csv")
    require(ROOT / "Latex_Yihui_MARL" / "tables" / "algorithm_comparison.tex")

    for speed in (12, 20):
        for formation in ("line", "grid"):
            require(
                ROOT
                / "ppo_formation_ablation"
                / f"surface_{formation}_n8_v{speed}_s0.zip",
                f"{formation} formation at {speed} m/s",
            )
        for formation in ("ring", "line", "grid"):
            for radius in (100, 200, 400, 600, 800):
                path = (
                    ROOT
                    / "robustness_eval"
                    / "formation_full_matrix"
                    / f"formation_{formation}_v{speed}_d{radius}.csv"
                )
                require(path)
                if path.is_file():
                    data = pd.read_csv(path)
                    if len(data) != 768:
                        FAILS.append(f"{path.name}: {len(data)}/768 rows")
                    if set(data["condition"]) != {"clean"}:
                        FAILS.append(f"{path.name}: unexpected conditions")
    require(ROOT / "robustness_eval" / "formation_full_matrix_summary.csv")
    require(ROOT / "Latex_Yihui_MARL" / "tables" / "formation_comparison.tex")

    require(ROOT / "Latex_Yihui_MARL" / "tables" / "surface_control_ablations.tex")
    require(ROOT / "Latex_Yihui_MARL" / "tables" / "planner_stratified.tex")
    require(ROOT / "Latex_Yihui_MARL" / "main.tex")
    require(ROOT / "Latex_Yihui_MARL" / "main.pdf")

    if FAILS:
        print("INCOMPLETE")
        for failure in FAILS:
            print(" -", failure)
        return 1
    print("COMPLETE: supplementary experiment artifacts verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
