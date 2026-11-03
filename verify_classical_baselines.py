#!/usr/bin/env python3
"""Verify the complete classical-planner comparison matrix."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "classical_eval"
DEPTHS = (0, 10, 20, 30, 40, 50, 60)
SPEEDS = (1.5, 2.0)
PLANNERS = ("potential", "astar", "coverage", "greedy")


def main() -> int:
    failures = []
    surface = sorted(OUT.glob("surface_*_test_n*_v*_s1.csv"))
    if len(surface) != 32:
        failures.append(f"surface raw files: {len(surface)}/32")
    surface_rows = 0
    for path in surface:
        data = pd.read_csv(path)
        surface_rows += len(data)
        if len(data) != 3840 or data.radius_km.nunique() != 5:
            failures.append(f"{path.name}: rows={len(data)}")

    underwater = sorted(OUT.glob("underwater_*_test_d*_v*_s10.csv"))
    if len(underwater) != 56:
        failures.append(f"underwater raw files: {len(underwater)}/56")
    underwater_rows = 0
    for path in underwater:
        data = pd.read_csv(path)
        underwater_rows += len(data)
        pairs = data[["storm", "deployment_index"]].drop_duplicates()
        if len(data) != 500 or len(pairs) != 500:
            failures.append(f"{path.name}: rows={len(data)}, pairs={len(pairs)}")
        if data.deployment_index.nunique() != 10:
            failures.append(f"{path.name}: deployment indices incomplete")
        if sorted(data.deployment_seed.unique()) != list(range(100, 110)):
            failures.append(f"{path.name}: deployment RNG seeds do not match MARL")

    for depth in DEPTHS:
        for speed in SPEEDS:
            for planner in PLANNERS:
                path = OUT / (
                    f"underwater_{planner}_test_d{depth}_v{speed}_s10.csv"
                )
                if not path.is_file():
                    failures.append(f"missing {path.name}")

    surface_summary = pd.read_csv(OUT / "surface_all_methods_summary.csv")
    underwater_summary = pd.read_csv(OUT / "underwater_all_methods_summary.csv")
    underwater_tests = pd.read_csv(OUT / "underwater_pairwise_tests.csv")
    if len(surface_summary) != 200:
        failures.append(f"surface summary rows: {len(surface_summary)}/200")
    if len(underwater_summary) != 70:
        failures.append(f"underwater summary rows: {len(underwater_summary)}/70")
    if len(underwater_tests) != 280:
        failures.append(f"underwater tests: {len(underwater_tests)}/280")

    manuscript_table = (
        ROOT / "Latex_Yihui_MARL" / "tables" /
        "classical_planner_comparison.tex"
    )
    if not manuscript_table.is_file() or manuscript_table.stat().st_size == 0:
        failures.append("manuscript comparison table missing")

    print(f"surface raw files: {len(surface)}/32 ({surface_rows} rows)")
    print(f"underwater raw files: {len(underwater)}/56 ({underwater_rows} rows)")
    print(f"surface summary rows: {len(surface_summary)}/200")
    print(f"underwater summary rows: {len(underwater_summary)}/70")
    print(f"underwater paired tests: {len(underwater_tests)}/280")
    if failures:
        print("\nINCOMPLETE")
        for failure in failures:
            print(" -", failure)
        return 1
    print("\nCOMPLETE: classical planner matrices and paired tests verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
