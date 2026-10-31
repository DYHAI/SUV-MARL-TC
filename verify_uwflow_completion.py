#!/usr/bin/env python3
"""Verify every artifact required by the final seven-depth experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import audit_hycom_currents


ROOT = Path(__file__).resolve().parent
DOC = ROOT / "Latex_Yihui_MARL"
DEPTHS = (0, 10, 20, 30, 40, 50, 60)
SPEEDS = ("1.5", "2.0")


def main() -> int:
    failures = []

    archive = audit_hycom_currents.audit(DEPTHS)
    for depth in DEPTHS:
        available = archive["depths"][str(depth)]["available"]
        print(f"HYCOM {depth:>2} m: {available}/500")
        if available != 500:
            failures.append(f"HYCOM {depth} m incomplete")

    for depth in DEPTHS:
        cache = Path(f"/home/ding/main/data/uwflow_cache/d{depth}_g0.25_c3")
        count = len(list(cache.glob("*.npz"))) if cache.exists() else 0
        print(f"cache {depth:>2} m: {count}/500")
        if count < 500:
            failures.append(f"cache {depth} m incomplete")

    model_count = 0
    for depth in DEPTHS:
        for speed in SPEEDS:
            missing = []
            for seed in range(10):
                name = (
                    f"ppo_uwflow_500_d{depth}_v{speed}_rC_20h_sb3_s{seed}.zip"
                )
                path = ROOT / "ppo_underwater_fleet" / name
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(name)
                else:
                    model_count += 1
            if missing:
                failures.append(
                    f"depth={depth}, speed={speed}: {len(missing)} models missing"
                )
    print(f"corrected models: {model_count}/140")

    eval_files = sorted((ROOT / "eval_out").glob("eval_info_info_d*_500.csv"))
    expected_eval = {
        f"eval_info_info_d{depth}_start200_20h_v{float(speed):g}_500.csv"
        for depth in DEPTHS for speed in SPEEDS
    }
    found_eval = {path.name for path in eval_files}
    print(f"evaluation files: {len(found_eval & expected_eval)}/14")
    if found_eval != expected_eval:
        failures.append("final evaluation files are not exactly the expected 14")

    for path in eval_files:
        if path.name not in expected_eval:
            continue
        data = pd.read_csv(path)
        policies = int(data.policy.nunique())
        rows = len(data)
        if policies != 12 or rows != 6000:
            failures.append(
                f"{path.name}: policies={policies}, rows={rows}; expected 12/6000"
            )

    required_files = (
        ROOT / "eval_uwflow_500_summary.psv",
        DOC / "underwater_final_analysis.csv",
        DOC / "underwater_final_comparison.csv",
        DOC / "underwater_final_report.md",
        DOC / "figures" / "underwater_methods_7depth.pdf",
        DOC / "figures" / "underwater_strategy_trajectories_multidepth.pdf",
        DOC / "tables" / "underwater_methods_10seed.tex",
        DOC / "main.pdf",
    )
    for path in required_files:
        print(f"artifact {path.name}: {'ok' if path.exists() else 'missing'}")
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"missing or empty artifact: {path}")

    if failures:
        print("\nINCOMPLETE")
        for failure in failures:
            print(" -", failure)
        return 1
    print("\nCOMPLETE: all seven-depth experiment artifacts verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
