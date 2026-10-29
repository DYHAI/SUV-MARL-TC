#!/usr/bin/env python3
"""Check core imports and required reproduction entry points."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


MODULES = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "xarray",
    "netCDF4",
    "gymnasium",
    "stable_baselines3",
    "torch",
)

FILES = (
    "fleet_segment_env.py",
    "underwater_fleet_flow_env.py",
    "train_fleet_segment_actfilter.py",
    "train_uwflow_sb3.py",
    "eval_deploy_xfer.py",
    "eval_uwflow_info.py",
    "audit_hycom_currents.py",
    "Latex_Yihui_MARL/main.tex",
    "Latex_Yihui_MARL/main.pdf",
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    print(f"Python: {sys.version.split()[0]}")
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"[ok] {name}: {version}")
        except Exception as exc:
            failures.append(f"import {name}: {exc}")
    for relative in FILES:
        path = root / relative
        if path.exists():
            print(f"[ok] {relative}")
        else:
            failures.append(f"missing {relative}")
    if failures:
        print("\nFailures:")
        for item in failures:
            print(f"- {item}")
        return 1
    print("\nRuntime and repository entry points are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
