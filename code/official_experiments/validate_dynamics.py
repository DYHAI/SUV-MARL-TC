#!/usr/bin/env python3
"""Numerical checks for the constrained planar dynamics."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from code.common.constrained_dynamics import (
    ConstrainedPlanarDynamics,
    DynamicsConfig,
)


def run_case(stage: str, vmax: float) -> dict[str, float]:
    os.environ["DYNAMICS_MODE"] = "constrained"
    config = DynamicsConfig.from_env(stage)
    dynamics = ConstrainedPlanarDynamics(1, vmax, config)
    dynamics.reset(seed=0)
    actions = [
        np.array([[1.0, 0.0]]),
        np.array([[0.0, 1.0]]),
        np.array([[-1.0, 0.0]]),
        np.array([[0.0, -1.0]]),
        np.array([[0.0, 0.0]]),
    ]
    max_speed = 0.0
    max_accel = 0.0
    max_yaw = 0.0
    for _ in range(4):
        for action in actions:
            prop, displacement, info = dynamics.step(action, 600.0)
            max_speed = max(max_speed, float(np.max(info["speed"])))
            max_accel = max(max_accel, float(np.max(info["accel_peak"])))
            max_yaw = max(max_yaw, float(np.max(info["yaw_rate_peak"])))
            assert np.isfinite(prop).all()
            assert np.isfinite(displacement).all()
    return {
        "vmax_mps": vmax,
        "accel_limit_mps2": config.accel_mps2,
        "decel_limit_mps2": config.decel_mps2,
        "yaw_limit_deg_s": np.degrees(config.yaw_rate_max_rad_s),
        "min_turn_radius_m": config.min_turn_radius_m,
        "observed_max_speed_mps": max_speed,
        "observed_max_accel_mps2": max_accel,
        "observed_max_yaw_deg_s": np.degrees(max_yaw),
    }


def main() -> None:
    out = {
        "surface": run_case("surface", 20.0),
        "subsurface": run_case("subsurface", 2.0),
    }
    for result in out.values():
        assert result["observed_max_speed_mps"] <= result["vmax_mps"] + 1e-9
        assert result["observed_max_accel_mps2"] <= max(
            result["accel_limit_mps2"], result["decel_limit_mps2"]
        ) + 1e-9
        assert (
            result["observed_max_yaw_deg_s"]
            <= result["yaw_limit_deg_s"] + 1e-9
        )
    path = Path(__file__).resolve().parent / "constraint_validation.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
