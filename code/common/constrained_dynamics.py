"""Planar dynamics with speed, acceleration, heading, and turn limits.

The final manuscript experiments instantiate this module with
``DYNAMICS_MODE=constrained``. The original instantaneous kinematic model is
retained only as a development option.

The default constrained values are deliberately conservative assumptions, not
calibrated Blue Whale specifications:

* surface: acceleration 0.02 m/s^2, deceleration 0.03 m/s^2,
  yaw-rate limit 1 deg/s, minimum turn radius 200 m;
* subsurface: acceleration 0.002 m/s^2, deceleration 0.003 m/s^2,
  yaw-rate limit 2 deg/s, minimum turn radius 100 m.

Every value can be overridden with the ``DYN_*`` environment variables listed
in ``from_env``.  Dynamics are integrated with 60 s substeps inside each
10 min control interval.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class DynamicsConfig:
    mode: str
    accel_mps2: float
    decel_mps2: float
    yaw_rate_max_rad_s: float
    min_turn_radius_m: float
    substep_s: float

    @classmethod
    def from_env(cls, stage: str) -> "DynamicsConfig":
        if stage not in {"surface", "subsurface"}:
            raise ValueError(f"unsupported dynamics stage: {stage}")
        prefix = "DYN_SURFACE" if stage == "surface" else "DYN_SUB"
        if stage == "surface":
            accel = _env_float("DYN_SURFACE_ACC_MS2", 0.02)
            decel = _env_float("DYN_SURFACE_DEC_MS2", 0.03)
            yaw_deg = _env_float("DYN_SURFACE_YAW_DEG_S", 1.0)
            radius = _env_float("DYN_SURFACE_MIN_TURN_RADIUS_M", 200.0)
        else:
            accel = _env_float("DYN_SUB_ACC_MS2", 0.002)
            decel = _env_float("DYN_SUB_DEC_MS2", 0.003)
            yaw_deg = _env_float("DYN_SUB_YAW_DEG_S", 2.0)
            radius = _env_float("DYN_SUB_MIN_TURN_RADIUS_M", 100.0)

        mode = os.environ.get("DYNAMICS_MODE", "kinematic").strip().lower()
        substep = _env_float(f"{prefix}_SUBSTEP_S", 60.0)
        if accel <= 0.0 or decel <= 0.0:
            raise ValueError("acceleration and deceleration limits must be positive")
        if yaw_deg <= 0.0 or radius <= 0.0 or substep <= 0.0:
            raise ValueError("yaw rate, turn radius, and substep must be positive")
        return cls(
            mode=mode,
            accel_mps2=accel,
            decel_mps2=decel,
            yaw_rate_max_rad_s=math.radians(yaw_deg),
            min_turn_radius_m=radius,
            substep_s=substep,
        )


class ConstrainedPlanarDynamics:
    """Second-order speed and first-order heading dynamics for horizontal motion.

    The policy command is still a unit-disk vector in ``[-1, 1]^2``.  It is
    converted to a desired speed and course.  The actual speed and heading then
    approach those targets subject to acceleration, deceleration, yaw-rate, and
    minimum-turn-radius limits.  Propulsion velocity is aligned with heading.
    """

    def __init__(self, n_auv: int, vmax: float, config: DynamicsConfig):
        self.n_auv = int(n_auv)
        self.vmax = float(vmax)
        self.config = config
        self.speed = np.zeros(self.n_auv, dtype=np.float64)
        self.heading = np.zeros(self.n_auv, dtype=np.float64)
        self.prop_velocity = np.zeros((self.n_auv, 2), dtype=np.float64)
        self.yaw_rate = np.zeros(self.n_auv, dtype=np.float64)
        self.accel = np.zeros(self.n_auv, dtype=np.float64)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        self.speed.fill(0.0)
        self.heading[:] = np.random.uniform(0.0, 2.0 * np.pi, self.n_auv)
        self.prop_velocity.fill(0.0)
        self.yaw_rate.fill(0.0)
        self.accel.fill(0.0)

    def step(
        self,
        actions: np.ndarray,
        dt_s: float,
        current_velocity: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Advance one control interval.

        Returns
        -------
        propulsion_velocity:
            Final propulsion velocity in m/s, shape ``(n, 2)``.
        displacement_m:
            Integrated ground displacement in metres, shape ``(n, 2)``.
        info:
            Final speed, heading, yaw rate, and along-heading acceleration.
        """
        a = np.asarray(actions, dtype=np.float64).reshape(self.n_auv, 2)
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
        a = np.clip(a, -1.0, 1.0)
        norm = np.linalg.norm(a, axis=1, keepdims=True)
        a = a / np.maximum(1.0, norm)

        desired_speed = self.vmax * np.linalg.norm(a, axis=1)
        moving = desired_speed > 1e-9
        desired_heading = self.heading.copy()
        desired_heading[moving] = np.arctan2(a[moving, 1], a[moving, 0])

        current = np.zeros((self.n_auv, 2), dtype=np.float64)
        if current_velocity is not None:
            current = np.asarray(current_velocity, dtype=np.float64).reshape(self.n_auv, 2)

        old_speed = self.speed.copy()
        displacement = np.zeros((self.n_auv, 2), dtype=np.float64)
        accel_peak = np.zeros(self.n_auv, dtype=np.float64)
        yaw_rate_peak = np.zeros(self.n_auv, dtype=np.float64)
        n_sub = max(1, int(math.ceil(float(dt_s) / self.config.substep_s)))
        sub_dt = float(dt_s) / n_sub

        for _ in range(n_sub):
            speed_error = desired_speed - self.speed
            max_up = self.config.accel_mps2 * sub_dt
            max_down = self.config.decel_mps2 * sub_dt
            speed_step = np.clip(speed_error, -max_down, max_up)
            accel_peak = np.maximum(accel_peak, np.abs(speed_step) / max(sub_dt, 1e-9))
            self.speed = np.clip(self.speed + speed_step, 0.0, self.vmax)

            heading_error = _wrap_angle(desired_heading - self.heading)
            radius_rate = np.full(self.n_auv, self.config.yaw_rate_max_rad_s)
            turning = moving & (self.speed > 1e-3)
            radius_rate[turning] = np.minimum(
                radius_rate[turning],
                self.speed[turning] / self.config.min_turn_radius_m,
            )
            heading_step = np.clip(
                heading_error,
                -radius_rate * sub_dt,
                radius_rate * sub_dt,
            )
            yaw_rate_peak = np.maximum(
                yaw_rate_peak, np.abs(heading_step) / max(sub_dt, 1e-9)
            )
            self.heading = _wrap_angle(self.heading + heading_step)

            self.prop_velocity[:, 0] = self.speed * np.cos(self.heading)
            self.prop_velocity[:, 1] = self.speed * np.sin(self.heading)
            displacement += (self.prop_velocity + current) * sub_dt

        self.accel[:] = (self.speed - old_speed) / max(float(dt_s), 1e-9)
        self.yaw_rate[:] = (
            heading_step / max(sub_dt, 1e-9)
            if "heading_step" in locals()
            else np.zeros(self.n_auv)
        )
        info = {
            "speed": self.speed.copy(),
            "heading": self.heading.copy(),
            "yaw_rate": self.yaw_rate.copy(),
            "accel": self.accel.copy(),
            "yaw_rate_peak": yaw_rate_peak.copy(),
            "accel_peak": accel_peak.copy(),
        }
        return self.prop_velocity.copy(), displacement, info
