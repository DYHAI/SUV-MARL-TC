"""Causal classical-planning baselines for the surface and subsurface tasks.

Both planners use the same measured wind/current values as the learned
policies, the vehicle navigation state, and the coastline distance field.
Neither planner receives the TC center, the complete reanalysis field, or
future environmental frames.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from fleet_segment_env import D2KM, STEP_MIN, coast_gradient, safety_filter


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _coast_interpolator(env):
    """Return a callable mapping [lat, lon] points to offshore distance."""
    if hasattr(env, "dist_interp"):
        return env.dist_interp
    field = env.d["dist_to_land"].astype(np.float32)
    return lambda pts: env._at(field, pts[:, 0], pts[:, 1])


class PotentialFieldPlanner:
    """Gradient/potential-field planner with coastline repulsion."""

    def __init__(
        self,
        kind: str,
        *,
        coast_margin_km: float = 100.0,
        coast_weight: float = 2.0,
    ) -> None:
        if kind not in ("surface", "subsurface"):
            raise ValueError(kind)
        self.kind = kind
        self.coast_margin_km = float(coast_margin_km)
        self.coast_weight = float(coast_weight)
        self._last_pos: np.ndarray | None = None
        self._last_speed: np.ndarray | None = None

    def reset(self, env) -> None:
        self._last_pos = None
        self._last_speed = None

    def _surface_value_and_gradient(self, obs: np.ndarray, env) -> Tuple[np.ndarray, np.ndarray]:
        n = env.n_auv
        flat = np.asarray(obs, dtype=np.float64).reshape(n, -1)
        cells = flat[0, : 3 * n].reshape(n, 3)
        speed = cells[:, 0]
        lat = np.asarray(env.lats, dtype=np.float64)
        lon = np.asarray(env.lons, dtype=np.float64)
        lat0 = float(np.mean(lat))
        x = (lon - float(np.mean(lon))) * D2KM * np.cos(np.deg2rad(lat0))
        y = (lat - lat0) * D2KM
        pts = np.stack([x, y], axis=1)
        grad = np.zeros((n, 2), dtype=np.float64)
        for i in range(n):
            delta = pts - pts[i]
            ds = speed - speed[i]
            keep = np.linalg.norm(delta, axis=1) > 0.5
            if keep.sum() < 2:
                continue
            a = delta[keep]
            b = ds[keep]
            # Ridge-regularized local plane fit of wind speed.
            lhs = a.T @ a + 1e-3 * np.eye(2)
            grad[i] = np.linalg.solve(lhs, a.T @ b)
            if np.linalg.norm(grad[i]) < 1e-8:
                j = int(np.argmax(np.where(keep, ds, -np.inf)))
                grad[i] = _unit(delta[j])
        return speed, grad

    def _subsurface_value_and_gradient(self, obs: np.ndarray, env) -> Tuple[np.ndarray, np.ndarray]:
        n = env.n_auv
        flat = np.asarray(obs, dtype=np.float64).reshape(n, -1)
        speed = flat[:, 12].copy()
        current = flat[:, 13:15].copy()
        pos = np.stack([np.asarray(env.lons), np.asarray(env.lats)], axis=1)
        grad = np.zeros((n, 2), dtype=np.float64)
        if self._last_pos is not None and self._last_speed is not None:
            for i in range(n):
                dlat = (pos[i, 1] - self._last_pos[i, 1]) * D2KM
                dlon = (
                    (pos[i, 0] - self._last_pos[i, 0])
                    * D2KM
                    * np.cos(np.deg2rad(pos[i, 1]))
                )
                dx = np.array([dlon, dlat], dtype=np.float64)
                dist = float(np.linalg.norm(dx))
                if dist > 0.25:
                    grad[i] = (speed[i] - self._last_speed[i]) * dx / (dist * dist)
        # Where the scalar gradient is not yet observable, move downstream.
        weak = np.linalg.norm(grad, axis=1) < 1e-8
        grad[weak] = current[weak]
        self._last_pos = pos.copy()
        self._last_speed = speed.copy()
        return speed, grad

    def act(self, obs: np.ndarray, env) -> np.ndarray:
        if self.kind == "surface":
            _, grad = self._surface_value_and_gradient(obs, env)
        else:
            _, grad = self._subsurface_value_and_gradient(obs, env)
        action = _unit(grad)
        coast = _coast_interpolator(env)
        land = np.asarray(coast(np.stack([env.lats, env.lons], axis=1)), dtype=np.float64)
        near = land < self.coast_margin_km
        if near.any():
            offshore = coast_gradient(np.asarray(env.lons), np.asarray(env.lats), coast)
            scale = self.coast_weight * np.clip(
                1.0 - land / max(self.coast_margin_km, 1.0), 0.0, 1.0
            )
            action[near] += scale[near, None] * offshore[near]
        action = _unit(action)
        if self.kind != "surface":
            return action
        return safety_filter(
            action,
            np.asarray(env.lons),
            np.asarray(env.lats),
            coast,
            self.coast_margin_km,
        )


@dataclass
class _AStarPlan:
    waypoint: np.ndarray
    age: int = 0


class RecedingHorizonAStarPlanner:
    """Receding-horizon A* over an online reconstructed local value map."""

    def __init__(
        self,
        kind: str,
        *,
        replan_steps: int = 15,
        horizon_steps: int = 3,
        coast_margin_km: float = 100.0,
        distance_penalty: float = 0.0005,
    ) -> None:
        if kind not in ("surface", "subsurface"):
            raise ValueError(kind)
        self.kind = kind
        self.replan_steps = int(replan_steps)
        self.horizon_steps = int(horizon_steps)
        self.coast_margin_km = float(coast_margin_km)
        self.distance_penalty = float(distance_penalty)
        self._shared: List[Tuple[float, float, float]] = []
        self._private: List[List[Tuple[float, float, float]]] = []
        self._plans: List[_AStarPlan | None] = []

    def reset(self, env) -> None:
        self._shared = []
        self._private = [[] for _ in range(env.n_auv)]
        self._plans = [None for _ in range(env.n_auv)]

    def _remember_surface(self, obs: np.ndarray, env) -> None:
        n = env.n_auv
        cells = np.asarray(obs, dtype=np.float64).reshape(n, -1)[0, : 3 * n].reshape(n, 3)
        for lat, lon, value in zip(env.lats, env.lons, cells[:, 0]):
            self._shared.append((float(lon), float(lat), float(value)))
        if len(self._shared) > 20_000:
            self._shared = self._shared[-20_000:]

    def _remember_subsurface(self, obs: np.ndarray, env) -> None:
        n = env.n_auv
        flat = np.asarray(obs, dtype=np.float64).reshape(n, -1)
        for i in range(n):
            self._private[i].append(
                (float(env.lons[i]), float(env.lats[i]), float(flat[i, 12]))
            )
            if len(self._private[i]) > 2_000:
                self._private[i] = self._private[i][-2_000:]

    def _samples(self, i: int) -> np.ndarray:
        samples = self._shared if self.kind == "surface" else self._private[i]
        return np.asarray(samples, dtype=np.float64)

    def _predict(self, samples: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return np.zeros_like(lon, dtype=np.float64)
        slon = samples[:, 0]
        slat = samples[:, 1]
        val = samples[:, 2]
        # The most recent 64 samples are sufficient for a local, causal estimate.
        slon = slon[-64:]
        slat = slat[-64:]
        val = val[-64:]
        dx = (lon[..., None] - slon[None, :]) * D2KM * np.cos(
            np.deg2rad(lat[..., None])
        )
        dy = (lat[..., None] - slat[None, :]) * D2KM
        dist2 = dx * dx + dy * dy
        w = 1.0 / np.maximum(dist2, 25.0)
        return np.sum(w * val, axis=-1) / np.sum(w, axis=-1)

    def _plan_one(self, i: int, env, step_km: float) -> np.ndarray:
        samples = self._samples(i)
        radius = self.horizon_steps
        offsets = np.arange(-radius, radius + 1, dtype=np.int64)
        dxs, dys = np.meshgrid(offsets, offsets, indexing="xy")
        valid = dxs * dxs + dys * dys <= radius * radius
        step_deg_lat = step_km / D2KM
        lat0 = float(env.lats[i])
        lon0 = float(env.lons[i])
        lats = lat0 + dys * step_deg_lat
        lons = lon0 + dxs * step_km / (
            D2KM * np.cos(np.deg2rad(np.clip(lats, -89.0, 89.0)))
        )
        pred = self._predict(samples, lons, lats)
        coast = _coast_interpolator(env)
        land = np.asarray(coast(np.stack([lats.ravel(), lons.ravel()], axis=1))).reshape(
            lats.shape
        )
        dist_km = np.hypot(dxs, dys) * step_km
        score = pred - self.distance_penalty * dist_km
        score[~valid] = -np.inf
        score[land <= 0.0] = -np.inf
        # The causal local gradient supplies a stable rolling target. A* then
        # plans around coastline cells instead of moving greedily toward noise
        # in the sparse instantaneous value map.
        recent_count = max(24, 3 * env.n_auv) if self.kind == "surface" else 24
        recent = samples[-recent_count:]
        goal = (0, 0)
        if recent.size:
            sx = (
                (recent[:, 0] - lon0)
                * D2KM
                * np.cos(np.deg2rad(np.clip(lat0, -89.0, 89.0)))
            )
            sy = (recent[:, 1] - lat0) * D2KM
            value = recent[:, 2]
            a = np.stack([sx, sy], axis=1)
            a = a - np.mean(a, axis=0, keepdims=True)
            b = value - np.mean(value)
            lhs = a.T @ a + 1e-2 * np.eye(2)
            grad = np.linalg.solve(lhs, a.T @ b)
            if np.linalg.norm(grad) > 1e-9:
                direction = _unit(grad)
                goal_x = int(np.rint(radius * direction[0]))
                goal_y = int(np.rint(radius * direction[1]))
            else:
                j = int(np.argmax(value))
                direction = _unit(np.array([sx[j], sy[j]], dtype=np.float64))
                goal_x = int(np.rint(radius * direction[0]))
                goal_y = int(np.rint(radius * direction[1]))
            goal = (
                int(np.clip(goal_y, -radius, radius)),
                int(np.clip(goal_x, -radius, radius)),
            )
        goal_idx = (goal[0] + radius, goal[1] + radius)
        if not valid[goal_idx] or land[goal_idx] <= 0.0:
            candidates = valid & (land > 0.0)
            candidates[radius, radius] = False
            if not candidates.any():
                return np.zeros(2, dtype=np.float64)
            candidate_score = np.where(candidates, score, -np.inf)
            gi, gj = np.unravel_index(
                int(np.argmax(candidate_score)), candidate_score.shape
            )
            goal = (int(dys[gi, gj]), int(dxs[gi, gj]))
        if goal == (0, 0):
            return np.zeros(2, dtype=np.float64)
        start = (0, 0)
        frontier = [(0.0, start)]
        cost = {start: 0.0}
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        while frontier:
            _, cur = heapq.heappop(frontier)
            if cur == goal:
                break
            cy, cx = cur
            for dy, dx in (
                (-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1),
            ):
                ny, nx = cy + dy, cx + dx
                if not (-radius <= ny <= radius and -radius <= nx <= radius):
                    continue
                idx = (ny + radius, nx + radius)
                if not valid[idx] or land[idx] <= 0.0:
                    continue
                edge_km = np.hypot(dx, dy) * step_km
                # Strictly positive edge costs prevent cycles in the parent map.
                step_cost = edge_km * (
                    1.0 - 0.8 * float(np.clip(pred[idx], 0.0, 1.0))
                )
                step_cost += 200.0 * max(
                    0.0, 1.0 - float(land[idx]) / self.coast_margin_km
                )
                nxt = (ny, nx)
                new_cost = cost[cur] + step_cost
                if new_cost >= cost.get(nxt, np.inf):
                    continue
                cost[nxt] = new_cost
                parent[nxt] = cur
                h = np.hypot(goal[1] - nx, goal[0] - ny) * step_km
                heapq.heappush(frontier, (new_cost + h, nxt))
        if goal not in parent:
            return np.zeros(2, dtype=np.float64)
        cur = goal
        while parent[cur] != start:
            cur = parent[cur]
        return _unit(np.array([cur[1], cur[0]], dtype=np.float64))

    def act(self, obs: np.ndarray, env) -> np.ndarray:
        if not self._plans:
            self.reset(env)
        if self.kind == "surface":
            self._remember_surface(obs, env)
        else:
            self._remember_subsurface(obs, env)
        step_km = float(env.vmax) * STEP_MIN * 60.0 / 1000.0
        action = np.zeros((env.n_auv, 2), dtype=np.float64)
        for i in range(env.n_auv):
            plan = self._plans[i]
            if plan is None or plan.age >= self.replan_steps:
                plan = _AStarPlan(self._plan_one(i, env, step_km))
                self._plans[i] = plan
            action[i] = plan.waypoint
            plan.age += 1
        action = _unit(action)
        if self.kind != "surface":
            return action
        return safety_filter(
            action,
            np.asarray(env.lons),
            np.asarray(env.lats),
            _coast_interpolator(env),
            self.coast_margin_km,
        )


class BoustrophedonCoveragePlanner:
    """Fixed-lawnmower coverage without access to environmental values.

    The fleet follows parallel east-west legs with 25% turn phases between
    vehicles.  This is a standard geometry-only coverage baseline and therefore
    provides a lower-information contrast to the value-driven planners.
    """

    def __init__(
        self,
        kind: str,
        *,
        lane_width_km: float = 40.0,
        leg_length_km: float = 80.0,
        coast_margin_km: float = 100.0,
    ) -> None:
        if kind not in ("surface", "subsurface"):
            raise ValueError(kind)
        self.kind = kind
        self.lane_width_km = float(lane_width_km)
        self.leg_length_km = float(leg_length_km)
        self.coast_margin_km = float(coast_margin_km)
        self._step = 0

    def reset(self, env) -> None:
        self._step = 0

    def act(self, obs: np.ndarray, env) -> np.ndarray:
        step_km = float(env.vmax) * STEP_MIN * 60.0 / 1000.0
        leg_steps = max(1, int(round(self.leg_length_km / max(step_km, 1e-9))))
        turn_steps = max(1, int(round(0.25 * self.lane_width_km / max(step_km, 1e-9))))
        period = 2 * leg_steps + 2 * turn_steps
        action = np.zeros((env.n_auv, 2), dtype=np.float64)
        for i in range(env.n_auv):
            phase = (self._step + int(round(i * period / env.n_auv))) % period
            if phase < leg_steps:
                action[i] = (1.0, 0.0)
            elif phase < leg_steps + turn_steps:
                action[i] = (0.0, 1.0)
            elif phase < 2 * leg_steps + turn_steps:
                action[i] = (-1.0, 0.0)
            else:
                action[i] = (0.0, 1.0)
        self._step += 1
        action = _unit(action)
        coast = _coast_interpolator(env)
        land = np.asarray(coast(np.stack([env.lats, env.lons], axis=1)), dtype=np.float64)
        if (land < self.coast_margin_km).any():
            offshore = coast_gradient(np.asarray(env.lons), np.asarray(env.lats), coast)
            scale = 2.0 * np.clip(
                1.0 - land / max(self.coast_margin_km, 1.0), 0.0, 1.0
            )
            action = _unit(action + scale[:, None] * offshore)
        if self.kind == "surface":
            return safety_filter(
                action,
                np.asarray(env.lons),
                np.asarray(env.lats),
                coast,
                self.coast_margin_km,
            )
        return action


class InformationGreedyPlanner(RecedingHorizonAStarPlanner):
    """One-step lookahead over a causal local value map.

    The planner evaluates eight unit actions using only observations already
    acquired.  It is the standard greedy alternative to multi-step A* planning.
    """

    def __init__(
        self,
        kind: str,
        *,
        coast_margin_km: float = 100.0,
    ) -> None:
        super().__init__(
            kind,
            replan_steps=1,
            horizon_steps=1,
            coast_margin_km=coast_margin_km,
        )

    def act(self, obs: np.ndarray, env) -> np.ndarray:
        if not self._plans:
            self.reset(env)
        if self.kind == "surface":
            self._remember_surface(obs, env)
        else:
            self._remember_subsurface(obs, env)
        step_km = float(env.vmax) * STEP_MIN * 60.0 / 1000.0
        actions = np.zeros((env.n_auv, 2), dtype=np.float64)
        directions = np.array(
            [
                [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
                [0.70710678, 0.70710678], [-0.70710678, 0.70710678],
                [0.70710678, -0.70710678], [-0.70710678, -0.70710678],
            ],
            dtype=np.float64,
        )
        coast = _coast_interpolator(env)
        for i in range(env.n_auv):
            samples = self._samples(i)
            lat0 = float(env.lats[i])
            lon0 = float(env.lons[i])
            cand_lat = lat0 + directions[:, 1] * step_km / D2KM
            cand_lon = lon0 + directions[:, 0] * step_km / (
                D2KM * np.cos(np.deg2rad(np.clip(lat0, -89.0, 89.0)))
            )
            values = self._predict(samples, cand_lon, cand_lat)
            clearance = np.asarray(
                coast(np.stack([cand_lat, cand_lon], axis=1)), dtype=np.float64
            )
            values = values + 2.0 * np.clip(
                clearance / max(self.coast_margin_km, 1.0), 0.0, 1.0
            )
            values[clearance <= 0.0] = -np.inf
            if not np.isfinite(values).any():
                actions[i] = (0.0, 0.0)
            else:
                actions[i] = directions[int(np.argmax(values))]
        actions = _unit(actions)
        if self.kind == "surface":
            return safety_filter(
                actions,
                np.asarray(env.lons),
                np.asarray(env.lats),
                coast,
                self.coast_margin_km,
            )
        return actions


def make_planner(name: str, kind: str):
    if name == "potential":
        return PotentialFieldPlanner(kind)
    if name == "astar":
        return RecedingHorizonAStarPlanner(kind)
    if name == "coverage":
        return BoustrophedonCoveragePlanner(kind)
    if name == "greedy":
        return InformationGreedyPlanner(kind)
    raise ValueError(name)
