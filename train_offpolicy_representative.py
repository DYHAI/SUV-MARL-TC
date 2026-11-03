#!/usr/bin/env python3
"""Train SAC or TD3 on representative multi-SUV observation tasks."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import psutil
import torch
from gymnasium import spaces
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from fleet_segment_env import FleetSegmentEnv
from underwater_fleet_flow_env import UnderwaterFleetFlowEnv


ROOT = Path(__file__).resolve().parent
SPLIT = ROOT / "data" / "typhoon_split_500.json"
SEGS = ROOT / "data" / "segments_500.csv"
OUT = ROOT / "ppo_algorithm_comparison"


class SurfaceOffPolicyEnv(gym.Env):
    def __init__(self, storms, segments, n_auv=8, vmax=12.0):
        self.inner = FleetSegmentEnv(
            storms=storms,
            segments_df=segments,
            use_uv=True,
            n_auv=n_auv,
            smooth_penalty=0.05,
            spacing_deg=0.25,
            patch_n=1,
            deploy_radius=800.0,
            formation="ring",
            ring_radius=100.0,
            sst_mode="none",
            use_mslp=False,
            vmax=vmax,
        )
        self.n_auv = n_auv
        self.obs_dim = self.inner.obs_dim
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (n_auv * self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            -1.0, 1.0, (n_auv * 2,), dtype=np.float32
        )
        self.executed = np.zeros((n_auv, 2), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.inner.reset(seed=seed)
        self.executed = np.zeros((self.n_auv, 2), dtype=np.float32)
        return obs.reshape(-1).astype(np.float32), info

    def step(self, action):
        raw = np.asarray(action, dtype=np.float32).reshape(self.n_auv, 2)
        self.executed = 0.2 * raw + 0.8 * self.executed
        obs, reward, term, trunc, info = self.inner.step(self.executed)
        reward = float(np.clip(float(reward) / self.n_auv, -5.0, 2.0))
        return obs.reshape(-1).astype(np.float32), reward, bool(term), bool(trunc), info


class UnderwaterOffPolicyEnv(gym.Env):
    def __init__(self, storms, segments, depth=30.0, vmax=1.5):
        self.inner = UnderwaterFleetFlowEnv(
            storms=storms,
            segments_df=segments,
            n_auv=4,
            vmax=vmax,
            reward_mode="C",
            N=1,
            use_frame_stack=True,
            frame_k=5,
            depth_m=depth,
            duration_h=20.0,
            start_radius=200.0,
            grid_deg=0.25,
            crop_margin=3.0,
        )
        self.n_auv = 4
        self.obs_dim = self.inner.obs_dim
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (self.n_auv * self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            -1.0, 1.0, (self.n_auv * 2,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.inner.reset(seed=seed)
        return obs.reshape(-1).astype(np.float32), info

    def step(self, action):
        raw = np.asarray(action, dtype=np.float32).reshape(self.n_auv, 2)
        obs, reward, term, trunc, info = self.inner.step(raw)
        reward = float(np.clip(float(np.mean(reward)), -5.0, 5.0))
        return obs.reshape(-1).astype(np.float32), reward, bool(term), bool(trunc), info


class ProgressCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.started = time.time()
        self.last = self.started
        self.max_tree_rss = 0

    def _on_step(self):
        process = psutil.Process()
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                pass
        self.max_tree_rss = max(self.max_tree_rss, rss)
        if self.num_timesteps % 10000 == 0:
            now = time.time()
            print(
                f"steps={self.num_timesteps} "
                f"fps={10000.0 / max(now - self.last, 1e-9):.1f} "
                f"elapsed_min={(now - self.started) / 60.0:.1f}",
                flush=True,
            )
            self.last = now
        return True


def make_env(task, storms, segments, rank, seed, n_auv, vmax, depth):
    def _init():
        sub_storms = storms[rank::n_workers]
        sub = segments[segments["storm"].isin(sub_storms)].reset_index(drop=True)
        np.random.seed(seed + rank)
        if task == "surface":
            env = SurfaceOffPolicyEnv(sub_storms, sub, n_auv=n_auv, vmax=vmax)
        else:
            env = UnderwaterOffPolicyEnv(sub_storms, sub, depth=depth, vmax=vmax)
        return Monitor(env)

    return _init


def main() -> None:
    global n_workers
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("surface", "underwater"))
    parser.add_argument("algorithm", choices=("sac", "td3"))
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-auv", type=int, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--depth", type=float, default=30.0)
    args = parser.parse_args()
    n_workers = args.workers
    if args.n_auv is None:
        args.n_auv = 8 if args.task == "surface" else 4

    os.environ.setdefault("DEPLOY_OCEAN_ONLY", "1")
    os.environ.setdefault("DEPLOY_MIN_COAST_KM", "100")
    os.environ.setdefault("SAFETY_MARGIN_KM", "100")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    split = json.loads(SPLIT.read_text())
    train = split["train"]
    segments = pd.read_csv(SEGS)
    segments = segments[segments["storm"].isin(train)].reset_index(drop=True)
    if args.vmax is None:
        args.vmax = 12.0 if args.task == "surface" else 1.5

    env_fns = [
        make_env(
            args.task,
            train,
            segments,
            rank,
            args.seed,
            args.n_auv,
            args.vmax,
            args.depth,
        )
        for rank in range(args.workers)
    ]
    env = SubprocVecEnv(env_fns, start_method="fork") if args.workers > 1 else DummyVecEnv(env_fns)

    common = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=None,
        verbose=0,
        seed=args.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = SAC(**common) if args.algorithm == "sac" else TD3(**common)
    start = time.time()
    callback = ProgressCallback()
    model.learn(total_timesteps=args.steps, callback=callback)
    elapsed = time.time() - start

    OUT.mkdir(parents=True, exist_ok=True)
    depth_tag = "" if args.task == "surface" else f"_d{args.depth:g}"
    tag = (
        f"{args.task}_{args.algorithm}_n{args.n_auv}_v{args.vmax:g}"
        f"{depth_tag}_s{args.seed}"
    )
    model.save(str(OUT / f"{tag}.zip"))
    gpu_max = float(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0
    summary = {
        "task": args.task,
        "algorithm": args.algorithm,
        "seed": args.seed,
        "steps": args.steps,
        "n_env": args.workers,
        "n_auv": args.n_auv,
        "vmax_ms": args.vmax,
        "depth_m": args.depth if args.task == "underwater" else None,
        "elapsed_s": elapsed,
        "fps": args.steps / max(elapsed, 1e-9),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "max_process_tree_rss_gib": callback.max_tree_rss / 2**30,
        "gpu_max_allocated_gib": gpu_max,
        "model": f"{tag}.zip",
    }
    pd.DataFrame([summary]).to_csv(OUT / f"{tag}_training.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
