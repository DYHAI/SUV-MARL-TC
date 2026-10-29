"""4-AUV 水下追流速 —— SB3 官方 PPO 版（与 train_underwater_fleet_flow_shared.py 的
手写 IPPO 平行，用于「官方实现 vs 手写实现」的算法消融）。

与原手写版的差异（有意为之，写在文档里以便解释数字差异）：
  * 奖励：手写 IPPO 用**每台 AUV 各自的奖励**、逐 agent GAE；
    SB3 走单智能体接口，用**团队均值奖励**（÷n_auv）；critic 共享、动作逐 agent 独立采样
    （SharedPerAgentPolicy：obs (B, n_auv*obs_dim) → 同一套 MLP 逐 agent 前向）。
  * 其余全部对齐：观测 17 维（1×1 流速 + 5 帧记忆 + 自身速度）、动作 2 维 ×4、
    运动方程含平流、奖励 B/C、Vmax、深度、续航、起点布防、台风集、seed。

用法:
  SPLIT_JSON=... SEGMENTS_CSV=... python train_uwflow_sb3.py \
      <total_steps> <out_name> <vmax> <reward_mode> <depth_m> <dur_h> <seed> [grid_deg] [start_km] [n_envs]
"""
import os, sys, time, json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from underwater_fleet_flow_env import UnderwaterFleetFlowEnv
from underwater_flow_env import preload
from fleet_sb3 import SharedPerAgentPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

BASE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.environ.get("SPLIT_JSON", os.path.join(BASE, "data", "typhoon_split_64.json"))
SEGCSV = os.environ.get("SEGMENTS_CSV", os.path.join(BASE, "data", "segments_64.csv"))
OUT = os.path.join(BASE, "ppo_underwater_fleet")
os.makedirs(OUT, exist_ok=True)

TOTAL_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "ppo_uwflow_sb3"
VMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
REWARD_MODE = sys.argv[4] if len(sys.argv) > 4 else "C"
DEPTH = float(sys.argv[5]) if len(sys.argv) > 5 else 50.0
DURATION_H = float(sys.argv[6]) if len(sys.argv) > 6 else 10.0
SEED = int(sys.argv[7]) if len(sys.argv) > 7 else 0
GRID_DEG = float(sys.argv[8]) if len(sys.argv) > 8 else 0.25
START_R = float(sys.argv[9]) if len(sys.argv) > 9 else 200.0
NENV = int(sys.argv[10]) if len(sys.argv) > 10 else 4

N_AUV = 4
FRAME_K = 5
OBS_DIM = FRAME_K * 3 + 2          # 17：1×1 流速 + 5 帧记忆 + 自身速度
CROP_M = 3.0


class FlatUnderwaterFleetEnv(gym.Env):
    """把 (n_auv, obs_dim) / (n_auv, 2) 展平成 SB3 的单智能体接口（团队均值奖励）。"""

    def __init__(self, storms, segments_df, n_auv=N_AUV, vmax=VMAX, reward_mode="C",
                 depth_m=DEPTH, duration_h=DURATION_H, start_radius=START_R,
                 grid_deg=GRID_DEG, crop_margin=CROP_M, reward_clip=(-5.0, 5.0)):
        self.inner = UnderwaterFleetFlowEnv(
            storms=storms, segments_df=segments_df, n_auv=n_auv, vmax=vmax,
            reward_mode=reward_mode, N=1, use_frame_stack=True, frame_k=FRAME_K,
            depth_m=depth_m, duration_h=duration_h, start_radius=start_radius,
            grid_deg=grid_deg, crop_margin=crop_margin)
        self.reward_clip = reward_clip
        self.n_auv = n_auv
        self.obs_dim = self.inner.obs_dim
        self.act_dim = 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(n_auv * self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(n_auv * self.act_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.inner.reset(seed=seed)
        return obs.reshape(-1).astype(np.float32), info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(self.n_auv, self.act_dim)
        obs, reward, term, trunc, info = self.inner.step(a)
        r = float(np.mean(reward))          # 团队均值（÷n_auv）
        if self.reward_clip is not None:
            r = float(np.clip(r, self.reward_clip[0], self.reward_clip[1]))
        return obs.reshape(-1).astype(np.float32), r, bool(term), bool(trunc), info


def make_env(storms, seg_df, worker_id, n_envs, **kw):
    def _init():
        sub = storms[worker_id::n_envs]
        sseg = seg_df[seg_df.storm.isin([s for s in sub])].reset_index(drop=True)
        return FlatUnderwaterFleetEnv(storms=sub, segments_df=sseg, **kw)
    return _init


class LogCallback:
    def __init__(self):
        self.n = 0

    def __call__(self, _locals, _globals):
        self.n += 1
        if self.n % 4 == 0:
            m = _locals["self"]
            print(f"[{time.strftime('%H:%M:%S')}] steps={m.num_timesteps}", flush=True)
        return True


def main():
    split = json.load(open(SPLIT))
    train = split["train"]
    seg = pd.read_csv(SEGCSV)
    print(f"[SB3] train storms={len(train)} vmax={VMAX} reward={REWARD_MODE} "
          f"depth={DEPTH}m dur={DURATION_H}h seed={SEED} grid={GRID_DEG} NENV={NENV}", flush=True)

    kw = dict(n_auv=N_AUV, vmax=VMAX, reward_mode=REWARD_MODE, depth_m=DEPTH,
              duration_h=DURATION_H, start_radius=START_R, grid_deg=GRID_DEG)
    # 预加载：fork 出的 worker 通过 COW 共享，省掉 19min 读盘
    if os.environ.get("PRELOAD", "1") == "1":
        t_p = time.time()
        preload(train, use_uv=True, use_anomaly=False, depth_m=DEPTH,
                crop_margin=CROP_M, grid_deg=GRID_DEG, native_res=False)
        print(f"preloaded {len(train)} storms in {time.time()-t_p:.0f}s", flush=True)

    venv = SubprocVecEnv([make_env(train, seg, i, NENV, **kw) for i in range(NENV)],
                         start_method="fork")
    policy_kwargs = dict(n_auv=N_AUV, obs_dim=OBS_DIM, act_dim=2, hidden=256)
    model = PPO(SharedPerAgentPolicy, venv, policy_kwargs=policy_kwargs,
                n_steps=2048, batch_size=256, n_epochs=10,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, clip_range_vf=0.2, ent_coef=0.01, vf_coef=0.5,
                max_grad_norm=0.5, verbose=0, seed=SEED, device="cpu")
    model.learn(total_timesteps=TOTAL_STEPS, callback=LogCallback())
    out = os.path.join(OUT, OUT_NAME if OUT_NAME.endswith(".zip") else OUT_NAME + ".zip")
    model.save(out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
