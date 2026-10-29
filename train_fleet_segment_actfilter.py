"""10min / 4-AUV：**MLP + 动作平滑**（对策略输出做一阶低通滤波）—— 第三组对照。

  执行动作： a_exec_t = (1-beta) * a_policy_t  +  beta * a_exec_{t-1}

网络与训练完全沿用基线 `train_fleet_segment.py`（PPO + SharedPerAgentPolicy 逐 agent 共享 MLP），
奖励、数据、布阵、超参**一律不动**；唯一新增的是这个动作滤波器（给系统"动作惯性"，
不引入任何可学习记忆、参数量不变）。

用法: python train_fleet_segment_actfilter.py <total_steps> <out_name.zip> [beta] [n_auv] [deploy]
"""
import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fleet_segment_env import FleetSegmentEnv, STEP_MIN, VMAX
from fleet_sb3 import SharedPerAgentPolicy

BASE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.environ.get("SPLIT_JSON", os.path.join(BASE, "data", "typhoon_split_64.json"))
SEGCSV = os.environ.get("SEGMENTS_CSV", os.path.join(BASE, "data", "segments_64.csv"))
OUT = os.path.join(BASE, "ppo_fleet")

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "ppo_fleet_actfilter.zip"
BETA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
N_AUV = int(sys.argv[4]) if len(sys.argv) > 4 else 4
DEPLOY = float(sys.argv[5]) if len(sys.argv) > 5 else 200.0

SPACING, PATCH_N, SMOOTH, RING_RADIUS, FORMATION, HIDDEN = 0.25, 1, 0.05, 100.0, "ring", 256
NENV = int(os.environ.get("NENV", "4"))
# 观测通道开关（2026-10-09 起支持）：SST_MODE=none/sst/anom/both，USE_MSLP=0/1
# 用途：主线最优配置 = 风(u/v/spd) + SST距平 + MSLP（见 README §11/§18）
SST_MODE = os.environ.get("SST_MODE", "none")
USE_MSLP = os.environ.get("USE_MSLP", "0") == "1"
# 陆罚扫参用：LAND_OUTSIDE_CLIP=1 时，**先对任务奖励裁剪(-5,2)、再把陆罚整体加上去**。
# 为什么必须这样：扁平奖励 = clip(各 AUV 奖励均值, -5, 2)，而陆罚是 `-P` 直接进单台奖励，
# 所以只要 P 大到把均值压过 -5 就被裁剪吃掉 —— 实测 P=20/40/640 在「一台 AUV 上岸」时
# 扁平奖励**都是 -5.00**，即 P≥~40 的扫参在旧口径下会得到完全相同的模型。
LAND_OUTSIDE_CLIP = os.environ.get("LAND_OUTSIDE_CLIP", "0") == "1"
N_STEPS, BATCH, N_EPOCHS = 2048, 256, 10          # 与基线一致
LR, GAMMA, GAE_LAMBDA, CLIP, CLIP_VF = 3e-4, 0.99, 0.95, 0.2, 0.2
# 随机种子：默认 0；多 seed 复现实验用环境变量覆盖（SEED=1/2…），
# 它同时决定 Python/NumPy 种子、每个 env 的 reset 种子(SEED+rank) 和 PPO 的 seed。
ENT, VF, MAXGRAD = 0.01, 0.5, 0.5
SEED = int(os.environ.get("SEED", "0"))


class FlatFleetSegmentEnvActFilter(gym.Env):
    """在基线 FlatFleetSegmentEnv 的基础上，只多一个动作低通滤波。"""

    def __init__(self, storms, segments_df, n_auv=N_AUV, beta=BETA):
        super().__init__()
        self.inner = FleetSegmentEnv(storms=storms, segments_df=segments_df, use_uv=True,
                                     n_auv=n_auv, smooth_penalty=SMOOTH, spacing_deg=SPACING,
                                     patch_n=PATCH_N, deploy_radius=DEPLOY,
                                     formation=FORMATION, ring_radius=RING_RADIUS,
                                     sst_mode=SST_MODE, use_mslp=USE_MSLP)
        self.n_auv = n_auv
        self.beta = float(beta)
        self.obs_dim = self.inner.obs_dim
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_auv * self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (n_auv * 2,), dtype=np.float32)
        self.a_exec = np.zeros((n_auv, 2), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.inner.reset(seed=seed)
        self.a_exec = np.zeros((self.n_auv, 2), dtype=np.float32)
        return obs.reshape(-1).astype(np.float32), info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(self.n_auv, 2)
        self.a_exec = (1.0 - self.beta) * a + self.beta * self.a_exec     # ← 唯一新增：动作低通
        a_use = np.clip(self.a_exec, -1.0, 1.0)
        obs, reward, term, trunc, info = self.inner.step(a_use)
        if LAND_OUTSIDE_CLIP:
            per = info["rewards"].astype(np.float32); lp = info["land_pen"].astype(np.float32)
            base = float(np.clip(float((per - lp).sum()) / self.n_auv, -5.0, 2.0))
            reward = base + float(lp.sum()) / self.n_auv       # 陆罚不被裁剪吃掉
        else:
            reward = float(np.clip(float(reward) / self.n_auv, -5.0, 2.0))
        return obs.reshape(-1).astype(np.float32), reward, term, trunc, info


def make_env(storms, segments_df, rank):
    def _init():
        # 与基线 train_fleet_segment.py 一致的**分片**：每个 env 只加载 storms[rank::NENV]。
        # 旧版给每个 env 都塞整份 train 列表 → NENV 份重复加载（52 台风时 4×52=208 次 load），
        # 200 台风下会直接把内存打爆；同时这也让「动作平滑组」与基线组的 env 数据划分不一致。
        sub = storms[rank::NENV]
        env = FlatFleetSegmentEnvActFilter(sub, segments_df)
        env.reset(seed=SEED + rank)
        return env
    return _init


class LogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.t0 = time.time()

    def _on_step(self):
        if self.n_calls % 1024 == 0:
            el = max(time.time() - self.t0, 1e-9)
            print("[%s] steps=%d  fps=%.0f" % (time.strftime("%H:%M:%S"), self.num_timesteps,
                                               self.num_timesteps / el), flush=True)
        return True


def main():
    random.seed(SEED); np.random.seed(SEED)
    split = json.load(open(SPLIT))
    train = split["train"]
    seg = pd.read_csv(SEGCSV)
    seg = seg[seg.storm.isin(train)].reset_index(drop=True)
    # obs 维度：每个 cell 3 维风(spd,u,v)；OBS_COAST_DIST=1 再 +1 维「离岸距离」；
    #           OBS_AGENT_ID=1 再 +N_AUV 维「身份 one-hot」
    # 每 cell 通道数：3 风(spd,u,v) + SST(1 或 2) + MSLP(3) + 离岸距离(1)
    n_sst = 2 if SST_MODE == "both" else (1 if SST_MODE in ("sst", "anom") else 0)
    nch = (3 + n_sst + (3 if USE_MSLP else 0)
           + (1 if os.environ.get("OBS_COAST_DIST", "0") == "1" else 0))
    # 身份编码："onehot"/"1" → +n_auv；"angle" → +2；"both" → +n_auv+2
    _mode = os.environ.get("OBS_AGENT_ID", "0")
    if _mode == "1":
        _mode = "onehot"
    n_id = (N_AUV if _mode in ("onehot", "both") else 0) + (2 if _mode in ("angle", "both") else 0)
    od = nch * N_AUV + 2 + n_id
    print("[config] MLP + 动作低通滤波  STEP_MIN=%d VMAX=%.2fm/s n_auv=%d deploy=%.0fkm  beta=%.2f  hidden=%d  "
          "n_steps=%d batch=%d n_env=%d total=%d seed=%d  coast_obs=%d  sst=%s  mslp=%d  agent_id=%s"
          % (STEP_MIN, VMAX, N_AUV, DEPLOY, BETA, HIDDEN, N_STEPS, BATCH, NENV, TOTAL, SEED,
             int(os.environ.get("OBS_COAST_DIST", "0") == "1"), SST_MODE, int(USE_MSLP),
             os.environ.get("OBS_AGENT_ID", "0")), flush=True)
    env = SubprocVecEnv([make_env(train, seg, i) for i in range(NENV)], start_method="fork")
    model = PPO(SharedPerAgentPolicy, env,
                policy_kwargs=dict(n_auv=N_AUV, obs_dim=od, act_dim=2, hidden=HIDDEN),
                n_steps=N_STEPS, batch_size=BATCH, n_epochs=N_EPOCHS,
                learning_rate=LR, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
                clip_range=CLIP, clip_range_vf=CLIP_VF, ent_coef=ENT, vf_coef=VF,
                max_grad_norm=MAXGRAD, verbose=0, seed=SEED, device="cpu")
    npar = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print("[config] 策略参数量 = %d" % npar, flush=True)
    model.learn(total_timesteps=TOTAL, callback=LogCallback())
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, OUT_NAME[:-4] if OUT_NAME.endswith(".zip") else OUT_NAME)
    model.save(p)
    print("saved", p + ".zip", flush=True)


if __name__ == "__main__":
    main()
