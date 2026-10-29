"""9-AUV 集群编队追踪台风眼 —— SB3 PPO + 并行环境 + 参数共享（逐 agent 独立决策）。

关键点：
- FlatFleetEnv 把 (n_auv, obs_dim)/(n_auv, act_dim) 展平成单智能体接口，便于 SB3。
- SharedPerAgentPolicy：obs (B, n_auv*obs_dim) → 逐 agent 用同一套 MLP 独立决策 → act (B, n_auv*act_dim)。
  actor 与 critic 都是 per-agent 共享；团队价值 = 各 agent 价值之和。
- SubprocVecEnv 并行，每个子进程只加载 storms 的一个子集（省内存）。
"""
import os, sys, time, json
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fleet_env import FleetEnv
from surface_wind_env import load_storm
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

BASE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(BASE, "data", "typhoon_split_64.json")


class FlatFleetEnv(gym.Env):
    """把 FleetEnv 的 (n_auv, obs_dim) 观测和 (n_auv, act_dim) 动作展平为 SB3 单接口。"""
    def __init__(self, storms, n_auv=9, vmax=20.0, smooth_penalty=0.0, reward_clip=(-5.0, 2.0)):
        self.inner = FleetEnv(storms=storms, use_uv=True, n_auv=n_auv, smooth_penalty=smooth_penalty)
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
        # 均值奖励（÷n_auv），再裁剪到 reward_clip
        reward = float(reward) / self.n_auv
        if self.reward_clip is not None:
            reward = float(np.clip(reward, self.reward_clip[0], self.reward_clip[1]))
        return obs.reshape(-1).astype(np.float32), reward, term, trunc, info


class SharedPerAgentPolicy(ActorCriticPolicy):
    """参数共享：同一套 actor/critic 逐 agent 独立前向；团队价值=各 agent 价值之和。"""
    def __init__(self, observation_space, action_space, lr_schedule,
                 n_auv=9, obs_dim=29, act_dim=2, hidden=256, log_std_init=0.0,
                 ortho_init=True, optimizer_class=th.optim.Adam, optimizer_kwargs=None, **kwargs):
        self.n_auv = n_auv
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.log_std_init = log_std_init
        super().__init__(observation_space, action_space, lr_schedule,
                         net_arch=[], ortho_init=ortho_init,
                         optimizer_class=optimizer_class, optimizer_kwargs=optimizer_kwargs,
                         **kwargs)

    def _build(self, lr_schedule):
        # 共享 per-agent actor/critic（不再用 mlp_extractor）
        self.actor_net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden), nn.Tanh(),
            nn.Linear(self.hidden, self.hidden), nn.Tanh(),
            nn.Linear(self.hidden, self.act_dim),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden), nn.Tanh(),
            nn.Linear(self.hidden, self.hidden), nn.Tanh(),
            nn.Linear(self.hidden, 1),
        )
        self.log_std = nn.Parameter(th.ones(self.act_dim) * self.log_std_init)
        # 初始化
        for m in [self.actor_net, self.critic_net]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        with th.no_grad():
            self.actor_net[-1].weight.mul_(0.01)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _reshape(self, obs):
        return obs.reshape(-1, self.n_auv, self.obs_dim)

    def _actor(self, obs):
        x = self._reshape(obs)                       # (B, n, obs_dim)
        B, n, _ = x.shape
        mean = self.actor_net(x.reshape(B * n, self.obs_dim))   # (B*n, act_dim)
        return mean.reshape(B, n, self.act_dim)

    def _critic(self, obs):
        x = self._reshape(obs)
        B, n, _ = x.shape
        v = self.critic_net(x.reshape(B * n, self.obs_dim)).reshape(B, n)  # (B, n)
        return v.sum(dim=-1, keepdim=True)           # (B, 1)

    def _dist(self, mean):
        B, n, a = mean.shape
        mean_flat = mean.reshape(B, n * a)
        log_std = self.log_std.unsqueeze(0).expand(n, a).reshape(n * a)
        return self.action_dist.proba_distribution(mean_flat, log_std)

    def forward(self, obs, deterministic=False):
        mean = self._actor(obs)
        dist = self._dist(mean)
        actions = dist.get_actions(deterministic=deterministic)   # (B, n*act) 展平，匹配 action_space
        values = self._critic(obs)
        log_prob = dist.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        mean = self._actor(obs)
        dist = self._dist(mean)
        aflat = actions.reshape(-1, self.n_auv * self.act_dim)
        values = self._critic(obs)
        log_prob = dist.log_prob(aflat)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs):
        return self._critic(obs)

    def get_distribution(self, obs):
        return self._dist(self._actor(obs))

    def _predict(self, observation, deterministic=False):
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(dict(n_auv=self.n_auv, obs_dim=self.obs_dim, act_dim=self.act_dim,
                         hidden=self.hidden, log_std_init=self.log_std_init))
        return data


def make_env(storms, n_auv, smooth_penalty, worker_id, n_envs):
    def _init():
        # 每个子进程只加载 storms 的一个子集，省内存
        sub = storms[worker_id::n_envs]
        return FlatFleetEnv(storms=sub, n_auv=n_auv, smooth_penalty=smooth_penalty)
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


def main(n_auv=9, total_steps=1_500_000, out_name="ppo_fleet_sb3_9auv.zip", n_envs=4, seed=0, smooth=0.0):
    split = json.load(open(SPLIT))
    train = split["train"]
    obs_dim = n_auv * 3 + 2
    env = SubprocVecEnv([make_env(train, n_auv, smooth, i, n_envs) for i in range(n_envs)])
    policy_kwargs = dict(n_auv=n_auv, obs_dim=obs_dim, act_dim=2, hidden=256)
    model = PPO(SharedPerAgentPolicy, env, policy_kwargs=policy_kwargs,
                n_steps=2048, batch_size=256, n_epochs=10,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, clip_range_vf=0.2, ent_coef=0.01, vf_coef=0.5,
                max_grad_norm=0.5, verbose=1, seed=seed, device="cpu")
    model.learn(total_timesteps=total_steps, callback=LogCallback())
    os.makedirs(os.path.join(BASE, "ppo_fleet"), exist_ok=True)
    out = os.path.join(BASE, "ppo_fleet", out_name)
    model.save(out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_auv", type=int, default=9)
    ap.add_argument("--total_steps", type=int, default=1_500_000)
    ap.add_argument("--out_name", default="ppo_fleet_sb3_9auv.zip")
    ap.add_argument("--n_envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smooth", type=float, default=0.0)
    a = ap.parse_args()
    main(n_auv=a.n_auv, total_steps=a.total_steps, out_name=a.out_name,
         n_envs=a.n_envs, seed=a.seed, smooth=a.smooth)
