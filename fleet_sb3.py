"""Shared per-agent actor-critic used by the final surface and underwater IPPO runs."""

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


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
