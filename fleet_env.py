"""9-AUV 集群编队追踪台风环境（海面，只观测风速+风向）。

设计（对齐水面单 AUV，台风位置不进入观测、只在奖励里出现）：
- 9 个 AUV，每个 AUV 独立控制（各自输出自己的动作）。
- 观测(每个 AUV，29 维)：集群 9 个点的风场信息(共享) 9*(spd,u,v)/50 = 27 + 自己的速度(vx,vy)/20 = 2。
- 每个 AUV 只传感自己单点的风速+风向，通过共享知道整个集群 9 个点的风场。
- 约束编队在一定半径范围内（离台风眼越远越惩罚）。
- 奖励：追踪台风(离眼越近越好) + 动作平滑(减少抖动) + 编队半径 + 陆地软惩罚。
动作(每个 AUV)：2 维 [-1,1] -> 期望速度 vx/vy = act * VMAX。
"""
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from surface_wind_env import load_storm, land_ocean_mask, D2KM, STEP_MIN, LAND_PENALTY, R_BUFFER, W_BUFFER

N_AUV = 9
VMAX = 20.0          # m/s
FLOW_NORM = 50.0     # 风速归一化（海表风速 ~0-50 m/s）
R_FORM = 200.0       # km，编队半径约束
W_FORM = 1.0         # 编队半径外漂移惩罚权重
START_RADIUS = 200.0 # km，初始布防半径


class FleetEnv(gym.Env):
    def __init__(self, storms, use_uv=True, vmax=VMAX, preloaded=None, n_auv=N_AUV, smooth_penalty=0.0):
        super().__init__()
        self.n_auv = n_auv
        if preloaded is not None:
            self.data = preloaded
        else:
            self.data = {s: load_storm(s, use_uv=use_uv) for s in storms}
        self.storms = list(self.data.keys())
        self.vmax = vmax
        self.use_uv = use_uv
        self.smooth_penalty = smooth_penalty
        self.prev_actions = np.zeros((n_auv, 2), dtype=np.float32)
        of, dist_to_land, elon, elat = land_ocean_mask()
        self.dist_interp = RegularGridInterpolator((elat, elon), dist_to_land, bounds_error=False, fill_value=0.0)

        # 每个 AUV：27(集群风场) + 2(自己速度)；台风位置不进入观测
        self.obs_dim = n_auv * 3 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_auv, self.obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_auv, 2), dtype=np.float32)
        self.step_count = 0

    def _wind_at(self, d, t, lon, lat):
        h0 = int(d["h0"][t]); h1 = int(d["h1"][t]); fr = float(d["frac"][t])
        spd = (1 - fr) * d["spd"][h0] + fr * d["spd"][h1]
        uu = (1 - fr) * d["u"][h0] + fr * d["u"][h1]
        vv = (1 - fr) * d["v"][h0] + fr * d["v"][h1]
        lon_g = d["lon"]; lat_g = d["lat"]
        Is = RegularGridInterpolator((lat_g, lon_g), spd, bounds_error=False, fill_value=0.0)
        Iu = RegularGridInterpolator((lat_g, lon_g), uu, bounds_error=False, fill_value=0.0)
        Iv = RegularGridInterpolator((lat_g, lon_g), vv, bounds_error=False, fill_value=0.0)
        p = np.array([[lat, lon]])
        return float(Is(p)[0]), float(Iu(p)[0]), float(Iv(p)[0])

    def _obs(self):
        d = self.d; t = self.t
        obs = np.zeros((self.n_auv, self.obs_dim), dtype=np.float32)
        wind = np.zeros((self.n_auv, 3), dtype=np.float32)
        for i in range(self.n_auv):
            s, u, v = self._wind_at(d, t, self.lons[i], self.lats[i])
            wind[i] = [s, u, v]
        for i in range(self.n_auv):
            obs[i, :self.n_auv*3] = (wind / FLOW_NORM).ravel()
            obs[i, self.n_auv*3:self.n_auv*3+2] = [self.vxs[i]/self.vmax, self.vys[i]/self.vmax]
        return obs

    def _reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s = np.random.choice(self.storms)
        self.d = self.data[s]
        t0 = int(np.where(self.d["in_domain"])[0][0])
        self.t = t0
        elo = self.d["track_lon"][t0]; ela = self.d["track_lat"][t0]
        self.lons = np.zeros(self.n_auv); self.lats = np.zeros(self.n_auv)
        self.vxs = np.zeros(self.n_auv); self.vys = np.zeros(self.n_auv)
        for i in range(self.n_auv):
            ang = 2 * np.pi * i / self.n_auv + np.random.uniform(0, 2*np.pi/self.n_auv)
            r = np.random.uniform(0.5, 1.0) * START_RADIUS
            self.lons[i] = elo + r*np.cos(ang)/(D2KM*np.cos(np.deg2rad(ela)))
            self.lats[i] = ela + r*np.sin(ang)/D2KM
        lo = self.d["lon"]; la = self.d["lat"]
        self.lons = np.clip(self.lons, lo.min(), lo.max())
        self.lats = np.clip(self.lats, la.min(), la.max())
        self.prev_actions = np.zeros((self.n_auv, 2), dtype=np.float32)
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, -1, 1)
        a_change = a - self.prev_actions
        self.prev_actions = a.copy()
        self.vxs = a[:, 0] * self.vmax
        self.vys = a[:, 1] * self.vmax
        dt = float(STEP_MIN * 60)
        dx_km = self.vxs * dt / 1000.0
        dy_km = self.vys * dt / 1000.0
        self.lons += dx_km / (D2KM * np.cos(np.deg2rad(self.lats)))
        self.lats += dy_km / D2KM
        lo = self.d["lon"]; la = self.d["lat"]
        self.lons = np.clip(self.lons, lo.min(), lo.max())
        self.lats = np.clip(self.lats, la.min(), la.max())
        self.t = min(self.t + 1, self.d["T"] - 1)
        self.step_count += 1

        eye_lon = self.d["track_lon"][self.t]; eye_lat = self.d["track_lat"][self.t]
        dx = (self.lons - eye_lon) * D2KM * np.cos(np.deg2rad(eye_lat))
        dy = (self.lats - eye_lat) * D2KM
        dist = np.hypot(dx, dy)
        rewards = -np.minimum(dist, 300.0) / 100.0
        rewards += (dist < 100.0).astype(np.float32)
        rewards -= W_FORM * np.maximum(0.0, dist - R_FORM) / 100.0
        if self.smooth_penalty > 0:
            rewards -= self.smooth_penalty * np.sum(a_change * a_change, axis=1)
        land_dist = self.dist_interp(np.stack([self.lats, self.lons], axis=1))
        on_land = land_dist <= 0.0
        rewards = np.where(on_land, rewards - LAND_PENALTY,
                           rewards - W_BUFFER * np.maximum(0.0, 1.0 - land_dist / R_BUFFER))

        terminated = False
        truncated = (self.t >= self.d["T"] - 1) or (not bool(self.d["in_domain"][self.t])) or (self.t >= self.d["landfall"])
        info = {"dist_km": dist, "eye_lon": eye_lon, "eye_lat": eye_lat, "storm": self.d["storm"],
                "rewards": rewards}
        return self._obs(), float(rewards.sum()), terminated, truncated, info

    def reset(self, seed=None, options=None):
        return self._reset(seed=seed, options=options)
