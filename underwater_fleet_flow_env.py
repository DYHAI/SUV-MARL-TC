"""4-AUV 水下 50m 追流速环境（IPPO，独立奖励，含流速平流，多种奖励模式）。

阶段二任务：海表追眼段结束时（t1），4 AUV 以台风眼为中心 200km 内随机布防、下潜到 50m，
固定 10 小时追「附近流速较大的区域」。

- 观测(每台, 29 维) = 3×3 patch 的 50m 流速(spd/u/v) + 自身速度(vx/vy)。不含台风位置。
- 动作(每台, 2 维) = 推进速度方向 [-1,1]²。
- 运动方程：绝对速度 = 推进速度(动作×Vmax) + 50m 背景流速（顺流加速/逆流减速）。
- 奖励(每台独立)：
    A: r = |V(x)| / V_NORM
    B: r = clip(|V|/V_NORM,0,RMAX) - λ|a|² - 陆地惩罚
    C: B + β·max(0, 推进方向·流速方向)  (顺流追踪)
    F: 爬坡/效率奖励
    G: 信息增益奖励（P3「找到核心区域去测量」）——
         r = w_cov·W(x)·[新增覆盖格点] + w_nov·W(x)·clip(d_nov/nov_ref,0,1)
             + w_var·| |V(x)| - mean_team|V| | / V_NORM - λ|a|² - 陆地惩罚
       其中 W(x)∈[0,1] 为该台风段的「信息价值图」= 0.5·(时均流速超出中位的异常)
       + 0.5·(时均流速的 3×3 局部空间方差)，高值区 = 台风海洋响应核心区。
       d_nov = 到「团队历史采样点」的最近距离（不含自己当前点）→ 奖励覆盖新区域、
       自动产生编队分散。该奖励下 4 台 AUV 的观测空间完全不变。
"""
import os, sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from underwater_flow_env import load_storm, D2KM, STEP_MIN, LAND_PENALTY, R_BUFFER, W_BUFFER

N_AUV = 4
VMAX = 2.0
V_NORM = 1.0          # 50m 流速归一化（m/s）
RMAX = 2.0            # 流速奖励限幅
LAMBDA_ENERGY = 0.05  # 能耗惩罚系数 λ|a|²
BETA_ALIGN = 0.5      # 顺流追踪系数 β
CLIMB_COEF = 1.0        # 爬坡奖励系数（流速提升为正时加分）
R_FLOW_W = 1.0        # 效率奖励 F：流速增益(climb) 权重
LAMBDA_EFF = 1.0     # 效率奖励 F：能耗惩罚系数 λ|a|²
START_RADIUS = 200.0  # 起点：眼 200km 内布防
DURATION_H = 10.0     # 固定 10 小时
DEPTH_M = 50
# ---- P3 信息增益奖励（reward_mode="G"）参数 ----
INFO_W_COV = 1.0      # 「覆盖面积」权重：新增覆盖格点 × 信息价值 W
INFO_W_NOV = 0.5      # 「新颖度」权重：到团队历史最近距离 × W（连续梯度，防原地打转）
INFO_W_VAR = 0.3      # 「流速方差」权重：团队采样流速的空间离散度
COV_DEG = 0.05        # 覆盖格点尺寸（°）≈5.5km，用于统计「新增覆盖面积」
NOV_REF_KM = 30.0     # 新颖度归一化参考距离（km）


class UnderwaterFleetFlowEnv(gym.Env):
    def __init__(self, storms, segments_df, n_auv=N_AUV, vmax=VMAX, reward_mode="B",
                 spacing_deg=0.25, N=3, start_radius=START_RADIUS, duration_h=DURATION_H, depth_m=DEPTH_M,
                 use_anomaly=False, grid_deg=0.25, crop_margin=3.0, use_bearing=False,
                 use_frame_stack=False, frame_k=5, use_flowdiff=False, native_res=False, storage16=False,
                 lambda_eff=None, r_flow_w=None, lambda_energy=None,
                 info_w_cov=INFO_W_COV, info_w_nov=INFO_W_NOV, info_w_var=INFO_W_VAR,
                 cov_deg=COV_DEG, nov_ref_km=NOV_REF_KM):
        # P4：能耗惩罚系数 λ（自适应/湍流的关键旋钮）。不传则用类默认 0.05
        self.lambda_energy = LAMBDA_ENERGY if lambda_energy is None else float(lambda_energy)
        super().__init__()
        self.n_auv = n_auv
        self.vmax = vmax
        self.reward_mode = reward_mode.upper()
        self.spacing_deg = spacing_deg
        self.N = N
        self.start_radius = start_radius
        self.duration_h = duration_h
        self.depth_m = depth_m
        self.max_steps = int(duration_h * 60 / STEP_MIN)
        self.data = {s: load_storm(s, use_uv=True, depth_m=depth_m, crop_margin=crop_margin, use_anomaly=use_anomaly, grid_deg=grid_deg, native_res=native_res) for s in storms}
        if storage16:
            # load_storm 现在带进程内缓存，就地改 dict 会污染缓存 → 先浅拷贝
            self.data = {k: dict(v) for k, v in self.data.items()}
            for _d in self.data.values():
                _d["u"]=_d["u"].astype(np.float16); _d["v"]=_d["v"].astype(np.float16)
                _d["spd"]=_d["spd"].astype(np.float16); _d["dist_to_land"]=_d["dist_to_land"].astype(np.float16)
        self.storage16=storage16
        self.lambda_eff = LAMBDA_EFF if lambda_eff is None else float(lambda_eff)
        self.r_flow_w = R_FLOW_W if r_flow_w is None else float(r_flow_w)
        # P3 信息增益奖励参数
        self.info_w_cov = float(info_w_cov)
        self.info_w_nov = float(info_w_nov)
        self.info_w_var = float(info_w_var)
        self.cov_deg = float(cov_deg)
        self.nov_ref_km = float(nov_ref_km)
        self.W = None          # 信息价值图（每 episode 由 reset 计算）
        self._cov_cells = set()
        self._hist_lat = []
        self._hist_lon = []
        self._cell_area_km2 = (self.cov_deg * D2KM) ** 2


        self.storms = list(self.data.keys())
        self.segments = segments_df[segments_df.storm.isin(self.storms)].reset_index(drop=True)
        extra = (3 if use_flowdiff else 0)
        if use_frame_stack and N > 1:
            self.obs_dim = 3 * N * N + frame_k * 3 + 2 + extra
        elif use_frame_stack:
            self.obs_dim = frame_k * 3 + 2 + extra
        else:
            self.obs_dim = 3 * N * N + 2 + (2 if use_bearing else 0)
        self.use_flowdiff = use_flowdiff
        self.use_bearing = use_bearing
        self.use_frame_stack = use_frame_stack
        self.frame_k = frame_k
        # 每 AUV 最近 K 帧 (spd,u,v) 列表。
        # 注意：必须初始化为空列表（而非 [None]*n_auv），否则
        # use_frame_stack=False 时 step() 里的 flow_hist[i].append(...) 会崩。
        self.flow_hist = [[] for _ in range(n_auv)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_auv, self.obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_auv, 2), dtype=np.float32)
        self.t = 0
        self.t_end = 0
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        row = self.segments.iloc[np.random.randint(len(self.segments))]
        s = row["storm"]
        self.d = self.data[s]
        # 段结束时的台风眼位置（segments 的 end_lon/end_lat）
        elo = float(row["end_lon"]); ela = float(row["end_lat"])
        # 在 HYCOM 时间网格上找离段结束位置最近的 10min 时刻（时间对齐）
        tlon = self.d["track_lon"]; tlat = self.d["track_lat"]
        dx = (tlon - elo) * D2KM * np.cos(np.deg2rad(ela))
        dy = (tlat - ela) * D2KM
        t_start = int(np.argmin(np.hypot(dx, dy)))
        self.t = t_start
        lf = int(self.d["landfall"])
        self.t_end = min(t_start + self.max_steps, self.d["T"] - 1, lf if lf > t_start else self.d["T"] - 1)
        self.lons = np.zeros(self.n_auv); self.lats = np.zeros(self.n_auv)
        for i in range(self.n_auv):
            ang = np.random.uniform(0, 2 * np.pi); r = np.random.uniform(0, self.start_radius)
            self.lons[i] = elo + r * np.cos(ang) / (D2KM * np.cos(np.deg2rad(ela)))
            self.lats[i] = ela + r * np.sin(ang) / D2KM
        lo = self.d["lon"]; la = self.d["lat"]
        self.lons = np.clip(self.lons, lo.min(), lo.max())
        self.lats = np.clip(self.lats, la.min(), la.max())
        self.vxs = np.zeros(self.n_auv); self.vys = np.zeros(self.n_auv)
        self.step_count = 0
        # P3 信息增益：重置覆盖/历史，并构建本段的「信息价值图」W
        self._cov_cells = set()
        self._hist_lat = []
        self._hist_lon = []
        self._build_info_map()
        if self.use_frame_stack:
            self.flow_hist = [[] for _ in range(self.n_auv)]
            for i in range(self.n_auv):
                s0 = self._at2("spd", self.lats[i], self.lons[i])
                u0 = self._at2("u", self.lats[i], self.lons[i])
                v0 = self._at2("v", self.lats[i], self.lons[i])
                self.flow_hist[i].append(np.array([s0, u0, v0], dtype=np.float32))
        return self._obs(), {}

    def _build_info_map(self):
        """构建本 episode 的「信息价值图」W（P3：核心区域 = 高流速异常 + 高局部方差）。

        W(lat,lon) = 0.5 * clip((spd_t - median)/(p95 - median), 0, 1)
                   + 0.5 * clip(var3x3(spd_t) / p95(var3x3), 0, 1)
        其中 spd_t 为 episode 时间窗 [t, t_end] 的时均流速（台风海洋响应尾迹区）。
        仅陆地/无数据格点为 0。W 只用于奖励塑形，AUV 观测空间不含 W。
        """
        from scipy.ndimage import uniform_filter
        d = self.d
        spd = d["spd"]
        if self.storage16:
            spd = spd.astype(np.float32)
        # 注意：d["spd"] 是 3 小时一帧的原始场，而 self.t 是 10min 网格索引，
        # 必须用 h0 映射到 3 小时索引，否则切片为空。
        t0 = int(self.t)
        t1 = int(min(max(self.t_end, t0), len(self.d["h0"]) - 1))
        h0i = int(self.d["h0"][t0])
        h1i = int(min(self.d["h0"][t1], spd.shape[0] - 1))
        m = spd[h0i:h1i + 1].mean(axis=0).astype(np.float32)
        self.M_mean = m          # episode 时间窗时均 |V|（评测/重构用的“真值场”）
        ocean = d["dist_to_land"] > 0.0
        vals = m[ocean]
        if vals.size < 10:
            self.W = np.zeros_like(m, dtype=np.float32)
            return
        med = float(np.median(vals))
        p95 = float(np.percentile(vals, 95))
        w1 = np.clip((m - med) / max(p95 - med, 1e-6), 0.0, 1.0)
        m2 = uniform_filter(m, size=3)
        mvar = np.maximum(uniform_filter(m * m, size=3) - m2 * m2, 0.0)
        v95 = float(np.percentile(mvar[ocean], 95))
        w2 = np.clip(mvar / max(v95, 1e-12), 0.0, 1.0)
        w = 0.5 * w1 + 0.5 * w2
        self.W = np.where(ocean, w, 0.0).astype(np.float32)

    def _interp_fields(self):
        h0 = int(self.d["h0"][self.t]); h1 = int(self.d["h1"][self.t]); fr = float(self.d["frac"][self.t])
        spd = (1 - fr) * self.d["spd"][h0] + fr * self.d["spd"][h1]
        uu = (1 - fr) * self.d["u"][h0] + fr * self.d["u"][h1]
        vv = (1 - fr) * self.d["v"][h0] + fr * self.d["v"][h1]
        if self.storage16:
            spd = spd.astype(np.float32); uu = uu.astype(np.float32); vv = vv.astype(np.float32)
        return spd, uu, vv

    def _at2(self, key, lat, lon):
        """单点两帧线性插值：只插 h0/h1 两个场再 fr 加权，避免每步混合全场。key=spd/u/v"""
        h0 = int(self.d["h0"][self.t]); h1 = int(self.d["h1"][self.t]); fr = float(self.d["frac"][self.t])
        f0 = self.d[key][h0]; f1 = self.d[key][h1]
        if self.storage16:
            f0 = f0.astype(np.float32); f1 = f1.astype(np.float32)
        return (1 - fr) * self._at(f0, lat, lon) + fr * self._at(f1, lat, lon)

    def _at(self, field, lats, lons):
        """双线性点插值（规则网格，map_coordinates 加速）: 输入标量或数组，返回同形。"""
        lat = self.d["lat"]; lon = self.d["lon"]
        scalar = np.ndim(lats) == 0
        li = (np.atleast_1d(np.asarray(lats, dtype=np.float64)) - lat[0]) / (lat[1] - lat[0])
        lj = (np.atleast_1d(np.asarray(lons, dtype=np.float64)) - lon[0]) / (lon[1] - lon[0])
        coords = np.stack([li, lj])
        r = map_coordinates(field, coords, order=1, mode="nearest")
        return float(r[0]) if scalar else r

    def _at2m(self, key, lats, lons):
        """批量两帧时间插值（多 AUV 点一次 map_coordinates），返回数组。"""
        h0 = int(self.d["h0"][self.t]); h1 = int(self.d["h1"][self.t]); fr = float(self.d["frac"][self.t])
        f0 = self.d[key][h0]; f1 = self.d[key][h1]
        if self.storage16:
            f0 = f0.astype(np.float32); f1 = f1.astype(np.float32)
        return (1 - fr) * self._at(f0, lats, lons) + fr * self._at(f1, lats, lons)

    def _obs(self):
        if self.use_frame_stack:
            obs = np.zeros((self.n_auv, self.obs_dim), dtype=np.float32)
            offs = (np.arange(self.N) - (self.N - 1) / 2) * self.spacing_deg
            for i in range(self.n_auv):
                h = list(self.flow_hist[i])
                if not h:
                    h = [np.zeros(3)]
                if len(h) < self.frame_k:
                    h = [h[0]] * (self.frame_k - len(h)) + h
                h = h[-self.frame_k:]
                parts = []
                if self.N > 1:   # patch + 历史
                    ys = self.lats[i] + offs; xs = self.lons[i] + offs
                    YY, XX = np.meshgrid(ys, xs)
                    parts += [self._at(Ispd, YY.ravel(), XX.ravel())/V_NORM,
                              self._at(Iu, YY.ravel(), XX.ravel())/V_NORM,
                              self._at(Iv, YY.ravel(), XX.ravel())/V_NORM]
                parts.append(np.concatenate(h))   # K帧单点流速
                if self.use_flowdiff:
                    if len(h) >= 2:
                        du = h[-1][1] - h[-2][1]; dv = h[-1][2] - h[-2][2]; ds = h[-1][0] - h[-2][0]
                        parts.append(np.array([du, dv, ds], dtype=np.float32))
                    else:
                        parts.append(np.zeros(3, dtype=np.float32))
                parts.append([self.vxs[i]/self.vmax, self.vys[i]/self.vmax])
                obs[i] = np.concatenate(parts).astype(np.float32)
            return obs
        Ispd, Iu, Iv = self._interp_fields()
        offs = (np.arange(self.N) - (self.N - 1) / 2) * self.spacing_deg
        obs = np.zeros((self.n_auv, self.obs_dim), dtype=np.float32)
        for i in range(self.n_auv):
            ys = self.lats[i] + offs; xs = self.lons[i] + offs
            YY, XX = np.meshgrid(ys, xs)
            pl = YY.ravel(); pn = XX.ravel()
            o = np.concatenate([
                self._at(Ispd, pl, pn) / V_NORM,
                self._at(Iu, pl, pn) / V_NORM,
                self._at(Iv, pl, pn) / V_NORM,
                [self.vxs[i] / self.vmax, self.vys[i] / self.vmax],
            ])
            if self.use_bearing:
                pspd = self._at(Ispd, pl, pn).reshape(self.N, self.N)
                iy, ix = np.unravel_index(int(np.argmax(pspd)), pspd.shape)
                dgx = (ix - (self.N - 1) / 2) * self.spacing_deg
                dgy = (iy - (self.N - 1) / 2) * self.spacing_deg
                nx = dgx * D2KM * np.cos(np.deg2rad(self.lats[i]))
                ny = dgy * D2KM
                nn = np.hypot(nx, ny)
                o = np.concatenate([o, np.array([nx / nn if nn > 1e-9 else 0.0, ny / nn if nn > 1e-9 else 0.0])])
            obs[i] = o.astype(np.float32)
        return obs

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32).reshape(self.n_auv, 2)
        actions = np.clip(actions, -1, 1)
        # The paper specifies a unit-disk action set. Component-wise clipping
        # alone would allow |a| up to sqrt(2), so project the joint command onto
        # the unit disk before scaling by the propulsion-speed limit.
        action_norm = np.linalg.norm(actions, axis=1, keepdims=True)
        actions = actions / np.maximum(1.0, action_norm)
        if self.use_frame_stack:
            Ispd = Iu = Iv = None
        else:
            Ispd, Iu, Iv = self._interp_fields()
        dt = float(STEP_MIN * 60)
        lo = self.d["lon"]; la = self.d["lat"]
        # ---- 批量采样：4 台 AUV 合并为一次 map_coordinates ----
        if self.use_frame_stack:
            cu = self._at2m("u", self.lats, self.lons)
            cv = self._at2m("v", self.lats, self.lons)
            cs_prev = self._at2m("spd", self.lats, self.lons)
        else:
            cu = self._at(Iu, self.lats, self.lons)
            cv = self._at(Iv, self.lats, self.lons)
            cs_prev = self._at(Ispd, self.lats, self.lons)
        spd_at = np.asarray(cs_prev, dtype=np.float32)
        self.vxs = actions[:, 0] * self.vmax
        self.vys = actions[:, 1] * self.vmax
        vx_abs = self.vxs + cu; vy_abs = self.vys + cv
        coslat = np.cos(np.deg2rad(self.lats))
        self.lons = np.clip(self.lons + vx_abs * dt / 1000.0 / (D2KM * coslat), lo.min(), lo.max())
        self.lats = np.clip(self.lats + vy_abs * dt / 1000.0 / D2KM, la.min(), la.max())
        if self.use_frame_stack:
            cs_new = self._at2m("spd", self.lats, self.lons)
            cu_new = self._at2m("u", self.lats, self.lons)
            cv_new = self._at2m("v", self.lats, self.lons)
        else:
            cs_new = self._at(Ispd, self.lats, self.lons)
            cu_new = self._at(Iu, self.lats, self.lons)
            cv_new = self._at(Iv, self.lats, self.lons)
        # Update the causal five-frame history in both observation modes. The
        # frame-stacked SB3 path previously omitted this update, leaving the
        # history fixed at the reset observation.
        for i in range(self.n_auv):
            self.flow_hist[i].append(
                np.array(
                    [cs_new[i], cu_new[i], cv_new[i]],
                    dtype=np.float32,
                )
            )
            if len(self.flow_hist[i]) > self.frame_k:
                self.flow_hist[i].pop(0)
        climb = np.maximum(0.0, cs_new - cs_prev) / V_NORM
        rewards = np.zeros(self.n_auv, dtype=np.float32)
        # ---- P3 信息增益（reward_mode="G"）团队级量预计算 ----
        if self.reward_mode == "G":
            ci = np.floor((self.lats - la.min()) / self.cov_deg).astype(np.int64)
            cj = np.floor((self.lons - lo.min()) / self.cov_deg).astype(np.int64)
            new_flag = np.zeros(self.n_auv, dtype=np.float32)
            for i in range(self.n_auv):
                key = (int(ci[i]), int(cj[i]))
                if key not in self._cov_cells:
                    self._cov_cells.add(key)
                    new_flag[i] = 1.0
            self._hist_lat.extend(self.lats.tolist())
            self._hist_lon.extend(self.lons.tolist())
            H_lat = np.asarray(self._hist_lat, dtype=np.float64)
            H_lon = np.asarray(self._hist_lon, dtype=np.float64)
            M = H_lat.size
            own = np.arange(M - self.n_auv, M)
            dyh = (self.lats[:, None] - H_lat[None, :]) * D2KM
            dxh = (self.lons[:, None] - H_lon[None, :]) * D2KM * coslat[:, None]
            dd = np.sqrt(dxh * dxh + dyh * dyh)
            dd[np.arange(self.n_auv), own] = np.inf
            d_nov = dd.min(axis=1)
            nov = np.clip(d_nov / self.nov_ref_km, 0.0, 1.0)
            W_at = np.atleast_1d(self._at(self.W, self.lats, self.lons)).astype(np.float32)
            team_mean_spd = float(cs_new.mean())
            cov_km2 = len(self._cov_cells) * self._cell_area_km2
        else:
            new_flag = nov = W_at = np.zeros(self.n_auv, dtype=np.float32)
            d_nov = np.full(self.n_auv, np.nan)
            cov_km2 = np.nan
        for i in range(self.n_auv):
            a = actions[i]
            cs = cs_prev[i]
            if self.reward_mode == "A":
                rewards[i] = min(cs / V_NORM, RMAX)
                continue
            if self.reward_mode == "G":
                # 覆盖面积 × 信息价值 + 新颖度 × 信息价值 + 采样流速方差 − 能耗
                r = (self.info_w_cov * float(W_at[i]) * float(new_flag[i])
                     + self.info_w_nov * float(W_at[i]) * float(nov[i])
                     + self.info_w_var * abs(float(cs_new[i]) - team_mean_spd) / V_NORM
                     - self.lambda_energy * float(a[0] ** 2 + a[1] ** 2))
            else:
                r = min(cs / V_NORM, RMAX) - self.lambda_energy * float(a[0] ** 2 + a[1] ** 2)
                if self.reward_mode in ("C", "D"):
                    spd_auv = np.hypot(self.vxs[i], self.vys[i])
                    if spd_auv > 1e-6 and cs > 1e-6:
                        align = (self.vxs[i] * cu[i] + self.vys[i] * cv[i]) / (spd_auv * cs)
                        r += BETA_ALIGN * max(0.0, align)
                if self.reward_mode == "D":
                    r += CLIMB_COEF * climb[i]
                if self.reward_mode == "F":
                    r = self.r_flow_w * climb[i] - self.lambda_eff * float(a[0] ** 2 + a[1] ** 2)
            land_dist = self._at(self.d["dist_to_land"].astype(np.float32) if self.storage16 else self.d["dist_to_land"], self.lats[i], self.lons[i])
            if land_dist <= 0.0:
                r -= LAND_PENALTY
            else:
                r -= W_BUFFER * max(0.0, 1.0 - land_dist / R_BUFFER)
            rewards[i] = r
        self.t = min(self.t + 1, self.d["T"] - 1)
        self.step_count += 1
        truncated = (self.t >= self.t_end) or (self.t >= self.d["T"] - 1) or (self.t >= int(self.d["landfall"]))
        terminated = False
        info = {"storm": self.d["storm"], "spd_at": spd_at, "t": self.t,
                "W_at": W_at, "new_cell": new_flag, "d_nov_km": d_nov, "cov_km2": cov_km2}
        return self._obs(), rewards, terminated, truncated, info
