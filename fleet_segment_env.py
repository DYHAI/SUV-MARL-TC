"""9-AUV 集群在 300-500km 段上追眼（海面）。

与 fleet_env.py 的区别：
- 每回合 = 一条 300-500km 轨迹段（data/segments_64.csv）。
- 每个 AUV 观测自己的 1x1 局部风场 cell = 该位置 spacing_deg x spacing_deg 尺度单元上的平均风(spd,u,v)，
  通过共享知道集群 9 个 cell 的风场。spacing_deg=0.25（~28km 单元）或 0.025（~2.8km 单元，细网格平均）。
- 编队半径约束 R_FORM=100 km（不是 200）。
- 初始在段首 200km 内随机布防，52 训练 / 12 测试。
观测(每AUV) = 9*(spd,u,v)/50 = 27 + 自身速度2 = 29 维。
"""
import os, sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from surface_wind_env import load_storm, land_ocean_mask, D2KM, STEP_MIN, LAND_PENALTY, R_BUFFER, W_BUFFER

N_AUV = 9
# Default remains 20 m/s for exact reproduction of existing experiments.
# Set FLEET_VMAX=12 for the Blue Whale-class 23 kn surface-speed scenario.
VMAX = float(os.environ.get("FLEET_VMAX", "20.0"))
FLOW_NORM = 50.0


def seg_idx(i10, step_min=None):
    """segments_64.csv 的 t0/t1 是 **10min 网格**索引；换算到当前 STEP_MIN 网格的索引。

    换决策步长（1/2/5/10 min）时必须用它，否则段的时间范围会错位。
    """
    return int(round(float(i10) * 10.0 / float(step_min or STEP_MIN)))


def smooth_scale(step_min=None):
    """平滑惩罚的**时间归一化**系数（相对 10min 基准）。

    坑：`rewards -= smooth * Σ(Δa)²` 是**按决策步**收的。决策步从 10min 缩到 1min，
    同一个 28h 任务的决策次数变 10 倍，等效平滑强度也跟着变——不归一化就无法比较不同步长。
    归一化后「单位物理时间内的平滑代价」一致：smooth_eff = smooth × (STEP_MIN/10)。
    """
    return float(step_min or STEP_MIN) / 10.0
SST_NORM = 30.0   # 原始 SST 归一化：(sst_K - 273.15)/SST_NORM
SST_ANOM_NORM = 3.0  # SST 距平归一化：sst_anom_degC / SST_ANOM_NORM
HYCOM_NORM = 3.0  # HYCOM 0m 原始流速归一化（~0-3 m/s）
HYCOM_ANOM_NORM = 1.0  # HYCOM 距平(台风诱发流)归一化（~0-1 m/s）
ZETA_NORM = 2e-4        # 相对涡度归一化（台风眼附近 ~1e-3 s^-1）
FRAMEDIFF_NORM = 3.0   # 风场逐小时帧差归一化（m/s per hour）
MSLP_REF = 100000.0    # Pa，海平面气压参考
MSLP_NORM = 5000.0     # Pa，气压值归一化
PGRAD_NORM = 0.02      # Pa/m，气压梯度归一化
PGRAD_CLIP = 5.0       # 气压梯度通道限幅（归一化后 ±5）；防止网格边缘假跳变把观测打爆
OBS_SANITIZE = os.environ.get("OBS_SANITIZE", "1") == "1"   # 观测净化开关（默认开）
# 观测里的「离岸距离」通道：OBS_COAST_DIST=1 时每个 cell 额外带一维「该 AUV 到岸线距离」
COAST_NORM = 100.0     # km，归一化尺度
COAST_CLIP_KM = 500.0  # km，超过截断（远海统一值）
R_FORM = 100.0      # km，编队半径约束（<=100km）
W_FORM = 1.0
START_RADIUS = 200.0


def deploy_formation(clon, clat, n_auv, formation="scatter", spread_radius=100.0, ring_radius=100.0):
    """在 (clon, clat) 周围布防 n_auv 个 AUV。
    formation="scatter": 在 spread_radius km 半径内随机散开（原两级布防第二步）；
    formation="ring": 在 ring_radius km 圆上均匀分布(4 AUV 即 90° 间隔)，整体随机旋转。
    返回 (lons, lats)。"""
    lons = np.zeros(n_auv); lats = np.zeros(n_auv)
    if formation == "ring":
        theta = np.random.uniform(0, 2 * np.pi)
        for i in range(n_auv):
            ang = theta + i * (2 * np.pi / n_auv)
            lons[i] = clon + ring_radius * np.cos(ang) / (D2KM * np.cos(np.deg2rad(clat)))
            lats[i] = clat + ring_radius * np.sin(ang) / D2KM
    else:
        for i in range(n_auv):
            ang = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0.0, 1.0) * spread_radius
            lons[i] = clon + r * np.cos(ang) / (D2KM * np.cos(np.deg2rad(clat)))
            lats[i] = clat + r * np.sin(ang) / D2KM
    return lons, lats


_LAND_INTERP = None


def land_interp():
    """到岸线距离的插值器（复用 land_ocean_mask 的缓存，模块级只建一次）。"""
    global _LAND_INTERP
    if _LAND_INTERP is None:
        _, dist_to_land, elon, elat = land_ocean_mask()
        _LAND_INTERP = RegularGridInterpolator((elat, elon), dist_to_land,
                                               bounds_error=False, fill_value=0.0)
    return _LAND_INTERP


def coast_gradient(lons, lats, interp):
    """到岸线距离在「km 平面」上的梯度（(n,2)，指向离岸更远 = 往海里）。"""
    e = 1e-3
    dx_km = e * D2KM * np.cos(np.deg2rad(lats))
    dy_km = e * D2KM
    de = (interp(np.stack([lats, lons + e], axis=1)) - interp(np.stack([lats, lons - e], axis=1))) / dx_km
    dn = (interp(np.stack([lats + e, lons], axis=1)) - interp(np.stack([lats - e, lons], axis=1))) / dy_km
    g = np.stack([de, dn], axis=1)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)


def safety_filter(a_use, lons, lats, interp, margin):
    """海岸安全滤波器（控制层硬约束）。

    距岸线不足 margin 时，把动作里「朝陆地」的分量投影掉（只能沿岸滑行或向海走）；
    已上陆的 AUV 额外叠加一个垂直岸线向海的推出分量。
    不依赖奖励，所以不会像"加大陆罚"那样逼策略全速逃离海岸：
    实测追踪只差 2.8% 的代价下，把上岸占比从 3.53% 压到 0.09%。
    """
    ld = interp(np.stack([lats, lons], axis=1))
    if (ld >= margin).all():
        return a_use
    n = coast_gradient(lons, lats, interp)          # 指向海里
    a = a_use.copy()
    dot = np.sum(a * n, axis=1, keepdims=True)
    a = a - np.minimum(dot, 0.0) * n                # 去掉向陆分量
    out = ld <= 0.0
    if out.any():
        a[out] = 0.5 * a[out] + 0.5 * n[out]        # 已在陆地：主动推出去
    return a


def sample_deployment(d, t0, n_auv, deploy_radius, formation="scatter", spread_radius=100.0,
                      ring_radius=100.0, deploy_mode="fixed", ocean_only=False,
                      min_coast_km=0.0, tries=300, fixed_km=None):
    """采样一次布防，返回 (lons, lats)。

    ocean_only=False 时**与旧实现逐位一致**（随机数抽取顺序完全一样），保证已有权重可复现。
    ocean_only=True 时做「最优 N 次拒绝采样」：
      反复重采，直到整队都满足「离岸 > min_coast_km」；300 次都不行就退化成
      **这 300 次里最小离岸距离最大的那一组** —— 所以永远不会比随机布防更差。
    为什么要这么做：实测随机布防有 6.5% 的 AUV 一开始就被放在陆地上，
    这部分贡献了全部「上岸时间」的 62%，而靠加大陆罚根本压不下去（还会让策略全速逃岸）。

    fixed_km>0 时进入「**固定距离布防、方位随机**」模式（布防半径迁移实验用）：
    布防中心固定离眼 fixed_km，方位角仍随机并做离岸拒绝采样。
    ⚠️ 实现上**照旧先抽一次 `cr`（随后丢弃）**，所以随机数消耗序列与均匀随机协议
    逐位相同 —— 同一 seed 下 fixed 与 uniform 的**方位角序列完全一致**，
    于是「同一台风同一段、只改布防距离」是严格配对的对比。
    """
    lo = d["lon"]; la = d["lat"]
    elo = d["track_lon"][t0]; ela = d["track_lat"][t0]
    best = None
    for _ in range(tries if ocean_only else 1):
        if deploy_mode == "safe":
            r34a = d.get("r34_km"); r50a = d.get("r50_km")
            r34 = float(r34a[t0]) if (r34a is not None and np.isfinite(r34a[t0])) else deploy_radius
            r50 = float(r50a[t0]) if (r50a is not None and np.isfinite(r50a[t0])) else 0.5 * r34
            r50 = min(r50, r34)
            Rlo = float(np.clip(r50, 10.0, 1000.0))
            Rhi = float(np.clip(r34, Rlo + 1.0, 1200.0))
            cr = np.random.uniform(Rlo, Rhi)
        elif deploy_mode in ("r34", "r50"):
            key = "r34_km" if deploy_mode == "r34" else "r50_km"
            Rarr = d.get(key)
            R = float(Rarr[t0]) if (Rarr is not None and np.isfinite(Rarr[t0])) else deploy_radius
            R = float(np.clip(R, 20.0, 1000.0))
            cr = np.random.uniform(0.0, 1.0) * R
        else:
            cr = np.random.uniform(0.0, 1.0) * deploy_radius
        # 固定距离模式：距离覆盖成定值，但**保留上面那一次 cr 抽样**以维持随机数流一致
        if fixed_km is not None and float(fixed_km) > 0.0:
            cr = float(fixed_km)
        cang = np.random.uniform(0, 2 * np.pi)
        clon = elo + cr * np.cos(cang) / (D2KM * np.cos(np.deg2rad(ela)))
        clat = ela + cr * np.sin(cang) / D2KM
        lons, lats = deploy_formation(clon, clat, n_auv, formation, spread_radius, ring_radius)
        lons = np.clip(lons, lo.min(), lo.max()); lats = np.clip(lats, la.min(), la.max())
        if not ocean_only:
            return lons, lats
        score = float(land_interp()(np.stack([lats, lons], axis=1)).min())
        if best is None or score > best[0]:
            best = (score, lons.copy(), lats.copy())
        if score > min_coast_km:
            return lons, lats
    return best[1], best[2]


class FleetSegmentEnv(gym.Env):
    def __init__(self, storms, segments_df, use_uv=True, vmax=VMAX, preloaded=None,
                 n_auv=N_AUV, smooth_penalty=0.0, spacing_deg=0.25, patch_n=1, deploy_radius=200.0,
                 spread_radius=100.0, formation="scatter", ring_radius=100.0, sst_mode="none", use_hycom0=False, hycom0_native=False, hycom0_anom=False, use_vorticity=False, use_framediff=False, use_mslp=False, deploy_mode="fixed",
                 comm_dropout=0.0, comm_outage_steps=3):
        super().__init__()
        self.n_auv = n_auv
        self.spacing_deg = spacing_deg
        self.patch_n = patch_n
        self.deploy_radius = deploy_radius
        # 布防半径迁移实验：DEPLOY_FIXED_KM>0 → 布防中心固定离眼该距离（方位仍随机）。
        # 与 deploy_radius 解耦：deploy_radius 仍然决定**数据裁剪范围**（=模型训练时的口径），
        # 这样用 800km 权重去 eval 100km 布防时，观测窗口不会被一起改掉。
        self.deploy_fixed_km = float(os.environ.get("DEPLOY_FIXED_KM", "0") or 0.0)
        # P5：布防方式。fixed=固定半径；r34/r50=按台风 7 级/10 级风圈半径布放
        self.deploy_mode = deploy_mode
        # P10 通信受限：邻居信息以概率 comm_dropout 丢失，中断持续 comm_outage_steps 步
        # （模拟卫星链路"过顶窗口"，比逐步独立丢包更真实）。0=完全通信，1=只看自己那格。
        self.comm_dropout = float(comm_dropout)
        # 避陆布防：DEPLOY_OCEAN_ONLY=1 时用「最优 N 次拒绝采样」布防（见 sample_deployment）。
        # DEPLOY_MIN_COAST_KM 可要求最小离岸距离（>0 即留安全裕度）。
        self.deploy_ocean_only = os.environ.get("DEPLOY_OCEAN_ONLY", "0") == "1"
        self.deploy_min_coast_km = float(os.environ.get("DEPLOY_MIN_COAST_KM", "0"))
        # 控制层安全滤波：SAFETY_MARGIN_KM>0 时，每步执行前把朝陆地的动作分量投影掉
        self.safety_margin_km = float(os.environ.get("SAFETY_MARGIN_KM", "0"))
        self.comm_outage_steps = int(comm_outage_steps)
        self._outage_left = np.zeros(n_auv, dtype=int)
        self.spread_radius = spread_radius   # AUV 在中心点周围 spread_radius km 内随机布防
        self.formation = formation
        self.ring_radius = ring_radius
        self.sst_mode = sst_mode
        self.use_hycom0 = use_hycom0
        self.hycom0_native = hycom0_native
        self.hycom0_anom = hycom0_anom
        self.use_vorticity = use_vorticity
        self.use_framediff = use_framediff
        self.use_mslp = use_mslp
        use_sst = (sst_mode != "none")
        if preloaded is not None:
            self.data = preloaded
        else:
            # 裁剪范围要覆盖「训练半径」和「实际布防距离」的较大者，否则远距离布防会被裁掉
            cm = (max(self.deploy_radius, self.deploy_fixed_km) + 200.0) \
                if hasattr(self, 'deploy_radius') else None
            self.data = {s: load_storm(s, use_uv=use_uv, crop_margin_km=cm, use_sst=use_sst, use_hycom0=use_hycom0, hycom0_native=hycom0_native, use_mslp=use_mslp) for s in storms}
        if use_vorticity or use_framediff:
            R = 6371000.0
            for _s, _d in self.data.items():
                _u = _d["u"]; _v = _d["v"]; _lat = _d["lat"]; _lon = _d["lon"]
                if use_vorticity:
                    dlon = np.median(np.diff(_lon)); dlat = np.median(np.diff(_lat))
                    dx = np.deg2rad(dlon) * R * np.cos(np.deg2rad(_lat))   # (lat,) 米/经度格
                    dy = np.deg2rad(dlat) * R                               # 米/纬度格(标量)
                    du_dy = np.gradient(_u, axis=1) / dy
                    dv_dx = np.gradient(_v, axis=2) / dx[None, :, None]
                    _d["zeta"] = (dv_dx - du_dy).astype(np.float32)
                if use_framediff:
                    dudt = np.zeros_like(_u); dvdt = np.zeros_like(_v)
                    dudt[1:] = _u[1:] - _u[:-1]
                    dvdt[1:] = _v[1:] - _v[:-1]
                    _d["dudt"] = dudt.astype(np.float32)
                    _d["dvdt"] = dvdt.astype(np.float32)
        # ⚠️ MSLP 的气压梯度**不再预计算**：预计算会为每个台风多存 2 个全场数组
        #    （实测每台风 157 MB → 94 MB，−29%）。500 台风阶段要 450 个训练台风，
        #    预计算版本需要 450×157 MB ≈ 71 GB，**超过整机 62 GB 跑不起来**，
        #    所以改成在 _obs() 里用插值器对 AUV 位置做有限差分即时求导（见 _obs）。
        # 内存优化：只用 SST 距平时，释放原始 sst（省一个全场数组）
        if sst_mode == "anom":
            for _d in self.data.values():
                _d.pop("sst", None)
        # 可选：把大场转 float16 省内存（STORAGE16=1），默认关闭以保持与既有实验逐位一致。
        # 只转 u/v/spd/sst_anom —— **msl 不转**：气压约 1.0e5 Pa，超出 float16 上限 65504 会变 inf。
        # 用途：deploy=800 + 风+SST距平+MSLP 这一档 fp32 需 ~69 GB > 本机 60 GB 可用，只能降精度跑。
        if os.environ.get("STORAGE16", "0") == "1":
            for _d in self.data.values():
                for _k in ("u", "v", "spd", "sst_anom"):
                    if _k in _d:
                        _d[_k] = _d[_k].astype(np.float16)
        self.storms = list(self.data.keys())
        self.vmax = vmax
        self.use_uv = use_uv
        # 时间归一化：smooth_penalty 以「10min 步」为基准，其它步长按 STEP_MIN/10 缩放，
        # 保证不同决策步长下「单位物理时间的平滑代价」一致（10min 时缩放=1，行为不变）。
        self.smooth_penalty = float(smooth_penalty) * smooth_scale()
        self.prev_actions = np.zeros((n_auv, 2), dtype=np.float32)
        of, dist_to_land, elon, elat = land_ocean_mask()
        self.dist_interp = RegularGridInterpolator((elat, elon), dist_to_land, bounds_error=False, fill_value=0.0)
        seg = segments_df[segments_df.storm.isin(self.storms)].reset_index(drop=True)
        self.segments = seg
        self.t0 = 0
        self.t1 = 1
        n_sst = 2 if sst_mode == "both" else (1 if sst_mode in ("sst", "anom") else 0)
        # OBS_COAST_DIST=1：每个 cell 加一维「到岸线距离」
        self.use_coast_obs = os.environ.get("OBS_COAST_DIST", "0") == "1"
        # OBS_AGENT_ID：给每台 AUV 加一段**独立身份编码**，打破"各台观测完全相同 → 共享策略输出相同动作
        # → 编队是刚体"的对称性（实测 reset 时 4 台 14 维观测逐位全同）。三种模式：
        #   "onehot"（旧写法 "1"）: n_auv 维 one-hot，每台绝对可分
        #   "angle"              : 2 维 [cos(2πi/n), sin(2πi/n)]，编码环形编队里的角度位置，长度与 n 无关
        #   "both"               : n_auv + 2 维（绝对可分 + 平滑几何先验）
        #   "0"（默认）          : 不加，维度与旧模型完全一致（旧权重可复现）
        self.agent_id_mode = os.environ.get("OBS_AGENT_ID", "0")
        if self.agent_id_mode == "1":
            self.agent_id_mode = "onehot"
        _onehot = self.agent_id_mode in ("onehot", "both")
        _angle = self.agent_id_mode in ("angle", "both")
        self._n_id = (n_auv if _onehot else 0) + (2 if _angle else 0)
        if self._n_id:
            blk = np.zeros((n_auv, self._n_id), dtype=np.float32)
            c = 0
            if _onehot:
                blk[np.arange(n_auv), np.arange(n_auv)] = 1.0
                c += n_auv
            if _angle:
                th = 2 * np.pi * np.arange(n_auv) / n_auv
                blk[:, c] = np.cos(th); blk[:, c + 1] = np.sin(th)
            self._id_block = blk
        self.nch = (3 + n_sst + (3 if use_hycom0 else 0) + (1 if use_vorticity else 0)
                    + (2 if use_framediff else 0) + (3 if use_mslp else 0)
                    + (1 if self.use_coast_obs else 0))
        self.obs_dim = (n_auv * patch_n * patch_n * self.nch + 2
                        + self._n_id)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_auv, self.obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_auv, 2), dtype=np.float32)
        self.step_count = 0

    def _obs(self):
        d = self.d; t = self.t
        h0 = int(d["h0"][t]); h1 = int(d["h1"][t]); fr = float(d["frac"][t])
        spd = (1 - fr) * d["spd"][h0] + fr * d["spd"][h1]
        uu = (1 - fr) * d["u"][h0] + fr * d["u"][h1]
        vv = (1 - fr) * d["v"][h0] + fr * d["v"][h1]
        lon_g = d["lon"]; lat_g = d["lat"]
        Is = RegularGridInterpolator((lat_g, lon_g), spd, bounds_error=False, fill_value=0.0)
        Iu = RegularGridInterpolator((lat_g, lon_g), uu, bounds_error=False, fill_value=0.0)
        Iv = RegularGridInterpolator((lat_g, lon_g), vv, bounds_error=False, fill_value=0.0)
        # 涡度 + 帧差通道
        extra_inters = []
        if self.use_vorticity:
            zeta_c = (1 - fr) * d["zeta"][h0] + fr * d["zeta"][h1]
            extra_inters.append(("zeta", RegularGridInterpolator((lat_g, lon_g), zeta_c, bounds_error=False, fill_value=0.0), ZETA_NORM))
        if self.use_framediff:
            dudt_c = (1 - fr) * d["dudt"][h0] + fr * d["dudt"][h1]
            dvdt_c = (1 - fr) * d["dvdt"][h0] + fr * d["dvdt"][h1]
            extra_inters.append(("dudt", RegularGridInterpolator((lat_g, lon_g), dudt_c, bounds_error=False, fill_value=0.0), FRAMEDIFF_NORM))
            extra_inters.append(("dvdt", RegularGridInterpolator((lat_g, lon_g), dvdt_c, bounds_error=False, fill_value=0.0), FRAMEDIFF_NORM))
        # MSLP 气压 + 气压梯度通道（指向低压中心=眼）
        if self.use_mslp:
            msl_c = (1 - fr) * d["msl"][h0] + fr * d["msl"][h1]
            msl_inters = [
                RegularGridInterpolator((lat_g, lon_g), msl_c, bounds_error=False, fill_value=MSLP_REF),
            ]
        # SST 通道（raw / anom / both）
        sst_inters = []
        if self.sst_mode in ("sst", "both"):
            raw_c = (1 - fr) * d["sst"][h0] + fr * d["sst"][h1]
            sst_inters.append(("raw", RegularGridInterpolator((lat_g, lon_g), raw_c, bounds_error=False, fill_value=273.15)))
        if self.sst_mode in ("anom", "both"):
            anom_c = (1 - fr) * d["sst_anom"][h0] + fr * d["sst_anom"][h1]
            sst_inters.append(("anom", RegularGridInterpolator((lat_g, lon_g), anom_c, bounds_error=False, fill_value=0.0)))
        # HYCOM 0m 通道（u / v / 流速大小 sqrt(u²+v²)）
        hy_inters = []
        if self.use_hycom0:
            su = "hycom0_u_anom" if self.hycom0_anom else "hycom0_u"
            sv = "hycom0_v_anom" if self.hycom0_anom else "hycom0_v"
            ss = "hycom0_spd_anom" if self.hycom0_anom else "hycom0_spd"
            hyu_c = (1 - fr) * d[su][h0] + fr * d[su][h1]
            hyv_c = (1 - fr) * d[sv][h0] + fr * d[sv][h1]
            hys_c = (1 - fr) * d[ss][h0] + fr * d[ss][h1]
            hlat_g = d["hycom0_lat"] if self.hycom0_native else lat_g
            hlon_g = d["hycom0_lon"] if self.hycom0_native else lon_g
            hy_inters = [
                RegularGridInterpolator((hlat_g, hlon_g), hyu_c, bounds_error=False, fill_value=0.0),
                RegularGridInterpolator((hlat_g, hlon_g), hyv_c, bounds_error=False, fill_value=0.0),
                RegularGridInterpolator((hlat_g, hlon_g), hys_c, bounds_error=False, fill_value=0.0),
            ]
        hy_norm = HYCOM_ANOM_NORM if self.hycom0_anom else HYCOM_NORM
        pn = self.patch_n; nch = self.nch
        if pn == 1:
            # 1x1 patch = spacing_deg x spacing_deg 单元上的平均(5x5 子采样)
            half = self.spacing_deg / 2.0
            sub = np.linspace(-half, half, 5)
            cells = np.zeros((self.n_auv, nch), dtype=np.float32)
            for i in range(self.n_auv):
                xs = self.lons[i] + sub
                ys = self.lats[i] + sub
                X, Y = np.meshgrid(xs, ys)
                pts = np.stack([Y.ravel(), X.ravel()], axis=1)
                cells[i, 0] = float(Is(pts).mean()) / FLOW_NORM
                cells[i, 1] = float(Iu(pts).mean()) / FLOW_NORM
                cells[i, 2] = float(Iv(pts).mean()) / FLOW_NORM
                ci = 3
                for kind, interp in sst_inters:
                    v = float(interp(pts).mean())
                    cells[i, ci] = (v - 273.15) / SST_NORM if kind == "raw" else v / SST_ANOM_NORM
                    ci += 1
                if self.use_hycom0:
                    cells[i, ci] = float(hy_inters[0](pts).mean()) / hy_norm
                    cells[i, ci + 1] = float(hy_inters[1](pts).mean()) / hy_norm
                    cells[i, ci + 2] = float(hy_inters[2](pts).mean()) / hy_norm
                    ci += 3
                for _name, _interp, _norm in extra_inters:
                    cells[i, ci] = float(_interp(pts).mean()) / _norm
                    ci += 1
                if self.use_mslp:
                    cells[i, ci] = (float(msl_inters[0](pts).mean()) - MSLP_REF) / MSLP_NORM
                    # 气压梯度：对同一个 MSLP 插值器在 AUV 位置做中心差分（单位 Pa/m，指向低压=眼）
                    _I = msl_inters[0]
                    _la = float(self.lats[i]); _lo = float(self.lons[i]); _e = 1e-3
                    _dxm = np.deg2rad(_e) * 6371000.0 * np.cos(np.deg2rad(_la))
                    _dym = np.deg2rad(_e) * 6371000.0
                    _gx = -(_I(np.array([[_la, _lo + _e]]))[0] - _I(np.array([[_la, _lo - _e]]))[0]) / (2 * _dxm)
                    _gy = -(_I(np.array([[_la + _e, _lo]]))[0] - _I(np.array([[_la - _e, _lo]]))[0]) / (2 * _dym)
                    # ⚠️ 必须限幅：差分步长 1e-3°(≈111m) << 气压网格 0.25°(≈28km)，
                    #    靠近网格边缘时 ±e 一个点在内、一个点落在界外走 fill_value=MSLP_REF，
                    #    会造成 ~1000 Pa 的假跳变 → 归一化后观测达 3.8e2（应 O(1)）→ 策略 NaN 崩训练。
                    #    （2026-10-12：deploy=400 的 windmslp/best 六组就是这样崩掉的。）
                    cells[i, ci + 1] = float(np.clip(_gx / PGRAD_NORM, -PGRAD_CLIP, PGRAD_CLIP))
                    cells[i, ci + 2] = float(np.clip(_gy / PGRAD_NORM, -PGRAD_CLIP, PGRAD_CLIP))
                    ci += 3
                if self.use_coast_obs:
                    # 该 AUV 到最近岸线的距离（复用安全滤波那个带缓存的距离场插值器）
                    _ld = float(land_interp()(np.array([[self.lats[i], self.lons[i]]]))[0])
                    cells[i, ci] = min(max(_ld, 0.0), COAST_CLIP_KM) / COAST_NORM
                    ci += 1
            obs = np.zeros((self.n_auv, self.obs_dim), dtype=np.float32)
            # P10 通信受限：按丢包状态屏蔽「邻居」cell（自己那一格永远可见）
            vis = np.ones(self.n_auv, dtype=bool)
            if self.comm_dropout > 0:
                for j in range(self.n_auv):
                    if self._outage_left[j] > 0:
                        self._outage_left[j] -= 1
                        vis[j] = False
                    elif np.random.rand() < self.comm_dropout:
                        self._outage_left[j] = self.comm_outage_steps
                        vis[j] = False
            self._comm_visible = int(vis.sum())
            cells_v = cells.copy()
            cells_v[~vis] = 0.0
            for i in range(self.n_auv):
                obs[i, :self.n_auv*nch] = cells_v.ravel()
                obs[i, i * nch:(i + 1) * nch] = cells[i]   # 自己的观测永远可见
                obs[i, self.n_auv*nch:self.n_auv*nch+2] = [self.vxs[i]/self.vmax, self.vys[i]/self.vmax]
                if self._n_id:
                    obs[i, self.n_auv * nch + 2:] = self._id_block[i]   # 身份/角度编码
            return self._sanitize_obs(obs)
        offs = (np.arange(pn) - (pn - 1) / 2) * self.spacing_deg
        cells = np.zeros((self.n_auv, pn * pn * nch), dtype=np.float32)
        for i in range(self.n_auv):
            xs = self.lons[i] + offs
            ys = self.lats[i] + offs
            X, Y = np.meshgrid(xs, ys)
            pts = np.stack([Y.ravel(), X.ravel()], axis=1)
            w = np.stack([Is(pts), Iu(pts), Iv(pts)], axis=1) / FLOW_NORM  # (pn*pn, 3)
            for kind, interp in sst_inters:
                v = interp(pts)
                col = (v - 273.15) / SST_NORM if kind == "raw" else v / SST_ANOM_NORM
                w = np.concatenate([w, col[:, None]], axis=1)
            if self.use_hycom0:
                for interp in hy_inters:
                    col = interp(pts) / hy_norm
                    w = np.concatenate([w, col[:, None]], axis=1)
            for _name, _interp, _norm in extra_inters:
                col = _interp(pts) / _norm
                w = np.concatenate([w, col[:, None]], axis=1)
            if self.use_mslp:
                w = np.concatenate([w, ((msl_inters[0](pts) - MSLP_REF) / MSLP_NORM)[:, None]], axis=1)
                _la = float(self.lats[i]); _lo = float(self.lons[i]); _e = 1e-3
                _dxm = np.deg2rad(_e) * 6371000.0 * np.cos(np.deg2rad(_la))
                _dym = np.deg2rad(_e) * 6371000.0
                _gx = -(msl_inters[0](np.array([[_la, _lo + _e]]))[0] - msl_inters[0](np.array([[_la, _lo - _e]]))[0]) / (2 * _dxm)
                _gy = -(msl_inters[0](np.array([[_la + _e, _lo]]))[0] - msl_inters[0](np.array([[_la - _e, _lo]]))[0]) / (2 * _dym)
                w = np.concatenate([w, np.full((w.shape[0], 1),
                                   np.clip(_gx / PGRAD_NORM, -PGRAD_CLIP, PGRAD_CLIP), dtype=w.dtype)], axis=1)
                w = np.concatenate([w, np.full((w.shape[0], 1),
                                   np.clip(_gy / PGRAD_NORM, -PGRAD_CLIP, PGRAD_CLIP), dtype=w.dtype)], axis=1)
            if self.use_coast_obs:
                _ld = float(land_interp()(np.array([[self.lats[i], self.lons[i]]]))[0])
                _v = min(max(_ld, 0.0), COAST_CLIP_KM) / COAST_NORM
                w = np.concatenate([w, np.full((w.shape[0], 1), _v, dtype=w.dtype)], axis=1)
            cells[i] = w.ravel()
        obs = np.zeros((self.n_auv, self.obs_dim), dtype=np.float32)
        for i in range(self.n_auv):
            obs[i, :self.n_auv*pn*pn*nch] = cells.ravel()
            obs[i, self.n_auv*pn*pn*nch:self.n_auv*pn*pn*nch+2] = [self.vxs[i]/self.vmax, self.vys[i]/self.vmax]
        return self._sanitize_obs(obs)

    def _sanitize_obs(self, obs):
        """兜底净化：把观测里的 NaN/Inf 抹掉并计数。

        2026-10-12 背景：MSLP 档在 deploy>=400 训练到 ~230k 步时崩成 NaN，
        traceback 里**只有第 4 行（某一台 AUV）是 nan**，说明是**那一台的输入观测**非有限，
        而不是权重发散。`np.clip(nan, -5, 5)` 仍然是 nan，所以梯度限幅防不住这种情况，
        必须在 _obs 出口统一净化，否则一帧 NaN 就会把整个策略打成 NaN。
        """
        if not OBS_SANITIZE:
            return obs
        bad = ~np.isfinite(obs)
        if bad.any():
            self._obs_nan_hits = int(getattr(self, "_obs_nan_hits", 0)) + int(bad.any(axis=1).sum())
            np.nan_to_num(obs, copy=False, nan=0.0, posinf=PGRAD_CLIP, neginf=-PGRAD_CLIP)
        return obs

    def _reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        idx = int(np.random.randint(len(self.segments)))
        seg = self.segments.iloc[idx]
        s = seg["storm"]
        self.d = self.data[s]
        self.t0 = seg_idx(seg["t0"]); self.t1 = seg_idx(seg["t1"])
        # 粗决策步（如 60min）下短段可能被 round 成 0 步，保底 1 步
        if self.t1 <= self.t0:
            self.t1 = self.t0 + 1
        self.t = self.t0
        self.vxs = np.zeros(self.n_auv); self.vys = np.zeros(self.n_auv)
        # 布防：ocean_only=False 时与旧实现逐位一致；=True 时避陆（最优 N 次拒绝采样）
        self.lons, self.lats = sample_deployment(
            self.d, self.t0, self.n_auv, self.deploy_radius, self.formation,
            self.spread_radius, self.ring_radius, self.deploy_mode,
            ocean_only=getattr(self, "deploy_ocean_only", False),
            min_coast_km=getattr(self, "deploy_min_coast_km", 0.0),
            fixed_km=getattr(self, "deploy_fixed_km", 0.0))
        self.prev_actions = np.zeros((self.n_auv, 2), dtype=np.float32)
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, -1, 1)
        a = np.nan_to_num(a, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)  # 动作里的 NaN 也兜底
        # 圆形裁剪：合速度 ≤ VMAX（每轴限幅会让对角线 |a| 达 √2，速度超 VMAX）
        norm = np.sqrt(a[:, 0]**2 + a[:, 1]**2)
        scale = np.where(norm > 1.0, 1.0 / np.maximum(norm, 1e-9), 1.0)   # 避免 0 除告警
        a = a * scale[:, None]
        if getattr(self, "safety_margin_km", 0.0) > 0:
            a = safety_filter(a, self.lons, self.lats, self.dist_interp, self.safety_margin_km)
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
        # 陆罚单独拆出来（数值与原来完全一致），供 info 透出 / 上层决定是否在裁剪之后叠加
        land_pen = np.where(on_land, -LAND_PENALTY,
                            -W_BUFFER * np.maximum(0.0, 1.0 - land_dist / R_BUFFER)).astype(np.float32)
        rewards = rewards + land_pen
        terminated = False
        truncated = (self.t >= self.t1) or (self.t >= self.d["T"] - 1) or \
                    (not bool(self.d["in_domain"][self.t])) or (self.t >= self.d["landfall"])
        info = {"dist_km": dist, "eye_lon": eye_lon, "eye_lat": eye_lat,
                "storm": self.d["storm"], "rewards": rewards, "t0": self.t0, "t1": self.t1,
                "land_pen": land_pen, "on_land": on_land, "land_dist": land_dist}
        return self._obs(), float(rewards.sum()), terminated, truncated, info

    def reset(self, seed=None, options=None):
        return self._reset(seed=seed, options=options)
