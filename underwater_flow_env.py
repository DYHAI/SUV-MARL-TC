"""水下 AUV 追踪台风垂直投影环境（50m，仅观测水平流速，10 分钟动作分辨率）。

数据源：HYCOM 50m 水平流速 water_u/water_v（3 小时一帧，0.08°）。
时间：3 小时帧之间线性插值到 10 分钟（STEP_MIN=10，每小时 6 步、每 3 小时 18 步）。
目标：台风最佳路径在水下的垂直投影（同一 lat/lon 轨迹）。
观测：AUV 周围 3x3（0.25° 间距）的流速大小 + u + v；动作 2 维 [-1,1] -> vx/vy。
奖励：越近眼越好 + 两级陆地软惩罚（上陆 -20，100km 缓冲带）。
"""
import os
from collections import deque
import numpy as np
import pandas as pd
import xarray as xr
import gymnasium as gym
from gymnasium import spaces
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

BASE = os.path.dirname(os.path.abspath(__file__))
UNION = os.path.join(BASE, "data", "union_region")
TRACK = os.path.join(BASE, "data", "selected_typhoons")

D2KM = 111.0
STEP_MIN = 10           # 动作分辨率：10 分钟
VMAX = 20.0             # m/s，与水上任务一致
FLOW_NORM = 2.0         # 流速归一化（~0-2 m/s）
CROP_MARGIN = 5.0       # 度，按轨迹裁剪每个台风的网格
LAND_PENALTY = 20.0
R_BUFFER = 100.0        # km 海岸软缓冲半径
W_BUFFER = 8.0
R_CLOSE = 100.0         # km 贴近眼奖励阈值

# ---- 进程内载入缓存（2026-10-10） ----
# 目的：矩阵扫描（几百个配置）时，每组都重读 450 个台风要 ~19min，纯浪费。
# 缓存后：主进程 preload() 一次，fork 出的训练/评测子进程通过 COW 共享只读数组，
# 后续配置换 hyper-param 不再读盘。
# 注意：缓存返回的是**同一个 dict**，调用方不得原地修改其数组（storage16 那种
# 就地 astype 必须先 dict 浅拷贝，见 UnderwaterFleetFlowEnv.__init__）。
_LOAD_CACHE = {}

# ---- 磁盘缓存（2026-10-10）----
# build_uwflow_cache.py 把预处理好的 0.25° 裁剪网格存成**不压缩** npz，
# load_storm 优先 mmap 读取：多个训练/评测进程共享同一份 page cache，
# 载入从 ~50min/450台风 降到 ~1min，且不重复占内存。
# 只对 (native_res=False, grid_deg=0.25, crop_margin=3.0) 生效，其余走 netCDF。
_DISK_CACHE_ROOT = os.path.expanduser(
    os.environ.get("UWFLOW_CACHE_DIR", "~/.cache/auv_tc/uwflow")
)


def _disk_cache_path(storm, depth_m, grid_deg, crop_margin, native_res):
    if native_res or grid_deg is None:
        return None
    if abs(float(grid_deg) - 0.25) > 1e-9 or abs(float(crop_margin) - 3.0) > 1e-9:
        return None
    p = os.path.join(_DISK_CACHE_ROOT, "d%g_g0.25_c3" % float(depth_m), storm + ".npz")
    return p if os.path.exists(p) else None


def _load_from_disk_cache(storm, p):
    z = np.load(p, mmap_mode="r")
    return dict(storm=storm, u=z["u"], v=z["v"], spd=z["spd"],
                dist_to_land=z["dist_to_land"], lon=z["lon"], lat=z["lat"],
                times=z["times"], track_lon=z["track_lon"], track_lat=z["track_lat"],
                track_x=z["track_x"], track_y=z["track_y"], track_cum=z["track_cum"],
                in_domain=z["in_domain"].astype(bool), T=int(z["T"]),
                h0=z["h0"], h1=z["h1"], frac=z["frac"],
                landfall=int(z["landfall"]))


def load_storm(storm, use_uv=True, use_anomaly=False, depth_m=50, crop_margin=CROP_MARGIN, grid_deg=None, native_res=False):
    key = (storm, bool(use_uv), bool(use_anomaly), float(depth_m),
           float(crop_margin), (None if grid_deg is None else float(grid_deg)), bool(native_res))
    if key in _LOAD_CACHE:
        return _LOAD_CACHE[key]
    if not use_anomaly:
        dp = _disk_cache_path(storm, depth_m, grid_deg, crop_margin, native_res)
        if dp is not None:
            d = _load_from_disk_cache(storm, dp)
            _LOAD_CACHE[key] = d
            return d
    ds = xr.open_dataset(os.path.join(UNION, storm, f"hycom_uv_{int(depth_m)}m.nc"))
    u = ds["water_u"].values.astype("float32")   # (time, depth=1, lat, lon) 3 小时一帧
    v = ds["water_v"].values.astype("float32")
    lon = ds["lon"].values.astype("float64")
    lat = ds["lat"].values.astype("float64")
    times = ds["time"].values
    ds.close()
    if native_res:
        u = u[:, 0, :, :]; v = v[:, 0, :, :]      # 原生：lat 0.04° 不降采样
    else:
        u = u[:, 0, ::2, :]; v = v[:, 0, ::2, :]  # lat 降采样 2x 到 0.08°
        lat = lat[::2]
    land_full = np.all(np.isnan(u) | np.isnan(v), axis=0).astype(np.float32)   # (lat, lon) 陆地=各时刻均 NaN
    u = np.nan_to_num(u, copy=False); v = np.nan_to_num(v, copy=False)
    spd = np.hypot(u, v)
    if use_anomaly:
        u = u - np.median(u, axis=0)
        v = v - np.median(v, axis=0)
        spd = np.hypot(u, v)

    tr = pd.read_csv(os.path.join(TRACK, storm, "track.csv"), parse_dates=["ISO_TIME"]).dropna(subset=["LAT", "LON"]).sort_values("ISO_TIME")
    tt = tr["ISO_TIME"].values.astype("datetime64[ns]").astype("int64")

    # 裁剪到轨迹 bbox + margin（用 3 小时轨迹的 min/max）
    tq3 = times.astype("datetime64[ns]").astype("int64")
    lon_t3 = np.interp(tq3, tt, tr["LON"].values)
    lat_t3 = np.interp(tq3, tt, tr["LAT"].values)
    lo0 = float(lon_t3.min() - crop_margin); lo1 = float(lon_t3.max() + crop_margin)
    la0 = float(lat_t3.min() - crop_margin); la1 = float(lat_t3.max() + crop_margin)
    jm = (lon >= lo0) & (lon <= lo1)
    im = (lat >= la0) & (lat <= la1)
    lon_c = lon[jm]; lat_c = lat[im]
    u = u[:, im, :][:, :, jm]; v = v[:, im, :][:, :, jm]; spd = spd[:, im, :][:, :, jm]
    land = land_full[im, :][:, jm]

    # 可选：空间插值到更粗均匀网格（如 0.25°，省内存；native_res 不做空间插值）
    if grid_deg is not None and not native_res:
        dlat0 = float(np.diff(lat_c).mean())
        if abs(grid_deg - dlat0) > 1e-3:
            lat_g = np.arange(np.ceil(lat_c.min()/grid_deg)*grid_deg, lat_c.max()+1e-9, grid_deg)
            lon_g = np.arange(np.ceil(lon_c.min()/grid_deg)*grid_deg, lon_c.max()+1e-9, grid_deg)
            ua = xr.DataArray(u, dims=("time","lat","lon"), coords={"lat": lat_c, "lon": lon_c})
            va = xr.DataArray(v, dims=("time","lat","lon"), coords={"lat": lat_c, "lon": lon_c})
            sa = xr.DataArray(spd, dims=("time","lat","lon"), coords={"lat": lat_c, "lon": lon_c})
            la = xr.DataArray(land, dims=("lat","lon"), coords={"lat": lat_c, "lon": lon_c})
            u = ua.interp(lat=lat_g, lon=lon_g, method="linear").values.astype("float32")
            v = va.interp(lat=lat_g, lon=lon_g, method="linear").values.astype("float32")
            spd = sa.interp(lat=lat_g, lon=lon_g, method="linear").values.astype("float32")
            land = (la.interp(lat=lat_g, lon=lon_g, method="nearest").values > 0.5).astype(np.float32)
            lat_c = lat_g; lon_c = lon_g

    # 到最近陆地的距离(km)
    ocean = (land < 0.5).astype(np.float32)
    lat_sp = D2KM * float(np.diff(lat_c).mean())
    lon_sp = D2KM * np.cos(np.radians(float(lat_c.mean()))) * float(np.diff(lon_c).mean())
    dist_to_land = distance_transform_edt(ocean, sampling=(lat_sp, lon_sp)).astype(np.float32)

    # 10 分钟时间网格 + 台风眼轨迹插值 + 3h->10min 插值索引
    times10 = pd.date_range(times[0], times[-1], freq=f"{STEP_MIN}min").values
    tq10 = times10.astype("datetime64[ns]").astype("int64")
    lon_t = np.interp(tq10, tt, tr["LON"].values)
    lat_t = np.interp(tq10, tt, tr["LAT"].values)
    n = len(times10)
    x_t = (lon_t - lon_t[0]) * D2KM * np.cos(np.deg2rad(lat_t[0]))
    y_t = (lat_t - lat_t[0]) * D2KM
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x_t), np.diff(y_t)))])
    sub = max(1, 180 // STEP_MIN)          # 每 3 小时 18 个 10min 步
    h0 = np.minimum(np.arange(n) // sub, len(times) - 1)
    h1 = np.minimum(h0 + 1, len(times) - 1)
    frac = (np.arange(n) % sub) / float(sub)

    # 最终登陆时刻索引（穿过台湾/菲律宾又回海的不算最终登陆）
    interp_land = RegularGridInterpolator((lat_c, lon_c), land, bounds_error=False, fill_value=1.0)
    oc = interp_land(np.stack([lat_t, lon_t], axis=1))
    on_land = oc > 0.5
    landfall = len(on_land)
    if on_land[-1]:
        i = len(on_land) - 1
        while i > 0 and on_land[i - 1]:
            i -= 1
        landfall = i

    in_domain = np.ones(n, dtype=bool)     # 裁剪后轨迹始终在网格内
    d = dict(storm=storm, spd=spd, u=u, v=v, lon=lon_c, lat=lat_c, times=times,
             track_lon=lon_t, track_lat=lat_t, track_x=x_t, track_y=y_t, track_cum=cum,
             in_domain=in_domain, T=n,
             h0=h0, h1=h1, frac=frac, dist_to_land=dist_to_land, landfall=landfall)
    _LOAD_CACHE[key] = d
    return d


def preload(storms, **kw):
    """把一批台风读入进程内缓存。fork 出的子进程可直接 COW 复用，不再读盘。"""
    n = 0
    for s in storms:
        load_storm(s, **kw)
        n += 1
    return n


def cache_clear():
    _LOAD_CACHE.clear()


class UnderwaterFlowEnv(gym.Env):
    def __init__(self, storms, use_uv=True, spacing_deg=0.25, N=3, vmax=VMAX,
                 start_radius=200.0, preloaded=None, use_anomaly=False, depth_m=50,
                 bearing_noise_deg=None, trail=False):
        super().__init__()
        if preloaded is not None:
            self.data = preloaded
        else:
            self.data = {s: load_storm(s, use_uv=use_uv, use_anomaly=use_anomaly, depth_m=depth_m) for s in storms}
        self.storms = list(self.data.keys())
        self.spacing_deg = spacing_deg; self.N = N; self.vmax = vmax
        self.use_uv = use_uv; self.start_radius = start_radius; self.use_anomaly = use_anomaly; self.depth_m = depth_m
        self.bearing_noise_deg = bearing_noise_deg
        self.trail = trail
        self.flow_norm = 1.0 if use_anomaly else FLOW_NORM
        self.nch = 3 if use_uv else 1
        self.obs_dim = self.nch * N * N + 2 + (2 if bearing_noise_deg is not None else 0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.step_count = 0

    def _reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s = np.random.choice(self.storms)
        self.d = self.data[s]
        t0 = 0
        self.t = t0
        ang = np.random.uniform(0, 2 * np.pi); r = np.random.uniform(0, self.start_radius)
        self.lon = self.d["track_lon"][t0] + r * np.cos(ang) / (D2KM * np.cos(np.deg2rad(self.d["track_lat"][t0])))
        self.lat = self.d["track_lat"][t0] + r * np.sin(ang) / D2KM
        lo = self.d["lon"]; la = self.d["lat"]
        self.lon = float(np.clip(self.lon, lo.min(), lo.max()))
        self.lat = float(np.clip(self.lat, la.min(), la.max()))
        self.vx = 0.0; self.vy = 0.0; self.step_count = 0
        self._tgt_lon = self.d["track_lon"][self.t]
        self._tgt_lat = self.d["track_lat"][self.t]
        return self._obs(), {}

    def _obs(self):
        t = self.t
        h0 = int(self.d["h0"][t]); h1 = int(self.d["h1"][t]); fr = float(self.d["frac"][t])
        spd = (1 - fr) * self.d["spd"][h0] + fr * self.d["spd"][h1]
        lon = self.d["lon"]; lat = self.d["lat"]
        offs = (np.arange(self.N) - (self.N - 1) / 2) * self.spacing_deg
        xs = self.lon + offs; ys = self.lat + offs
        X, Y = np.meshgrid(xs, ys)
        pts = np.stack([Y.ravel(), X.ravel()], axis=1)
        Ip = RegularGridInterpolator((lat, lon), spd, bounds_error=False, fill_value=0.0)
        patch = Ip(pts).reshape(self.N, self.N)
        obs = [patch.ravel() / self.flow_norm]
        if self.use_uv:
            uu = (1 - fr) * self.d["u"][h0] + fr * self.d["u"][h1]
            vv = (1 - fr) * self.d["v"][h0] + fr * self.d["v"][h1]
            Iu = RegularGridInterpolator((lat, lon), uu, bounds_error=False, fill_value=0.0)
            Iv = RegularGridInterpolator((lat, lon), vv, bounds_error=False, fill_value=0.0)
            obs.append(Iu(pts).reshape(self.N, self.N).ravel() / self.flow_norm)
            obs.append(Iv(pts).reshape(self.N, self.N).ravel() / self.flow_norm)
        obs.append(np.array([self.vx / self.vmax, self.vy / self.vmax], dtype=np.float32))
        if self.bearing_noise_deg is not None:
            clon = self._tgt_lon; clat = self._tgt_lat
            dx = (clon - self.lon) * D2KM * np.cos(np.deg2rad(clat))
            dy = (clat - self.lat) * D2KM
            theta = np.arctan2(dx, dy)
            if self.bearing_noise_deg > 0:
                theta = theta + np.random.normal(0.0, np.deg2rad(self.bearing_noise_deg))
            obs.append(np.array([np.cos(theta), np.sin(theta)], dtype=np.float32))
        return np.concatenate(obs).astype(np.float32)

    def step(self, action):
        a = np.clip(action, -1, 1)
        self.vx = a[0] * self.vmax; self.vy = a[1] * self.vmax
        dt = float(STEP_MIN * 60)
        dx_km = self.vx * dt / 1000.0
        dy_km = self.vy * dt / 1000.0
        self.lon += dx_km / (D2KM * np.cos(np.deg2rad(self.lat)))
        self.lat += dy_km / D2KM
        lo = self.d["lon"]; la = self.d["lat"]
        self.lon = float(np.clip(self.lon, lo.min(), lo.max()))
        self.lat = float(np.clip(self.lat, la.min(), la.max()))
        self.t = min(self.t + 1, self.d["T"] - 1)
        self.step_count += 1
        if self.trail:
            ax = (self.lon - self.d["track_lon"][0]) * D2KM * np.cos(np.deg2rad(self.d["track_lat"][0]))
            ay = (self.lat - self.d["track_lat"][0]) * D2KM
            tx = self.d["track_x"][:self.t+1]; ty = self.d["track_y"][:self.t+1]
            j = int(np.argmin((tx-ax)**2 + (ty-ay)**2))
            d_trail = float(np.hypot(tx[j]-ax, ty[j]-ay))
            gap = float(self.d["track_cum"][self.t] - self.d["track_cum"][j])
            eye_lon = self.d["track_lon"][j]; eye_lat = self.d["track_lat"][j]
            dist = d_trail
            reward = -min(d_trail, 300.0) / 100.0 - min(gap, 300.0) / 100.0
            reward += 1.0 if d_trail < R_CLOSE else 0.0
        else:
            eye_lon = self.d["track_lon"][self.t]; eye_lat = self.d["track_lat"][self.t]
            dx = (self.lon - eye_lon) * D2KM * np.cos(np.deg2rad(eye_lat))
            dy = (self.lat - eye_lat) * D2KM
            dist = float(np.hypot(dx, dy))
            reward = -min(dist, 300.0) / 100.0
            reward += 1.0 if dist < R_CLOSE else 0.0
        self._tgt_lon = eye_lon; self._tgt_lat = eye_lat
        interp = RegularGridInterpolator((la, lo), self.d["dist_to_land"], bounds_error=False, fill_value=0.0)
        land_dist = float(interp([[self.lat, self.lon]])[0])
        on_land = land_dist <= 0.0
        if on_land:
            reward -= LAND_PENALTY
        else:
            reward -= W_BUFFER * max(0.0, 1.0 - land_dist / R_BUFFER)
        terminated = False
        truncated = (self.t >= self.d["T"] - 1) or (not bool(self.d["in_domain"][self.t])) or (self.t >= self.d["landfall"])
        info = {"dist_km": dist, "eye_lon": eye_lon, "eye_lat": eye_lat, "storm": self.d["storm"], "on_land": on_land}
        return self._obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        return self._reset(seed=seed, options=options)


class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, k=8):
        super().__init__(env)
        self.k = k
        self._hist = deque(maxlen=k)
        lo = np.tile(env.observation_space.low, k).astype(np.float32)
        hi = np.tile(env.observation_space.high, k).astype(np.float32)
        self.observation_space = spaces.Box(low=lo, high=hi, dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._hist.clear()
        for _ in range(self.k):
            self._hist.append(obs)
        return self._obs(), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._hist.append(obs)
        return self._obs(), reward, term, trunc, info

    def _obs(self):
        return np.concatenate(list(self._hist)).astype(np.float32)
