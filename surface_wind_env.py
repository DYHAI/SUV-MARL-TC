
"""
海表 AUV 追踪台风眼环境（仅观测海表风场）。

状态(obs)：以 AUV 为中心、±PATCH_HALF_KM 的 N×N 风速(可加 u/v)局部场 + AUV 自身速度。
动作(act)：连续 2 维 [-1,1]² -> 期望速度 [vx,vy] = act*VMAX (m/s)，方向为 东/北。
动力学：1 小时步长，位置 += 速度*dt（换算经纬度），台风眼随时间沿 best track 移动。
奖励：r = -dist_eye_km / 100 （越近越大/越不惩罚）。

用法：本文件被 train/eval 脚本 import。
"""
import os
import warnings
from collections import deque
import numpy as np
import pandas as pd
import xarray as xr
import gymnasium as gym
from gymnasium import spaces
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

STORMS=["DOKSURI_2023","HAIKUI_2023","GUCHOL_2023","KHANUN_2023",
        "HINNAMNOR_2022","TOKAGE_2022","YAGI_2024","BEBINCA_2024"]
BASE=os.path.dirname(os.path.abspath(__file__))
ERA=os.path.join(BASE,"data","era5_wind")
TRACK=os.path.join(BASE,"data","selected_typhoons")
UNION=os.path.join(BASE,"data","union_region")
SST=os.path.join(BASE,"data","era5_sst")
MSLP=os.path.join(BASE,"data","era5_mslp")
R_EARTH=6371.0
D2KM=111.0
# 控制步长(分钟)；风场为逐小时，决策步在小时帧间线性插值。
# 可用环境变量 STEP_MIN=1/2/5/10 覆盖（决定步长消融实验）；默认 10。
STEP_MIN=int(os.environ.get("STEP_MIN", "10"))
# 上陆重罚。默认 20（消融实验统一值）；可用环境变量覆盖，用于陆罚扫参（须在 import 前设置）。
LAND_PENALTY=float(os.environ.get("LAND_PENALTY", "20"))
R_BUFFER=100.0      # 海岸软缓冲半径 km
W_BUFFER=8.0        # 缓冲区内惩罚幅值（近岸较弱惩罚）


def _align_grid(xr_var, lat_ref, lon_ref):
    """把某个变量的空间网格对齐到风场网格。

    ⚠️ 为什么必须做（2026-10-09 实测踩坑）：同一个台风的三个变量**可能来自不同 bbox 口径**的下载——
      老 200 是并集 bbox（73°×50°），新增台风是 tight（轨迹+450 km）或"合并请求"切出来的，
      混用会出现「wind 199 个纬度、msl 只有 95 个」→ 后面 `msl[:, im, :]` 直接 IndexError，
      而且这个错会**让整个 500 阶段训练一起崩**。
    ERA5 都是 0.25° 规则网格，插值到风场网格在**重合区域是逐点恒等**的；
    不重合的部分（tight 网格覆盖不到并集边缘）只在 track±450 km 之外，而训练侧本来就只保留 ±400 km，
    不会用到，因此不影响数据一致性。
    """
    try:
        same = (xr_var.sizes.get("latitude") == lat_ref.size
                and xr_var.sizes.get("longitude") == lon_ref.size
                and np.allclose(xr_var["latitude"].values, lat_ref)
                and np.allclose(xr_var["longitude"].values, lon_ref))
    except Exception:
        same = False
    if same:
        return xr_var
    return xr_var.interp(latitude=lat_ref, longitude=lon_ref, method="linear")


def load_storm(storm, use_uv=False, crop_margin_km=None, use_sst=False, use_hycom0=False, hycom0_native=False, use_mslp=False):
    ds=xr.open_dataset(os.path.join(ERA,storm,"era5_10m_wind_hourly.nc"))
    u=ds["u10"].astype("float32").values   # (T, lat, lon)  lat 降序（逐小时）
    v=ds["v10"].astype("float32").values
    lon=ds["longitude"].values.astype("float64")  # 升序
    lat=ds["latitude"].values.astype("float64")   # 降序
    times=ds["time"].values
    ds.close()
    lat_asc=lat[::-1].copy()
    u=u[:,::-1,:].copy(); v=v[:,::-1,:].copy()
    spd=np.hypot(u,v)
    sst=None
    sst_anom=None
    if use_sst:
        ds2=xr.open_dataset(os.path.join(SST,storm,"era5_sst_hourly.nc"))
        sst_xr=ds2["sst"].astype("float32")
        # 对齐到风场时间轴（个别台风风/SST 时间窗差 1 小时，nearest 对齐避免越界）
        sst_xr=sst_xr.reindex(time=times, method="nearest")
        # 空间网格对齐（并集 bbox / tight / 合并切分三种口径可能不一致）
        sst_xr=_align_grid(sst_xr, lat, lon)
        sst_raw=sst_xr.values  # (T_wind, lat降序, lon)，陆地 NaN
        ds2.close()
        sst_raw=sst_raw[:,::-1,:].copy()  # 翻成升序 lat
        # 背景 = 每个格点的时间中位(忽略 NaN)，异常 = SST - 背景 → 只保留台风冷尾迹信号
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sst_bg=np.nanmedian(sst_raw, axis=0, keepdims=True)
        sst_anom=sst_raw-sst_bg
        sst_anom=np.nan_to_num(sst_anom, nan=0.0)  # 陆地异常=0
        sst=np.nan_to_num(sst_raw, nan=273.15)     # 原始 SST，陆地填 0°C

    msl=None
    if use_mslp:
        ds3=xr.open_dataset(os.path.join(MSLP,storm,"era5_mslp_hourly.nc"))
        msl_xr=ds3["msl"].astype("float32")
        msl_xr=msl_xr.reindex(time=times, method="nearest")
        msl_xr=_align_grid(msl_xr, lat, lon)     # 同上：网格口径可能不同
        msl=msl_xr.values
        ds3.close()
        msl=msl[:,::-1,:].copy()  # 翻成升序 lat

    hy_u=hy_v=hy_spd=None
    hy_lat=hy_lon=None
    if use_hycom0:
        hy=xr.open_dataset(os.path.join(UNION,storm,"hycom_uv_0m.nc"))
        hyu=hy["water_u"].astype("float32").isel(depth=0)   # (T, lat升序, lon升序)
        hyv=hy["water_v"].astype("float32").isel(depth=0)
        if hycom0_native:
            # 保留 HYCOM 自身空间分辨率，仅时间插值到逐小时
            hyu=hyu.interp(time=times, method="linear")
            hyv=hyv.interp(time=times, method="linear")
            hy_lat=hy["lat"].values.astype("float64")  # 升序
            hy_lon=hy["lon"].values.astype("float64")  # 升序
        else:
            # 空间插值到 ERA5 网格(lat_asc, lon) + 时间插值到逐小时
            hyu=hyu.interp(lat=lat_asc, lon=lon, time=times, method="linear")
            hyv=hyv.interp(lat=lat_asc, lon=lon, time=times, method="linear")
            hy_lat=lat_asc; hy_lon=lon
        hy_u_raw=hyu.values.astype("float32"); hy_v_raw=hyv.values.astype("float32")
        hy.close()
        # HYCOM 距平（台风诱发流场）= 减去时间中位背景，只保留 Ekman 输运 + 惯性振荡
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            hy_u_anom = hy_u_raw - np.nanmedian(hy_u_raw, axis=0, keepdims=True)
            hy_v_anom = hy_v_raw - np.nanmedian(hy_v_raw, axis=0, keepdims=True)
        hy_u_anom = np.nan_to_num(hy_u_anom, nan=0.0)
        hy_v_anom = np.nan_to_num(hy_v_anom, nan=0.0)
        hy_spd_anom = np.hypot(hy_u_anom, hy_v_anom)
        # 原始流速（对照用）
        hy_u=np.nan_to_num(hy_u_raw, nan=0.0); hy_v=np.nan_to_num(hy_v_raw, nan=0.0)
        hy_spd=np.hypot(hy_u, hy_v)
    tr=pd.read_csv(os.path.join(TRACK,storm,"track.csv"),parse_dates=["ISO_TIME"]).dropna(subset=["LAT","LON"]).sort_values("ISO_TIME")
    tt=tr["ISO_TIME"].values.astype("datetime64[ns]").astype("int64")

    # 10min 时间网格 + 台风眼轨迹插值（风场仍逐小时，靠 h0/h1/frac 在观测时插值）
    times10=pd.date_range(times[0], times[-1], freq=f"{STEP_MIN}min").values
    tq10=times10.astype("datetime64[ns]").astype("int64")
    lon_t=np.interp(tq10, tt, tr["LON"].values)
    lat_t=np.interp(tq10, tt, tr["LAT"].values)
    n=len(times10)
    # ---- 风圈半径（四象限均值，nmi → km）：用于 P5「按 7 级/10 级风圈布放」 ----
    def _radii(cols):
        v = tr[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True).values
        ok = ~np.isnan(v)
        if ok.sum() < 2:
            return np.full(n, np.nan)
        return np.interp(tq10, tt[ok], v[ok])
    NM2KM = 1.852
    r34_km = _radii(["USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW"]) * NM2KM
    r50_km = _radii(["USA_R50_NE", "USA_R50_SE", "USA_R50_SW", "USA_R50_NW"]) * NM2KM
    sub=max(1, 60//STEP_MIN)                     # 每个小时内有多少步(10min -> 6)
    h0=np.minimum(np.arange(n)//sub, len(times)-1)
    h1=np.minimum(h0+1, len(times)-1)
    frac=(np.arange(n)%sub)/float(sub)

    # 按轨迹 bbox + 布防半径缓冲裁剪，大幅省内存（AUV 只在轨迹附近活动）
    if crop_margin_km is not None:
        mid_lat = float(np.mean(lat_t))
        lon_margin = crop_margin_km / (111.0 * np.cos(np.deg2rad(mid_lat)))
        lat_margin = crop_margin_km / 111.0
        lo0 = float(lon_t.min() - lon_margin); lo1 = float(lon_t.max() + lon_margin)
        la0 = float(lat_t.min() - lat_margin); la1 = float(lat_t.max() + lat_margin)
        jm = (lon >= lo0) & (lon <= lo1)
        im = (lat_asc >= la0) & (lat_asc <= la1)
        if jm.sum() >= 2 and im.sum() >= 2:
            lon = lon[jm]; lat_asc = lat_asc[im]
            u = u[:, im, :][:, :, jm]; v = v[:, im, :][:, :, jm]; spd = spd[:, im, :][:, :, jm]
            if use_sst:
                sst = sst[:, im, :][:, :, jm]
                sst_anom = sst_anom[:, im, :][:, :, jm]
            if use_mslp:
                msl = msl[:, im, :][:, :, jm]
            if use_hycom0:
                if hycom0_native:
                    jm_hy = (hy_lon >= lo0) & (hy_lon <= lo1)
                    im_hy = (hy_lat >= la0) & (hy_lat <= la1)
                    hy_lon = hy_lon[jm_hy]; hy_lat = hy_lat[im_hy]
                    hy_u = hy_u[:, im_hy, :][:, :, jm_hy]
                    hy_v = hy_v[:, im_hy, :][:, :, jm_hy]
                    hy_spd = hy_spd[:, im_hy, :][:, :, jm_hy]
                    hy_u_anom = hy_u_anom[:, im_hy, :][:, :, jm_hy]
                    hy_v_anom = hy_v_anom[:, im_hy, :][:, :, jm_hy]
                    hy_spd_anom = hy_spd_anom[:, im_hy, :][:, :, jm_hy]
                else:
                    hy_u = hy_u[:, im, :][:, :, jm]
                    hy_v = hy_v[:, im, :][:, :, jm]
                    hy_spd = hy_spd[:, im, :][:, :, jm]
                    hy_u_anom = hy_u_anom[:, im, :][:, :, jm]
                    hy_v_anom = hy_v_anom[:, im, :][:, :, jm]
                    hy_spd_anom = hy_spd_anom[:, im, :][:, :, jm]

    in_domain=(lon_t>=lon.min())&(lon_t<=lon.max())&(lat_t>=lat_asc.min())&(lat_t<=lat_asc.max())
    d=dict(storm=storm, lon=lon, lat=lat_asc, times=times,
           track_lon=lon_t, track_lat=lat_t, in_domain=in_domain, T=n,
           h0=h0, h1=h1, frac=frac,
           r34_km=r34_km, r50_km=r50_km,
           landfall=final_landfall_index(lon_t, lat_t))
    d["spd"]=spd
    if use_uv:
        d["u"]=u; d["v"]=v
    if use_sst:
        d["sst"]=sst
        d["sst_anom"]=sst_anom
    if use_mslp:
        d["msl"]=msl
    if use_hycom0:
        d["hycom0_u"]=hy_u; d["hycom0_v"]=hy_v; d["hycom0_spd"]=hy_spd
        d["hycom0_u_anom"]=hy_u_anom; d["hycom0_v_anom"]=hy_v_anom; d["hycom0_spd_anom"]=hy_spd_anom
        d["hycom0_lat"]=hy_lat; d["hycom0_lon"]=hy_lon
    return d

_LAND_CACHE=None
def land_ocean_mask():
    """HYCOM 陆地掩膜插值到 ERA5 并集网格，返回 (ocean_frac, lon, lat_asc)。带缓存。"""
    global _LAND_CACHE
    if _LAND_CACHE is None:
        hy=xr.open_dataset(os.path.join(UNION,"DOKSURI_2023","hycom_uv_0m.nc"))
        u=hy["water_u"].isel(time=0, depth=0).values
        hlon=hy["lon"].values.astype("float64"); hlat=hy["lat"].values.astype("float64")
        hy.close()
        ocean=(~np.isnan(u)).astype("float32")
        ds=xr.open_dataset(os.path.join(ERA,"DOKSURI_2023","era5_10m_wind_hourly.nc"))
        elon=ds["longitude"].values.astype("float64"); elat=ds["latitude"].values[::-1].astype("float64")
        ds.close()
        interp=RegularGridInterpolator((hlat,hlon),ocean,bounds_error=False,fill_value=0.0)
        EL,ELAT=np.meshgrid(elon,elat)
        of=interp(np.stack([ELAT.ravel(),EL.ravel()],axis=1)).reshape(EL.shape).astype("float32")
        # 到最近陆地的距离(km)：distance_transform_edt(海洋掩膜)=每个网格点到最近陆地(0)的距离
        ocean_mask=(of>=0.5).astype(np.float32)
        lat_sp=111.0*float(np.diff(elat).mean())
        lon_sp=111.0*np.cos(np.radians(float(elat.mean())))*float(np.diff(elon).mean())
        dist_to_land=distance_transform_edt(ocean_mask, sampling=(lat_sp,lon_sp)).astype(np.float32)
        _LAND_CACHE=(of,dist_to_land,elon,elat)
    return _LAND_CACHE

def final_landfall_index(lon_t, lat_t):
    """返回"最终登陆"时刻索引：台风眼上陆且不再回到海洋（穿过台湾/菲律宾又回海的不算）。
    若整段都未最终登陆（或结束在海上/域外），返回 len(lon_t) 表示不因登陆终止。"""
    of,_,elon,elat=land_ocean_mask()
    interp=RegularGridInterpolator((elat,elon),of,bounds_error=False,fill_value=0.0)
    oc=interp(np.stack([lat_t,lon_t],axis=1))
    on_land=oc<0.5
    if on_land[-1]:
        i=len(on_land)-1
        while i>0 and on_land[i-1]:
            i-=1
        return i
    return len(on_land)

class SurfaceWindEnv(gym.Env):
    def __init__(self, storms=STORMS, spacing_deg=0.25, N=3, vmax=20.0,
                 use_uv=False, start_radius=200.0, land_penalty=5.0, max_episode_steps=None,
                 preloaded=None, smooth_penalty=0.0):
        super().__init__()
        if preloaded is not None:
            self.data = preloaded
        else:
            self.data = {s: load_storm(s, use_uv=use_uv) for s in storms}
        self.storms = list(self.data.keys())
        self.spacing_deg=spacing_deg; self.N=N; self.vmax=vmax
        self.use_uv=use_uv; self.start_radius=start_radius; self.land_penalty=land_penalty
        self.smooth_penalty=smooth_penalty; self.prev_action=None
        of, dist_to_land, elon, elat = land_ocean_mask()
        self.land_of=of; self.land_lon=elon; self.land_lat=elat
        self.land_interp=RegularGridInterpolator((elat,elon),of,bounds_error=False,fill_value=0.0)
        self.dist_interp=RegularGridInterpolator((elat,elon),dist_to_land,bounds_error=False,fill_value=0.0)
        self.nch=3 if use_uv else 1
        self.obs_dim=self.nch*N*N + 2
        self.observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space=spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.step_count=0

    def _reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s=np.random.choice(self.storms)
        self.d=self.data[s]
        t0=int(np.where(self.d["in_domain"])[0][0])   # 台风生成点（首个域内时刻）
        self.t=t0
        # AUV 在台风起始点 start_radius 内随机布防
        ang=np.random.uniform(0,2*np.pi); r=np.random.uniform(0,self.start_radius)
        self.lon=self.d["track_lon"][t0] + r*np.cos(ang)/(D2KM*np.cos(np.deg2rad(self.d["track_lat"][t0])))
        self.lat=self.d["track_lat"][t0] + r*np.sin(ang)/D2KM
        lo=self.d["lon"]; la=self.d["lat"]
        self.lon=float(np.clip(self.lon, lo.min(), lo.max()))
        self.lat=float(np.clip(self.lat, la.min(), la.max()))
        self.vx=0.0; self.vy=0.0
        self.prev_action=np.zeros(2, dtype=np.float32)
        self.step_count=0
        return self._obs(), {}

    def _obs(self):
        t=self.t
        h0=int(self.d["h0"][t]); h1=int(self.d["h1"][t]); fr=float(self.d["frac"][t])
        lon=self.d["lon"]; lat=self.d["lat"]
        offs=(np.arange(self.N)-(self.N-1)/2)*self.spacing_deg
        xs=self.lon+offs   # 0.25° 间距
        ys=self.lat+offs
        X,Y=np.meshgrid(xs,ys)
        pts=np.stack([Y.ravel(),X.ravel()],axis=1)  # [lat,lon]
        spd=(1-fr)*self.d["spd"][h0]+fr*self.d["spd"][h1]
        if self.use_uv:
            uu=(1-fr)*self.d["u"][h0]+fr*self.d["u"][h1]
            vv=(1-fr)*self.d["v"][h0]+fr*self.d["v"][h1]
        Ip=RegularGridInterpolator((lat,lon),spd,bounds_error=False,fill_value=0.0)
        patch=Ip(pts).reshape(self.N,self.N)
        obs=[patch.ravel()/50.0]
        if self.use_uv:
            Iu=RegularGridInterpolator((lat,lon),uu,bounds_error=False,fill_value=0.0)
            Iv=RegularGridInterpolator((lat,lon),vv,bounds_error=False,fill_value=0.0)
            obs.append(Iu(pts).reshape(self.N,self.N).ravel()/50.0)
            obs.append(Iv(pts).reshape(self.N,self.N).ravel()/50.0)
        obs.append(np.array([self.vx/self.vmax, self.vy/self.vmax], dtype=np.float32))
        return np.concatenate(obs).astype(np.float32)

    def step(self, action):
        a=np.clip(action,-1,1)
        a_change = a - (self.prev_action if self.prev_action is not None else a)
        self.prev_action = a.copy()
        self.vx=a[0]*self.vmax; self.vy=a[1]*self.vmax
        dt=float(STEP_MIN*60)
        dx_km=self.vx*dt/1000.0
        dy_km=self.vy*dt/1000.0
        self.lon += dx_km/(D2KM*np.cos(np.deg2rad(self.lat)))
        self.lat += dy_km/D2KM
        # 边界 clamp
        lo=self.d["lon"]; la=self.d["lat"]
        self.lon=float(np.clip(self.lon, lo.min(), lo.max()))
        self.lat=float(np.clip(self.lat, la.min(), la.max()))
        self.t=min(self.t+1, self.d["T"]-1)
        self.step_count+=1
        eye_lon=self.d["track_lon"][self.t]; eye_lat=self.d["track_lat"][self.t]
        dx=(self.lon-eye_lon)*D2KM*np.cos(np.deg2rad(eye_lat))
        dy=(self.lat-eye_lat)*D2KM
        dist=float(np.hypot(dx,dy))
        reward=-min(dist,300.0)/100.0   # 距离奖励限幅 [-3,0]
        # 若非常贴近眼，加小奖励
        reward += 1.0 if dist<100.0 else 0.0
        if self.smooth_penalty > 0:      # 动作平滑惩罚（减少路径抖动）
            reward -= self.smooth_penalty * float(np.sum(a_change*a_change))
        # 陆地惩罚（与水下 AUV 一致：上陆 -20，100km 软缓冲 -8·(1-d/100)）
        land_dist=float(self.dist_interp([[self.lat,self.lon]])[0])
        on_land=land_dist<=0.0
        if on_land:
            reward -= LAND_PENALTY
        else:
            reward -= W_BUFFER*max(0.0, 1.0-land_dist/R_BUFFER)
        terminated=False
        truncated=(self.t>=self.d["T"]-1) or (not bool(self.d["in_domain"][self.t])) or (self.t>=self.d["landfall"])
        info={"dist_km":dist, "eye_lon":eye_lon, "eye_lat":eye_lat, "storm":self.d["storm"], "on_land":on_land}
        return self._obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        return self._reset(seed=seed, options=options)

class FrameStack(gym.ObservationWrapper):
    """把最近 k 步观测拼成一个向量，给 MLP 提供短时记忆。"""
    def __init__(self, env, k=8):
        super().__init__(env)
        self.k=k
        self._hist=deque(maxlen=k)
        lo=np.tile(env.observation_space.low, k).astype(np.float32)
        hi=np.tile(env.observation_space.high, k).astype(np.float32)
        self.observation_space=spaces.Box(low=lo, high=hi, dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, info=self.env.reset(seed=seed, options=options)
        self._hist.clear()
        for _ in range(self.k):
            self._hist.append(obs)
        return self._obs(), info

    def step(self, action):
        obs, reward, term, trunc, info=self.env.step(action)
        self._hist.append(obs)
        return self._obs(), reward, term, trunc, info

    def _obs(self):
        return np.concatenate(list(self._hist)).astype(np.float32)
