"""500 台风阶段：把『已有 200 + 新增 300』的轨迹切成 300-500 km 弧长连续段。

与 200 阶段（slice_segments_200.py）**切段算法逐行一致**（STRIDE=250km → 重叠均值 ~150km），
区别只有三点：
  · 台风清单 = typhoons_500_split.all_tags()（500 个）
  · 划分 = data/typhoon_split_500.json（450 训 / 50 测，由 typhoons_500_split.py 生成）
  · 输出 = data/segments_500.csv

⚠️ 依赖 ERA5 风场的时间轴（读 `era5_10m_wind_hourly.nc` 的 time 坐标）——
   所以**必须等 300 个新台风的 ERA5 下载完**才能跑完整版；
   数据没齐时会跳过缺文件台风并打印告警（可反复跑，齐了就是终版）。

用法: python slice_segments_500.py
"""
import os, json
import numpy as np
import pandas as pd
import xarray as xr

import typhoons_500_split as S

BASE = os.path.dirname(os.path.abspath(__file__))
ERA = os.path.join(BASE, "data", "era5_wind")
TRACK = os.path.join(BASE, "data", "selected_typhoons")
SPLIT = os.path.join(BASE, "data", "typhoon_split_500.json")
OUT_FINAL = os.path.join(BASE, "data", "segments_500.csv")
OUT_PARTIAL = os.path.join(BASE, "data", "segments_500_partial.csv")

D2KM = 111.0
STEP_MIN = 10
STRIDE = 250.0
LEN_MIN, LEN_MAX = 300.0, 500.0
SEED = 0


def load_track10(storm):
    ds = xr.open_dataset(os.path.join(ERA, storm, "era5_10m_wind_hourly.nc"))
    lon = ds["longitude"].values.astype("float64")
    lat = ds["latitude"].values.astype("float64")
    times = ds["time"].values
    ds.close()
    tr = pd.read_csv(os.path.join(TRACK, storm, "track.csv"), parse_dates=["ISO_TIME"]) \
           .dropna(subset=["LAT", "LON"]).sort_values("ISO_TIME")
    tt = tr["ISO_TIME"].values.astype("datetime64[ns]").astype("int64")
    times10 = pd.date_range(times[0], times[-1], freq=f"{STEP_MIN}min").values
    tq10 = times10.astype("datetime64[ns]").astype("int64")
    return (np.interp(tq10, tt, tr["LON"].values), np.interp(tq10, tt, tr["LAT"].values),
            lon, lat)


def cut_storm(s, rng):
    lon_t, lat_t, lon, lat = load_track10(s)
    dx = np.diff(lon_t) * D2KM * np.cos(np.deg2rad(lat_t[:-1]))
    dy = np.diff(lat_t) * D2KM
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(dx, dy))])
    total = float(cum[-1])
    rows, start = [], 0.0
    while start + LEN_MIN <= total:
        L = min(float(rng.uniform(LEN_MIN, LEN_MAX)), total - start)
        if L < LEN_MIN:
            break
        t0 = int(np.searchsorted(cum, start, side="left"))
        t1 = int(np.searchsorted(cum, start + L, side="right"))
        t0 = max(0, min(t0, len(cum) - 1)); t1 = max(t0 + 1, min(t1, len(cum) - 1))
        if t1 <= t0:
            break
        rows.append(dict(storm=s, t0=int(t0), t1=int(t1), arc_km=round(float(cum[t1] - cum[t0]), 1),
                         start_lon=round(float(lon_t[t0]), 4), start_lat=round(float(lat_t[t0]), 4),
                         end_lon=round(float(lon_t[t1]), 4), end_lat=round(float(lat_t[t1]), 4)))
        start += STRIDE
    return rows


def main():
    tags, seasons = S.all_tags()
    # ⚠️ 2026-10-09：判定口径走 typhoons_500_split.ready()（READY_MODE=era5 默认）。
    #    500 阶段追踪任务只读 ERA5（风/SST/MSLP），HYCOM 是另一条独立队列（服务器吞吐 40 文件/小时），
    #    等 HYCOM 齐会白等十几小时。要严格口径就 READY_MODE=all。
    ready = [t for t in tags if S.ready(t)]
    print(f"台风全集 {len(tags)}；数据齐全 {len(ready)}；缺 {len(tags)-len(ready)}"
          f"（READY_MODE={S.READY_MODE}）", flush=True)
    if len(ready) < len(tags):
        print("⚠️ 数据未齐，本次只切已齐的台风（可反复跑）")

    rng = np.random.default_rng(SEED)
    rows = []
    for i, s in enumerate(ready):
        try:
            rows += cut_storm(s, rng)
        except Exception as e:
            print(f"  跳过 {s}: {type(e).__name__} {str(e)[:60]}")
        if (i + 1) % 50 == 0:
            print(f"  切段 {i+1}/{len(ready)}  累计 {len(rows)} 段", flush=True)

    df = pd.DataFrame(rows)
    # ⚠️ 数据没齐时**只写 _partial**，避免半成品被当成终版（终版只在 500 个全齐时才写）
    complete = (len(ready) == len(tags))
    OUT = OUT_FINAL if complete else OUT_PARTIAL
    df.to_csv(OUT, index=False)
    print(f"\n总段数 {len(df)} -> {OUT}" + ("" if complete else "  ← 半成品（还缺台风）"))
    if len(df):
        print(f"段长 min/median/max = {df.arc_km.min():.0f}/{df.arc_km.median():.0f}/{df.arc_km.max():.0f} km")
        per = df.groupby("storm").size()
        print(f"每台风段数 min/median/max = {per.min()}/{per.median():.0f}/{per.max()}")
    # 划分：只在数据齐时生成终版
    if complete:
        train, test = S.make_split(ready, seasons)
        json.dump({"train": train, "test": test, "split_by": "storm",
                   "rule": f"season-stratified, seed={S.SEED}, n_test={S.N_TEST}",
                   "n_storms_total": len(ready)},
                  open(SPLIT, "w"), ensure_ascii=False, indent=2)
        print(f"\n划分 -> {SPLIT}: 训练 {len(train)} / 测试 {len(test)}")
        print(f"  训练段数 {int(df.storm.isin(train).sum())} / 测试段数 {int(df.storm.isin(test).sum())}")


if __name__ == "__main__":
    main()
