# -*- coding: utf-8 -*-
"""四组模型统一评测：追踪距离 / 航迹长度 / 平均航速 / 最大速度 / 能耗代理 / 动作抖动。

支持三种策略类型（自动判断）：
  mlp  : stable_baselines3.PPO  + SharedPerAgentPolicy
  rec  : sb3_contrib.RecurrentPPO + SharedPerAgentRecurrentPolicy（需带 LSTM 隐状态）
  actf : PPO + 动作一阶低通（需带训练时同一个 β）

能耗代理：动作 a∈[-1,1]² 圆形裁剪 → 速度 v=a·Vmax(20 m/s)。位移功耗 ∝ v³（阻力主导），
低速近似 ∝ v²。这里同时给 E2=Σ|v|²Δt、E3=Σ|v|³Δt（按每台 AUV、每回合计），
并折算到「每小时」和「每公里航迹」。

用法: python eval_fleet_energy.py <model.zip> <kind:mlp|rec|actf> [beta] [n_auv] [deploy] [spacing]
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fleet_segment_env import (FleetSegmentEnv, D2KM, deploy_formation, seg_idx, STEP_MIN,
                               sample_deployment, safety_filter)
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from recurrent_shared_policy import SharedPerAgentRecurrentPolicy  # noqa: F401

BASE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.environ.get("SPLIT_JSON", os.path.join(BASE, "data", "typhoon_split_64.json"))
SEGCSV = os.environ.get("SEGMENTS_CSV", os.path.join(BASE, "data", "segments_64.csv"))

MODEL = sys.argv[1]
KIND = sys.argv[2] if len(sys.argv) > 2 else "mlp"
BETA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
N_AUV = int(sys.argv[4]) if len(sys.argv) > 4 else 4
DEPLOY = float(sys.argv[5]) if len(sys.argv) > 5 else 200.0
SPACING = float(sys.argv[6]) if len(sys.argv) > 6 else 0.25
RING_RADIUS, FORMATION, PATCH_N = 100.0, "ring", 1
VMAX = 20.0
SEEDS = 1
SPLIT_KEY = os.environ.get("SPLIT_KEY", "test")
# 避陆布防（默认 0 = 旧口径，保证历史数字逐位可复现）
DEPLOY_OCEAN_ONLY = os.environ.get("DEPLOY_OCEAN_ONLY", "0") == "1"
DEPLOY_MIN_COAST_KM = float(os.environ.get("DEPLOY_MIN_COAST_KM", "0"))
SAFETY_MARGIN_KM = float(os.environ.get("SAFETY_MARGIN_KM", "0"))


def run_segment(model, env, seg, seed):
    s = seg["storm"]; d = env.data[s]
    env.d = d; env.t0 = seg_idx(seg["t0"]); env.t1 = seg_idx(seg["t1"]); env.t = env.t0
    np.random.seed(seed)
    elo = d["track_lon"][env.t0]; ela = d["track_lat"][env.t0]
    env.vxs = np.zeros(env.n_auv); env.vys = np.zeros(env.n_auv)
    if DEPLOY_OCEAN_ONLY or DEPLOY_MIN_COAST_KM > 0:
        env.lons, env.lats = sample_deployment(
            d, env.t0, env.n_auv, DEPLOY, FORMATION, 100.0, RING_RADIUS, "fixed",
            ocean_only=DEPLOY_OCEAN_ONLY, min_coast_km=DEPLOY_MIN_COAST_KM)
    else:
        # 旧口径（抽取顺序保持原样）：cang -> cr -> 编队 -> clip
        cang = np.random.uniform(0, 2 * np.pi); cr = np.random.uniform(0, 1.0) * DEPLOY
        clon = elo + cr * np.cos(cang) / (D2KM * np.cos(np.deg2rad(ela)))
        clat = ela + cr * np.sin(cang) / D2KM
        env.lons, env.lats = deploy_formation(clon, clat, env.n_auv, FORMATION, 100.0, RING_RADIUS)
        lo = d["lon"]; la = d["lat"]
        env.lons = np.clip(env.lons, lo.min(), lo.max())
        env.lats = np.clip(env.lats, la.min(), la.max())
    env.prev_actions = np.zeros((env.n_auv, 2), dtype=np.float32)
    obs = env._obs().reshape(-1).astype(np.float32)
    state, ep_start = None, np.ones((1,), dtype=bool)
    a_exec = np.zeros((env.n_auv, 2), dtype=np.float32)
    prev = np.zeros((env.n_auv, 2), dtype=np.float32)
    dt_h = STEP_MIN / 60.0
    path = np.zeros(env.n_auv); e2 = np.zeros(env.n_auv); e3 = np.zeros(env.n_auv)
    vmax_seen = np.zeros(env.n_auv); jerk = np.zeros(env.n_auv); dists = []
    while True:
        if KIND == "rec":
            a, state = model.predict(obs, state=state, episode_start=ep_start, deterministic=True)
            ep_start = np.zeros((1,), dtype=bool)
        else:
            a, _ = model.predict(obs, deterministic=True)
        a = np.asarray(a, dtype=np.float32).reshape(env.n_auv, 2)
        if KIND == "actf":
            a_exec = (1.0 - BETA) * a + BETA * a_exec
            a_use = np.clip(a_exec, -1.0, 1.0)
        else:
            a_use = a
        # 与环境 step() 完全一致的圆形裁剪：先每轴裁到 [-1,1]，再保证合动作模 ≤1
        a_use = np.clip(a_use, -1.0, 1.0)
        nrm = np.linalg.norm(a_use, axis=1, keepdims=True)
        a_use = a_use / np.maximum(1.0, nrm)
        v = np.hypot(a_use[:, 0], a_use[:, 1]) * VMAX                  # 实际航速 m/s
        path += v * dt_h * 3600.0 / 1000.0                             # km
        e2 += v ** 2 * dt_h                                            # (m/s)^2 · h
        e3 += v ** 3 * dt_h                                            # (m/s)^3 · h
        vmax_seen = np.maximum(vmax_seen, v)
        jerk += np.abs(a_use - prev).sum(1)
        prev = a_use.copy()
        if SAFETY_MARGIN_KM > 0:
            a_use = safety_filter(
                a_use, env.lons, env.lats, env.dist_interp, SAFETY_MARGIN_KM
            )
        obs, _r, _t, trunc, info = env.step(a_use)
        obs = obs.reshape(-1).astype(np.float32)
        dists.append(float(np.mean(info["dist_km"])))
        if trunc:
            break
    T_h = len(dists) * dt_h
    return dict(dist=float(np.mean(dists)), path=float(path.mean()), T_h=float(T_h),
                final_dist=float(dists[-1]) if dists else float("nan"),
                e2=float(e2.mean()), e3=float(e3.mean()), vmax=float(vmax_seen.mean()),
                jerk=float(jerk.mean()),
                vmean=float(path.mean() * 1000.0 / (T_h * 3600.0)))   # km/h -> m/s


def main():
    model = RecurrentPPO.load(MODEL, device="cpu") if KIND == "rec" else PPO.load(MODEL, device="cpu")
    split = json.load(open(SPLIT)); storms = split[SPLIT_KEY]
    seg = pd.read_csv(SEGCSV); seg = seg[seg.storm.isin(storms)].reset_index(drop=True)
    env = FleetSegmentEnv(storms=storms, segments_df=seg, use_uv=True, n_auv=N_AUV,
                          smooth_penalty=0.05, spacing_deg=SPACING, patch_n=PATCH_N,
                          deploy_radius=DEPLOY, formation=FORMATION, ring_radius=RING_RADIUS)
    rows = []
    for _, row in seg.iterrows():
        for sd in range(SEEDS):
            rows.append(run_segment(model, env, row, sd))
    df = pd.DataFrame(rows)
    print("%s  kind=%s beta=%s  storms=%d segments=%d" % (os.path.basename(MODEL), KIND, BETA, len(storms), len(rows)))
    print("  追踪距离      %.1f km" % df.dist.mean())
    print("  最终中心距离  %.1f km" % df.final_dist.mean())
    print("  航迹长度      %.1f km  (单 AUV，全程)" % df.path.mean())
    print("  平均航速      %.2f m/s" % df.vmean.mean())
    print("  最大速度      %.2f m/s  (段均值)" % df.vmax.mean())
    print("  平均航行时间  %.2f h" % df.T_h.mean())
    print("  能耗 E2=Σv²h  %.0f" % df.e2.mean())
    print("  能耗 E3=Σv³h  %.0f" % df.e3.mean())
    print("  E3/小时       %.0f" % (df.e3 / df.T_h).mean())
    print("  E3/航迹km     %.1f" % (df.e3 / df.path).mean())
    print("  动作抖动 |Δa| %.2f" % df.jerk.mean())


if __name__ == "__main__":
    main()
