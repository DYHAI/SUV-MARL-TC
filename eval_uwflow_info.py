# -*- coding: utf-8 -*-
"""P3 评测：把观测收益从「自身位置流速」换成「覆盖/采样到的流场信息量」。

对同一批测试台风段、同一组随机种子（同起点布防），分别跑：
  idle(漂流) / random / 旧奖励(E 配置, 追自身位置流速) / 新奖励 G(信息增益)
统计两族指标：

【旧族 · 策略自身收益】
  mean_V        AUV 所在位置的流速均值（旧奖励优化的目标）
  mean_a2       推进能耗代理

【新族 · 科学目标（找到核心区域去测量）】
  cov_area_km2  4 台 AUV 覆盖到的不重复面积（0.05° 格点累加）
  info_cov_pct  覆盖到的「信息价值」占评估区总价值的比例（W 加权覆盖率）
  core_hit_pct  评估区内 W 前 10% 的核心区被采样到的比例
  recon_rmse    用采样点 IDW 重构评估区时均流场，与 HYCOM 真值的 RMSE（m/s）

评估区定义：以 episode 时间窗内台风眼轨迹均值位置为圆心、半径 R_EVAL=300km 的海洋格点
（不含陆地）。所有策略在同一 episode（同种子 → 同段、同布防）上对比。

用法:
  python eval_uwflow_info.py <n_seed> <depth_m> <start_km> <dur_h> <model.pt> [model2.pt ...]
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from underwater_fleet_flow_env import UnderwaterFleetFlowEnv, D2KM  # noqa: E402
from underwater_flow_env import STEP_MIN  # noqa: E402


class Actor(nn.Module):
    """与 train_underwater_fleet_flow_shared.Actor 完全一致（本地复制，避免 import
    训练脚本时它按自己的 sys.argv 解析参数而报错）。"""

    def __init__(self, obs_dim, act_dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, act_dim))
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, x):
        mean = self.net(x)
        std = self.log_std.exp().clamp(min=1e-4, max=1.0)
        return mean, std

BASE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.environ.get("SPLIT_JSON", os.path.join(BASE, "data", "typhoon_split_64.json"))
SEGCSV = os.environ.get("SEGMENTS_CSV", os.path.join(BASE, "data", "segments_64.csv"))
OUT = os.path.abspath(
    os.environ.get("UWFLOW_MODEL_DIR", os.path.join(BASE, "ppo_underwater_fleet"))
)
EVAL_OUT = os.path.abspath(
    os.environ.get("UWFLOW_EVAL_DIR", os.path.join(BASE, "eval_out"))
)
R_EVAL_KM = 250.0
IDW_POWER = 2.0
IDW_K = 8


def make_env(depth, start, dur, storms, seg, n_patch=1, frame_stack=True,
             vmax=None, reward=None, grid=None):
    native = os.environ.get("NATIVE_RES", "0") == "1"
    vmax = float(os.environ.get("VMAX", 1.5)) if vmax is None else float(vmax)
    reward = os.environ.get("REWARD", "G") if reward is None else reward
    grid = float(os.environ.get("GRID_DEG", 0.25)) if grid is None else float(grid)
    depths_env = os.environ.get("DEPTHS", "").strip()
    if depths_env:      # P19 × P3：4 台 AUV 分别观测不同深度
        from underwater_fleet_depth_env import UnderwaterFleetFlowEnv as _Env
        extra = {"depths": [float(x) for x in depths_env.split(",")]}
    else:
        _Env = UnderwaterFleetFlowEnv
        extra = {}
    return _Env(
        storms=storms, segments_df=seg, vmax=vmax, reward_mode=reward,
        N=n_patch, use_frame_stack=frame_stack, frame_k=5,
        start_radius=float(start), crop_margin=(3.0 if start <= 200 else 6.0),
        depth_m=depth, grid_deg=grid, duration_h=float(dur), native_res=native, **extra)


def load_actor(name, obs_dim):
    w = torch.load(os.path.join(OUT, name), weights_only=True)
    if isinstance(w, dict) and "actor" in w:
        w = w["actor"]
    a = Actor(obs_dim)
    a.load_state_dict(w)
    a.eval()
    return a


def load_policy(name, obs_dim):
    """统一入口：支持手写 IPPO 的 .pt 和 SB3 PPO 的 .zip，统一产出 policy(obs, env)->(n,2)。"""
    if name.endswith(".zip"):
        from stable_baselines3 import PPO
        model = PPO.load(os.path.join(OUT, name), device="cpu")

        def pol(o, e, model=model):
            a, _ = model.predict(np.asarray(o, dtype=np.float32).reshape(-1), deterministic=True)
            return np.clip(np.asarray(a, dtype=np.float32).reshape(e.n_auv, 2), -1, 1)
        return pol
    actor = load_actor(name, obs_dim)

    def pol(o, e, actor=actor):
        ot = torch.tensor(o, dtype=torch.float32)
        with torch.no_grad():
            mean, _ = actor.dist(ot)
        return np.clip(mean.numpy(), -1, 1)
    return pol


def rollout(env, policy, seed, row=None):
    """跑一个 episode，返回采样点与动作记录。policy(obs, env)->(n,2) action。"""
    if row is not None:
        env.segments = pd.DataFrame([row]).reset_index(drop=True)
    np.random.seed(seed)
    obs, _ = env.reset()
    slat, slon, sspd, sact, sW = [], [], [], [], []
    done = False
    while not done:
        a = policy(obs, env)
        slat.append(env.lats.copy()); slon.append(env.lons.copy())
        obs, _r, term, trunc, info = env.step(a)
        sspd.append(np.asarray(info["spd_at"], dtype=np.float64))
        sW.append(np.asarray(info["W_at"], dtype=np.float64))
        sact.append(np.linalg.norm(np.asarray(a, dtype=np.float64), axis=1))
        done = bool(term or trunc)
    return (np.concatenate(slat), np.concatenate(slon), np.concatenate(sspd),
            np.concatenate(sact), np.concatenate(sW))


def _region(env):
    """评估区：以布防时刻（t0）台风眼位置为圆心、R_EVAL_KM 为半径的海洋格点。

    布防就在 t0 眼位周围 200km 内，AUV 10h 最多走 ~54km，所以「任务区」≈ 以 t0 眼位
    为心的 250km 圆盘；用这个固定区域保证各策略比较的是同一块海域。
    """
    d = env.d
    t0 = int(env.t); t1 = int(env.t_end)
    clon = float(d["track_lon"][t0])
    clat = float(d["track_lat"][t0])
    LON, LAT = np.meshgrid(d["lon"], d["lat"])
    dx = (LON - clon) * D2KM * np.cos(np.deg2rad(clat))
    dy = (LAT - clat) * D2KM
    m = (np.hypot(dx, dy) < R_EVAL_KM) & (d["dist_to_land"] > 0.0)
    return m, clat, clon


def idw_to_region(env, slat, slon, sspd, region):
    """把采样点 IDW 重构到评估区格点，与真值时均场比较。"""
    d = env.d
    glat, glon = np.meshgrid(d["lat"], d["lon"], indexing="ij")
    ty = glat[region]; tx = glon[region]
    if len(slat) == 0:
        return np.nan, np.nan
    rec = np.empty(ty.size)
    for k in range(ty.size):
        dd = np.hypot((slat - ty[k]) * D2KM,
                      (slon - tx[k]) * D2KM * np.cos(np.deg2rad(ty[k])))
        idx = np.argsort(dd)[:IDW_K]
        dist = np.maximum(dd[idx], 1e-6)
        w = 1.0 / dist ** IDW_POWER
        rec[k] = float(np.sum(w * sspd[idx]) / np.sum(w))
    truth = env.M_mean[region]
    rmse = float(np.sqrt(np.mean((rec - truth) ** 2)))
    corr = float(np.corrcoef(rec, truth)[0, 1]) if ty.size > 3 else np.nan
    return rmse, corr


def episode_metrics(env, slat, slon, sspd, sact, sW):
    d = env.d
    region, _, _ = _region(env)
    lat_g = d["lat"]; lon_g = d["lon"]
    cov = env.cov_deg
    lat0 = float(lat_g.min()); lon0 = float(lon_g.min())
    # 覆盖格点：**直接从 4 台 AUV 的轨迹点算**，对所有策略一视同仁。
    # （旧实现读 env._cov_cells，而它只在 reward_mode=="G" 时才会被 step() 填充，
    #   导致 B/C 奖励的模型覆盖率永远是 0。）
    covered_grid = np.zeros_like(region, dtype=bool)
    cov_cells = set()
    for clat_, clon_ in zip(np.asarray(slat).ravel(), np.asarray(slon).ravel()):
        i = int(np.floor((clat_ - lat0) / cov))
        j = int(np.floor((clon_ - lon0) / cov))
        cov_cells.add((i, j))
        ii = int(round((lat0 + (i + 0.5) * cov - lat_g[0]) / (lat_g[1] - lat_g[0])))
        jj = int(round((lon0 + (j + 0.5) * cov - lon_g[0]) / (lon_g[1] - lon_g[0])))
        if 0 <= ii < len(lat_g) and 0 <= jj < len(lon_g):
            covered_grid[ii, jj] = True
    n_cells = len(cov_cells)
    # 航迹长度 / 平均航速（用户要的另外两个指标）
    n_auv = env.n_auv
    LAT = np.asarray(slat, dtype=np.float64).reshape(-1, n_auv)
    LON = np.asarray(slon, dtype=np.float64).reshape(-1, n_auv)
    if LAT.shape[0] > 1:
        seg = np.hypot((LAT[1:] - LAT[:-1]) * D2KM,
                       (LON[1:] - LON[:-1]) * D2KM * np.cos(np.deg2rad(LAT[:-1])))
        per_auv_path = seg.sum(axis=0)
        path_km = float(per_auv_path.mean())
    else:
        path_km = 0.0
    dur_s = max(len(LAT) - 1, 1) * float(STEP_MIN) * 60.0
    mean_speed = path_km * 1000.0 / dur_s
    # 推进速度（不含水流）：动作幅值 × Vmax。地速 = 推进 + 流，所以地速可以 > Vmax。
    prop_speed = float(np.mean(sact)) * float(getattr(env, "vmax", np.nan))
    sspd_arr = np.asarray(sspd, dtype=np.float64)
    W = env.W
    Wr = W[region]
    tot_W = float(Wr.sum())
    info_cov = float(W[region & covered_grid].sum() / tot_W) if tot_W > 0 else np.nan
    # 核心区（W 前 10%）命中率
    thr = float(np.percentile(Wr, 90)) if Wr.size > 10 else np.inf
    core = region & (W >= thr)
    core_hit = float(covered_grid[core].mean()) if core.sum() > 0 else np.nan
    cov_in_region = region & covered_grid
    core_prec = (float((W[cov_in_region] >= thr).mean())
                 if cov_in_region.sum() > 0 else np.nan)
    # ---- 分层调查口径：每台 AUV 用「自己那一层」的 W 计覆盖 ----
    info_layer = np.nan
    core_layer = np.nan
    if getattr(env, "W_by_depth", None) and getattr(env, "_cov_cells_i", None):
        num = 0.0; den = 0.0; nhit = 0.0; ncore = 0.0
        for i, dd in enumerate(env.depths):
            Wd = env.W_by_depth.get(dd)
            if Wd is None:
                continue
            cg = np.zeros_like(region, dtype=bool)
            for (ii, jj) in env._cov_cells_i[i]:
                clat = lat0 + (ii + 0.5) * cov
                clon = lon0 + (jj + 0.5) * cov
                gi = int(round((clat - lat_g[0]) / (lat_g[1] - lat_g[0])))
                gj = int(round((clon - lon_g[0]) / (lon_g[1] - lon_g[0])))
                if 0 <= gi < len(lat_g) and 0 <= gj < len(lon_g):
                    cg[gi, gj] = True
            Wr_d = Wd[region]
            den += float(Wr_d.sum())
            num += float(Wd[region & cg].sum())
            thr_d = float(np.percentile(Wr_d, 90)) if Wr_d.size > 10 else np.inf
            core_d = region & (Wd >= thr_d)
            ncore += float(core_d.sum())
            nhit += float(cg[core_d].sum())
        if den > 0:
            info_layer = num / den * 100.0
        if ncore > 0:
            core_layer = nhit / ncore * 100.0
    rmse, corr = idw_to_region(env, slat, slon, sspd, region)
    return {
        "storm": env.d["storm"],
        "mean_V": float(sspd.mean()),
        "mean_a2": float((sact ** 2).mean()),
        "cov_area_km2": float(n_cells * env._cell_area_km2),
        "info_cov_pct": info_cov * 100.0,
        "core_hit_pct": core_hit * 100.0,
        "core_prec_pct": core_prec * 100.0,
        "info_layer_pct": info_layer,
        "core_layer_pct": core_layer,
        "recon_rmse": rmse,
        "recon_corr": corr,
        "path_km": path_km,
        "mean_speed_ms": mean_speed,
        "prop_speed_ms": prop_speed,
        "V_p50": float(np.percentile(sspd_arr, 50)),
        "V_p90": float(np.percentile(sspd_arr, 90)),
        "V_p95": float(np.percentile(sspd_arr, 95)),
        "high_pct": float((sspd_arr > 0.5).mean() * 100.0),
        "mean_W_at": float(np.nanmean(sW)) if len(sW) else np.nan,
        "region_cells": int(region.sum()),
    }


def main():
    n_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    depth = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
    start = float(sys.argv[3]) if len(sys.argv) > 3 else 200.0
    dur = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
    models = sys.argv[5:]

    split = json.load(open(SPLIT)); test = split["test"]
    seg = pd.read_csv(SEGCSV)
    seg = seg[seg.storm.isin(test)].reset_index(drop=True)
    picks = [sub.iloc[len(sub) // 2] for _, sub in seg.groupby("storm")]
    seg_pick = pd.DataFrame(picks).reset_index(drop=True)

    env = make_env(depth, start, dur, test, seg_pick)
    obs_dim = env.obs_dim
    print("test storms=%d segs=%d obs_dim=%d depth=%gm start=%gkm dur=%gh"
          % (len(test), len(seg_pick), obs_dim, depth, start, dur), flush=True)
    print("R_EVAL=%.0fkm  episodes=%d  models=%s\n" % (R_EVAL_KM, n_seed, models), flush=True)

    policies = [
        ("idle(漂流)", lambda o, e: np.zeros((e.n_auv, 2), dtype=np.float32)),
        ("random", lambda o, e: np.random.uniform(-1, 1, (e.n_auv, 2)).astype(np.float32)),
    ]
    for name in models:
        pol = load_policy(name, obs_dim)
        label = os.path.basename(name).replace(".pt", "").replace(".zip", "")
        policies.append((label, pol))

    rows = []
    for name, pol in policies:
        accs = []
        for storm in test:
            row = seg_pick[seg_pick.storm == storm].iloc[0]
            for si in range(n_seed):
                m = episode_metrics(env, *rollout(env, pol, seed=si + 100, row=row))
                m["policy"] = name; m["seed"] = si
                accs.append(m)
                rows.append(m)
        df = pd.DataFrame(accs)
        print("%-30s |V|=%.4f a2=%.3f cov=%6.0fkm2 info=%4.1f%% coreHit=%4.1f%% "
              "corePrec=%4.1f%% rmse=%.4f corr=%.3f" % (
                  name, df.mean_V.mean(), df.mean_a2.mean(), df.cov_area_km2.mean(),
                  df.info_cov_pct.mean(), df.core_hit_pct.mean(), df.core_prec_pct.mean(),
                  df.recon_rmse.mean(), df.recon_corr.mean()), flush=True)
        if df.info_layer_pct.notna().any():
            print("%-30s   ↳ 分层口径: infoLayer=%4.1f%%  coreLayer=%4.1f%%" % (
                "", df.info_layer_pct.mean(), df.core_layer_pct.mean()), flush=True)

    out = EVAL_OUT
    os.makedirs(out, exist_ok=True)
    tag = "info_d%g_start%g_%gh%s" % (depth, start, dur, os.environ.get("EVAL_TAG", ""))
    pd.DataFrame(rows).to_csv(os.path.join(out, "eval_info_%s.csv" % tag), index=False)
    print("\nsaved %s/eval_info_%s.csv" % (out, tag))


if __name__ == "__main__":
    main()
