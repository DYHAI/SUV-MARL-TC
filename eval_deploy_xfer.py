"""布防半径**迁移**评测：用「训练半径 R_train 的权重」去跑「固定距离 D 布防」。

与 `eval_land_fraction.py` 的区别（其余口径**逐行照搬**，保证数字可比）：

1. `XFER_R_TRAIN`（环境变量）决定 en v 的 `deploy_radius` → 只影响**数据裁剪范围**和
   模型"以为"的布防上限；实际布防距离由 `DEPLOY_FIXED_KM` 覆盖（见 fleet_segment_env
   里 `sample_deployment(..., fixed_km=)`）：**布防中心固定离眼 D km，方位角随机**
   （照旧先抽一次 cr 再丢弃，所以随机数流与均匀协议逐位一致 → 同 seed 下方位序列相同）。
2. **每段的种子随段号变化**（`seed = 100000 + 97*idx + sd`），并且**与 D 无关**：
   - 消掉 `eval_land_fraction.py` 里 `SEEDS=1, seed=0` 导致「768 段全部同一个方位角」
     的退化（那会让布防永远落在台风路径的同一侧）；
   - 同一段在 5 个 D 上用的是**同一个方位角** → 严格配对，D 之间的差异纯粹来自距离。

用法:
    XFER_R_TRAIN=800 DEPLOY_FIXED_KM=200 STORAGE16=0 \
      SST_MODE=none USE_MSLP=0 \
      python eval_deploy_xfer.py <model.zip> <n_auv> <D_km> [seeds]
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fleet_segment_env import (FleetSegmentEnv, D2KM, deploy_formation, seg_idx, STEP_MIN,
                               VMAX, R_BUFFER, safety_filter)
from stable_baselines3 import PPO

BASE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.environ.get("SPLIT_JSON", os.path.join(BASE, "data", "typhoon_split_500.json"))
SEGCSV = os.environ.get("SEGMENTS_CSV", os.path.join(BASE, "data", "segments_500.csv"))
SPLIT_KEY = os.environ.get("SPLIT_KEY", "test")
DEPLOY_OCEAN_ONLY = os.environ.get("DEPLOY_OCEAN_ONLY", "1") == "1"
DEPLOY_MIN_COAST_KM = float(os.environ.get("DEPLOY_MIN_COAST_KM", "100"))
SAFETY_MARGIN_KM = float(os.environ.get("SAFETY_MARGIN_KM", "100"))
SST_MODE = os.environ.get("SST_MODE", "none")
USE_MSLP = os.environ.get("USE_MSLP", "0") == "1"
STORAGE16 = os.environ.get("STORAGE16", "0") == "1"
# XFER_DUMP=1 → 额外把**逐段**指标写 logs_xfer/<XFER_TAG>_seg.csv（配对检验用）
XFER_DUMP = os.environ.get("XFER_DUMP", "0") == "1"

MODEL = sys.argv[1]
N_AUV = int(sys.argv[2]) if len(sys.argv) > 2 else 4
D_FIXED = float(sys.argv[3]) if len(sys.argv) > 3 else 200.0
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
R_TRAIN = float(os.environ.get("XFER_R_TRAIN", "200"))

SPACING, PATCH_N, FORMATION, RING_RADIUS, BETA = 0.25, 1, "ring", 100.0, 0.8


def deploy_fixed(d, t0, n_auv, D, seed):
    """固定距离 D、方位随机（离岸拒绝采样）——随机数流与 sample_deployment 一致。"""
    from fleet_segment_env import land_interp
    lo = d["lon"]; la = d["lat"]
    elo = float(d["track_lon"][t0]); ela = float(d["track_lat"][t0])
    np.random.seed(seed)
    best = None
    cang = 0.0
    for _ in range(300):
        _ = np.random.uniform(0.0, 1.0) * D                # 与 sample_deployment 同序（丢弃）
        cang = np.random.uniform(0, 2 * np.pi)
        clon = elo + D * np.cos(cang) / (D2KM * np.cos(np.deg2rad(ela)))
        clat = ela + D * np.sin(cang) / D2KM
        lons, lats = deploy_formation(clon, clat, n_auv, FORMATION, 100.0, RING_RADIUS)
        lons = np.clip(lons, lo.min(), lo.max()); lats = np.clip(lats, la.min(), la.max())
        score = float(land_interp()(np.stack([lats, lons], axis=1)).min())
        if best is None or score > best[0]:
            best = (score, lons.copy(), lats.copy())
        if score > DEPLOY_MIN_COAST_KM:
            break
    clon_f = float(np.mean(best[1])); clat_f = float(np.mean(best[2]))
    centre_km = float(np.hypot((clon_f - elo) * D2KM * np.cos(np.deg2rad(ela)),
                               (clat_f - ela) * D2KM))
    return best[1], best[2], centre_km, float(best[0])


def run_segment(model, env, seg, idx, sd):
    s = seg["storm"]; d = env.data[s]
    env.d = d; env.t0 = seg_idx(seg["t0"]); env.t1 = seg_idx(seg["t1"]); env.t = env.t0
    # 种子只依赖 (段号, sd)，**与 D 无关** → 5 个 D 共用同一方位角，严格配对
    seed = 100000 + 97 * idx + sd
    env.vxs = np.zeros(env.n_auv); env.vys = np.zeros(env.n_auv)
    env.lons, env.lats, dep_centre_km, dep_coast_km = deploy_fixed(
        d, env.t0, env.n_auv, D_FIXED, seed)
    env.prev_actions = np.zeros((env.n_auv, 2), dtype=np.float32)
    obs = env._obs().reshape(-1).astype(np.float32)
    ld0 = env.dist_interp(np.stack([env.lats, env.lons], axis=1))
    init_land_frac = float((ld0 <= 0.0).mean())
    init_on_land = ld0 <= 0.0
    land_steps_init = 0; land_steps_late = 0

    a_exec = np.zeros((env.n_auv, 2), dtype=np.float32)
    land_steps = 0; buf_steps = 0; any_land_steps = 0; tot_steps = 0; touched = False
    path = np.zeros(env.n_auv); dt_h = STEP_MIN / 60.0
    dists = []
    while True:
        a, _ = model.predict(obs, deterministic=True)
        a = np.asarray(a, dtype=np.float32).reshape(env.n_auv, 2)
        a_exec = (1.0 - BETA) * a + BETA * a_exec
        a_use = np.clip(a_exec, -1.0, 1.0)
        nrm = np.linalg.norm(a_use, axis=1, keepdims=True)
        a_use = a_use / np.maximum(1.0, nrm)
        if SAFETY_MARGIN_KM > 0:
            a_use = safety_filter(a_use, env.lons, env.lats, env.dist_interp, SAFETY_MARGIN_KM)
        v = np.hypot(a_use[:, 0], a_use[:, 1]) * VMAX
        path += v * (STEP_MIN * 60) / 1000.0
        obs, _r, _t, trunc, info = env.step(a_use)
        obs = obs.reshape(-1).astype(np.float32)
        ld = info["land_dist"]
        onl = ld <= 0.0
        inb = (ld > 0.0) & (ld < R_BUFFER)
        land_steps += int(onl.sum()); buf_steps += int(inb.sum())
        land_steps_init += int((onl & init_on_land).sum())
        land_steps_late += int((onl & ~init_on_land).sum())
        any_land_steps += int(onl.any()); tot_steps += 1
        if onl.any():
            touched = True
        dists.append(float(np.mean(info["dist_km"])))
        if trunc:
            break
    n_auv_steps = tot_steps * env.n_auv
    return dict(
        land_frac=land_steps / max(n_auv_steps, 1),
        buf_frac=buf_steps / max(n_auv_steps, 1),
        any_land_step_frac=any_land_steps / max(tot_steps, 1),
        touched=float(touched),
        init_land_frac=init_land_frac,
        land_frac_init=land_steps_init / max(n_auv_steps, 1),
        land_frac_late=land_steps_late / max(n_auv_steps, 1),
        dep_centre_km=dep_centre_km, dep_coast_km=dep_coast_km,
        path=float(path.mean()),
        vmean=float(path.mean() * 1000.0 / (tot_steps * STEP_MIN * 60.0 + 1e-9)),
        dist=float(np.mean(dists)),
        final_dist=float(dists[-1]) if dists else float("nan"),
        T_h=float(tot_steps * STEP_MIN / 60.0))


def main():
    print("XFER 训练半径 R_train=%.0f km | 固定布防距离 D=%.0f km | 台数=%d | seeds=%d"
          % (R_TRAIN, D_FIXED, N_AUV, SEEDS), flush=True)
    model = PPO.load(MODEL, device="cpu")
    split = json.load(open(SPLIT)); storms = split[SPLIT_KEY]
    seg = pd.read_csv(SEGCSV); seg = seg[seg.storm.isin(storms)].reset_index(drop=True)
    lim = int(os.environ.get("XFER_LIMIT", "0"))
    if lim > 0:                     # 冒烟测试用：只跑前 lim 段
        seg = seg.iloc[:lim].reset_index(drop=True)
    os.environ["DEPLOY_FIXED_KM"] = str(D_FIXED)
    os.environ["STORAGE16"] = "1" if STORAGE16 else "0"
    env = FleetSegmentEnv(storms=storms, segments_df=seg, use_uv=True, n_auv=N_AUV,
                          smooth_penalty=0.05, spacing_deg=SPACING, patch_n=PATCH_N,
                          deploy_radius=R_TRAIN, formation=FORMATION, ring_radius=RING_RADIUS,
                          sst_mode=SST_MODE, use_mslp=USE_MSLP, vmax=VMAX)
    rows = []
    for idx, (_, row) in enumerate(seg.iterrows()):
        for sd in range(SEEDS):
            rows.append(run_segment(model, env, row, idx, sd))
            if XFER_DUMP:
                rows[-1]["idx"] = idx; rows[-1]["storm"] = row["storm"]
                rows[-1]["t0"] = float(seg_idx(row["t0"])); rows[-1]["sd"] = sd
        if (idx + 1) % 100 == 0:
            print("  ... %d/%d 段" % (idx + 1, len(seg)), flush=True)
    df = pd.DataFrame(rows)
    # 逐段落盘：用于「同一段、同一方位、不同权重/不同 D」的**配对**显著性检验
    # （聚合行只有均值，做不了配对检验；2026-10-13 加）
    if XFER_DUMP and "idx" in df.columns:
        tag = os.environ.get("XFER_TAG", "xfer")
        keep = ["idx", "storm", "t0", "sd", "dist", "path", "vmean", "T_h", "land_frac"]
        df[keep].to_csv(os.path.join(BASE, "logs_xfer", "%s_seg.csv" % tag), index=False)
    # 段级误差棒（768 个测试段 = 768 个不同的方位角实现，天然给出不确定度）
    n_seg = max(len(df), 1)
    sem_dist = float(df.dist.std(ddof=1) / np.sqrt(n_seg)) if n_seg > 1 else 0.0
    sem_path = float(df.path.std(ddof=1) / np.sqrt(n_seg)) if n_seg > 1 else 0.0
    sem_vmean = float(df.vmean.std(ddof=1) / np.sqrt(n_seg)) if n_seg > 1 else 0.0
    print(f"{os.path.basename(MODEL)}  n_auv={N_AUV} R_train={R_TRAIN:.0f} D={D_FIXED:.0f} "
          f"seeds={SEEDS} storms={len(storms)} rows={len(df)}")
    print(f"  追踪距离                 {df.dist.mean():6.1f} ± {sem_dist:.1f} km (SEM, n={n_seg})")
    print(f"  航迹长度(单台)           {df.path.mean():6.1f} ± {sem_path:.1f} km (SEM)")
    print(f"  平均航速                 {df.vmean.mean():6.2f} ± {sem_vmean:.2f} m/s (SEM)")
    print(f"  初始布防中心离眼         {df.dep_centre_km.mean():6.1f} km  (标称 D={D_FIXED:.0f})")
    print(f"  初始布防最小离岸         {df.dep_coast_km.mean():6.1f} km")
    print(f"  初始布防就在陆地占比     {df.init_land_frac.mean()*100:6.2f} %")
    print(f"  上岸占比(AUV-step)       {df.land_frac.mean()*100:6.2f} %")
    print(f"  缓冲带0-100km占比        {df.buf_frac.mean()*100:6.2f} %")
    print(f"  触陆回合占比             {df.touched.mean()*100:6.2f} %")
    print(f"    其中「初始就在陆地」贡献 {df.land_frac_init.mean()*100:6.2f} %")
    print(f"    其中「初始在海上后上陆」贡献 {df.land_frac_late.mean()*100:6.2f} %")
    print(f"  平均航行时间             {df.T_h.mean():6.2f} h")
    # R|D|N|dist|dist_sem|path|path_sem|vmean|vmean_sem|T_h|land_frac|touched|dep_centre|dep_coast|init_land|rows
    print("ROW|%.0f|%.0f|%d|%.1f|%.2f|%.1f|%.2f|%.2f|%.3f|%.2f|%.4f|%.3f|%.1f|%.1f|%.2f|%.0f"
          % (R_TRAIN, D_FIXED, N_AUV, df.dist.mean(), sem_dist, df.path.mean(), sem_path,
             df.vmean.mean(), sem_vmean, df.T_h.mean(), df.land_frac.mean(), df.touched.mean(),
             df.dep_centre_km.mean(), df.dep_coast_km.mean(), df.init_land_frac.mean(), len(df)),
          flush=True)
    final_mean = float(df.final_dist.mean())
    final_sem = (
        float(df.final_dist.std(ddof=1) / np.sqrt(n_seg))
        if n_seg > 1 else 0.0
    )
    print(
        "FINALROW|%.0f|%.0f|%d|%.1f|%.2f|%.0f"
        % (R_TRAIN, D_FIXED, N_AUV, final_mean, final_sem, len(df)),
        flush=True,
    )


if __name__ == "__main__":
    main()
