"""500 台风阶段的**划分**：450 训练 / 50 测试（按台风分层抽样，不按段）。

与 200 阶段的口径一致：
  · 台风全集 = 已有 200 个（2019–2025 + 2018-12）+ 新增 300 个（2007–2018）
  · 只保留**数据可用**的台风（HYCOM 4 类 + ERA5 3 类都齐）
  · 测试集按**年份分层**抽 50 个，训练集 = 其余 450 个
  · 输出 data/typhoon_split_500.json（沿用 200 阶段的 JSON 结构）

用法: python typhoons_500_split.py            # 只看可用数 + 生成划分
"""
import json, os
from collections import Counter

import numpy as np

import typhoons_200 as T2
import typhoons_500 as T5
import typhoons_2020plus as T0

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "typhoon_split_500.json")
# ⚠️ 数据没齐时**只写 _partial**，避免半成品划分被下游（launch_stage500.sh / 评测）当成终版。
OUT_PARTIAL = os.path.join(ROOT, "data", "typhoon_split_500_partial.json")
N_TEST = 50
SEED = 0

# 「数据齐全」的口径（2026-10-09 加）：
#   READY_MODE=era5（默认）→ 只要 ERA5 三件套（wind/sst/mslp）+ track 就参与 500 阶段切段/划分。
#       为什么：500 阶段的奖励/观测只用到 ERA5（风 + SST距平 + MSLP），**根本不读 HYCOM**；
#       而 HYCOM（uv/t 各深度）与 ERA5 是两套独立下载队列，HYCOM 服务器吞吐封顶 ~40 文件/小时，
#       等它齐要多花十几小时，纯属白等。
#   READY_MODE=all → 严格口径（HYCOM 4 类 + ERA5 3 类全齐），给**流场类任务**用。
READY_MODE = os.environ.get("READY_MODE", "era5")


def _list200():
    """⚠️ 必须先恢复 typhoons_200 的年段参数！

    `typhoons_200` 与 `typhoons_500` 都在 import 时改**同一个底层模块** `typhoons_2020plus`
    的全局 `MIN_SEASON/MAX_SEASON`，同时 import 会让后者覆盖前者：
    实测 import typhoons_500 之后，`T2.build_storm_list()` 从 200 个塌成 **2 个**
    → 500 台风的并集被算成 302。所以调用前显式把年段设回各自的原值。
    """
    T0.MIN_WIND, T0.MIN_SEASON, T0.MAX_SEASON = 0, 2018, 2025
    return T2.build_storm_list()


def _list500new():
    T0.MIN_WIND, T0.MIN_SEASON, T0.MAX_SEASON = 0, 2004, 2018
    return T5.build_storm_list()


def all_tags():
    """已有 200 + 新增 300（去重），同时给出 tag → season 映射。"""
    m = {}
    for s in _list200():
        m[T2.tag_of(s)] = int(s["season"])
    for s in _list500new():
        m[T5.tag_of(s)] = int(s["season"])
    return sorted(m), m


def data_ready(tag):
    """7 个文件是否齐（HYCOM 0/50m uv + 0/50m temp + ERA5 wind/sst/mslp）。"""
    need = [("union_region", "hycom_uv_0m.nc"), ("union_region", "hycom_uv_50m.nc"),
            ("union_region", "hycom_temp_0m.nc"), ("union_region", "hycom_temp_50m.nc"),
            ("era5_wind", "era5_10m_wind_hourly.nc"), ("era5_sst", "era5_sst_hourly.nc"),
            ("era5_mslp", "era5_mslp_hourly.nc")]
    return all(os.path.exists(os.path.join(ROOT, "data", d, tag, f)) for d, f in need)


def data_ready_era5(tag):
    """500 阶段追踪任务实际需要的三个 ERA5 文件（+ track.csv）。"""
    need = [("era5_wind", "era5_10m_wind_hourly.nc"), ("era5_sst", "era5_sst_hourly.nc"),
            ("era5_mslp", "era5_mslp_hourly.nc")]
    if not all(os.path.exists(os.path.join(ROOT, "data", d, tag, f)) for d, f in need):
        return False
    return os.path.exists(os.path.join(ROOT, "data", "selected_typhoons", tag, "track.csv"))


def ready(tag):
    """按 READY_MODE 判定该台风能否进入本阶段。"""
    return data_ready_era5(tag) if READY_MODE == "era5" else data_ready(tag)


def make_split(tags, seasons, n_test=N_TEST, seed=SEED):
    """按年份分层抽 n_test 个测试台风。"""
    rng = np.random.default_rng(seed)
    # 数据不足时直接按比例缩水，避免 rest 为空
    n_test = min(n_test, max(1, len(tags) // 10))
    test = []
    for y in sorted(set(seasons.values())):
        pool = sorted(t for t in tags if seasons[t] == y)
        k = int(round(n_test * len(pool) / len(tags)))
        if k <= 0:
            continue
        test += list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    test = sorted(set(test))
    rest = sorted(set(tags) - set(test))
    while len(test) > n_test:
        test.remove(rng.choice(test))
    while len(test) < n_test:
        p = rng.choice(rest); test.append(p); rest.remove(p)
    test = sorted(test)
    return sorted(set(tags) - set(test)), test


def main():
    tags, season_map = all_tags()
    ready_tags = [t for t in tags if ready(t)]
    print(f"台风全集 {len(tags)}（200 已有 + 300 新增，去重后）")
    print(f"数据齐全 {len(ready_tags)}  →  还缺 {len(tags)-len(ready_tags)}   （READY_MODE={READY_MODE}）")
    if len(ready_tags) < 500:
        print("⚠️ 数据还没齐，先不生成终版划分（可反复跑，齐了再生成）")
    # ⚠️ 原始版本这里写的是 `season_of(t)`，但本文件**没有** season_of → main() 一跑就 NameError。
    #    台风的年份映射就是 all_tags() 返回的 season_map，直接用即可。
    seasons = {t: season_map[t] for t in ready_tags}
    train, test = make_split(ready_tags, seasons)
    complete = (len(ready_tags) == len(tags))
    out_path = OUT if complete else OUT_PARTIAL
    json.dump({"train": train, "test": test, "split_by": "storm",
               "rule": f"season-stratified, seed={SEED}, n_test={N_TEST}",
               "n_storms_total": len(ready_tags),
               "note": f"500 台风阶段；READY_MODE={READY_MODE}；参与划分的台风数据齐全"},
              open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"划分 -> {out_path}: 训练 {len(train)} / 测试 {len(test)}"
          + ("" if complete else "  ← 半成品（还缺台风，不是 500 终版）"))
    print("  训练集年份:", dict(sorted(Counter(seasons[t] for t in train).items())))
    print("  测试集年份:", dict(sorted(Counter(seasons[t] for t in test).items())))


if __name__ == "__main__":
    main()
