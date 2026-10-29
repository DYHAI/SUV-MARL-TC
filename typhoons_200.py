# -*- coding: utf-8 -*-
"""扩展到 ~200 个台风：**2019–2025 全部热带系统**（不设强度阈值），bbox 与数据格式与原有 64 个完全一致。

为什么是"2019–2025 全部系统"：
  · 要凑到 200 个，在**不换 HYCOM 数据源**的前提下，唯一可行的取法是放宽强度阈值
    （2020–2025 且 ≥64kt 只有 64 个；≥34kt 也只有 140 个；2018–2025 且 ≥34kt 有 198 个，
     但其中 29 个早于 2018-12-04，超出 GLBy0.08/expt_93.0 的覆盖起点）。
  · 2019-01-01 之后全部落在现有两个 HYCOM 源（expt_93.0 / ESPC-D-V02）覆盖内，
    下载脚本无需改 URL/网格，产物格式与原来 64 个逐字段一致。
  · 结果为 **201 个**台风，且**包含**原来那 64 个 → 已有的 64 份数据可直接复用，只补下新的。

用法：其它下载脚本把 `import typhoons_2020plus as T` 换成 `import typhoons_200 as T` 即可。
"""
import json
import os
import re
import pandas as pd

import typhoons_2020plus as T  # noqa: E402

# 覆盖模块级常量（build_storm_list 在调用时读取这些全局量）
T.MIN_WIND = 0        # 不设强度阈值
T.MIN_SEASON = 2018   # 放宽到 2018（下面再用 HYCOM 覆盖起点 2018-12-04 卡）
T.MAX_SEASON = 2025

# ---------------------------------------------------------------- 热带气旋筛选
# IBTrACS 的 NATURE 字段：TS=热带气旋、SS=副热带、ET=温带、MX=混合（含热带段的过渡型）、
# NR/DS/WV/LO=扰动类。
# 「热带气旋」判定 = 至少有一段被标记为 **TS 或 MX**：
#   · 2019-2025 共 197 个纯热带气旋（TS）
#   · + 2018-12-04（HYCOM expt_93.0 覆盖起点）之后当年 2 个 → 199
#   · + NEPARTAK_2021（NATURE=MX，具名，峰值 40 kt，72 个时次）→ **200 个**
# 被排除的是 3 个非热带系统：HAITANG_2022(SS 副热带, 无 WMO 风速)、
# 2025 年 2 个 UNNAMED(NR 未报告, 仅 5/9 个时次)。
TC_START = "2018-12-04"
TC_NATURE = {"TS", "MX"}
_BASE_BUILD = T.build_storm_list


def build_storm_list():
    storms = _BASE_BUILD()
    ib = T._read_ibtracs()
    nat = {sid: set(g["NATURE"].dropna()) for sid, g in ib.groupby("SID")}
    keep = []
    for s in storms:
        if not (nat.get(s["sid"], set()) & TC_NATURE):    # 非热带气旋（副热带/扰动/温带）
            continue
        if pd.Timestamp(s["tmin"]) < pd.Timestamp(TC_START):   # HYCOM 覆盖起点之前
            continue
        keep.append(s)
    return keep

ensure_track_meta = T.ensure_track_meta
hycom_source = T.hycom_source
window = T.window
snap3 = T.snap3
sanitize = T.sanitize
LON0, LON1, LAT0, LAT1 = T.LON0, T.LON1, T.LAT0, T.LAT1
EXPT93_END = T.EXPT93_END
SELECTED = T.SEL


def existing(name, season, root):
    """该台风是否已有某个数据文件（用于跳过已下载）。"""
    import os
    return os.path.exists(os.path.join(root, f"{name}_{season}"))


# ---------------------------------------------------------------- 唯一命名
# 全系统清单里有 8 组重名（多个未命名系统都叫 UNNAMED_YYYY，另有 HAITANG_2022 两次），
# 直接拼 {name}_{season} 会覆盖。这里统一加 SID 后缀消歧；**原 64 个清单无重名**，
# 所以原有目录名保持不变、数据完全复用。
_BASE = None
_UNIQUE = None


def _build_tags():
    global _BASE, _UNIQUE
    if _UNIQUE is not None:
        return
    storms = T.build_storm_list()
    counts = {}
    for s in storms:
        counts[f"{s['name']}_{s['season']}"] = counts.get(f"{s['name']}_{s['season']}", 0) + 1
    _BASE = {}
    _UNIQUE = {}
    for s in storms:
        base = f"{s['name']}_{s['season']}"
        tag = base if counts[base] == 1 else f"{base}_{str(s['sid'])[:7]}"
        _BASE[s["sid"]] = base
        _UNIQUE[s["sid"]] = tag


def tag_of(storm):
    """该台风的唯一目录名（重名时带 SID 后缀）。"""
    _build_tags()
    return _UNIQUE[storm["sid"]]


def storage_root(storm):
    """数据放哪块盘：原有目录（LVM）优先；全新台风放 NVMe（424 GB 空闲那块）。

    返回 (union_root, era5_root, nvme_target) —— nvme_target 非 None 时需要建 symlink。
    """
    sel = T.SEL
    idx = os.path.join(sel, "_storage_map.json")
    m = json.load(open(idx)) if os.path.exists(idx) else {}
    t = tag_of(storm)
    if t in m:
        return m[t]["union"], m[t]["era5"], m[t].get("target")
    return None, None, None


def ensure_track_meta(storm):  # noqa: F811  —— 用唯一 tag 覆盖原实现
    folder = os.path.join(T.SEL, tag_of(storm))
    os.makedirs(folder, exist_ok=True)
    track = os.path.join(folder, "track.csv")
    if not os.path.exists(track):
        ib = T._read_ibtracs()
        g = ib[ib.SID == storm["sid"]].sort_values("t")
        g.to_csv(track, index=False)
    t0, t1 = T.window(storm)
    src = ("HYCOM GLBy0.08/expt_93.0 uv3z" if t1 <= T.EXPT93_END else "HYCOM ESPC-D-V02 u3z/v3z")
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump({"name": storm["raw_name"], "season": storm["season"], "sid": storm["sid"],
                   "tag": tag_of(storm), "window": [str(t0), str(t1)],
                   "bbox": [T.LON0, T.LAT0, T.LON1, T.LAT1], "source": src}, f)
    return folder


def main():
    storms = build_storm_list()
    print("2019-2025 全部热带系统（bbox 内）共 %d 个台风" % len(storms))
    from collections import Counter
    print("  按年份:", dict(sorted(Counter(s["season"] for s in storms).items())))


if __name__ == "__main__":
    main()
