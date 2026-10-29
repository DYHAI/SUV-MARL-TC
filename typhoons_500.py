"""把台风数据集从 200 扩到 ~500：**再补 300 个**（2004-07 → 2018-03），格式与之前完全一致。

口径（与 200 那批一致）：
  · bbox lon 98.7–171.6 / lat 6.8–56.7
  · IBTrACS NATURE ∈ {TS, MX}（热带气旋）
  · 不设风速阈值
新增的唯一变化 = **时间往前扩**：因为 2019–2025 的全部热带系统只有 200 个，要凑 300 个必须回溯。

HYCOM 源（都是 GOFS 3.1 家族、1/12°、3 小时、41 层，格式与现有 200 批一致）按时间接力：
  · 2004-07 → 2015-12 : GLBv0.08/expt_53.X/data/{year}
  · 2015-12 → 2016-09 : GLBv0.08/expt_56.3
  · 2016-09 → 2017-02 : GLBv0.08/expt_57.2
  · 2017-02 → 2017-05 : GLBv0.08/expt_92.8
  · 2017-05 → 2017-07 : GLBv0.08/expt_57.7
  · 2017-07 → 2018-03 : GLBv0.08/expt_92.9
  · ❌ 2018-04 → 2018-12-03 : **真缺口，无任何源**（落在这里的台风必须剔除）
  · 2018-12-04 → 2024-09 : GLBy0.08/expt_93.0（现有 200 批用的）
  · 2024-08-10 → 今     : ESPC-D-V02（现有）
边界是月级探测定出来的，比较模糊，所以下载器**按顺序逐个试**、拿到非空数据就用，
不依赖精确边界（见 hycom_candidates()）。

用法：其它下载脚本把 `import typhoons_200 as T` 改成 `import typhoons_500 as T` 即可。
"""
import json, os, re
import pandas as pd

import typhoons_2020plus as T

T.MIN_WIND = 0
T.MIN_SEASON = 2004
T.MAX_SEASON = 2018

TC_NATURE = {"TS", "MX"}
COVER_LO = pd.Timestamp("2004-07-01 00:00:00")
GAP0 = pd.Timestamp("2018-04-01 00:00:00")
GAP1 = pd.Timestamp("2018-12-03 00:00:00")
N_NEW = 300          # 目标新增数（总数 200 + 300 = 500）
PAD_H = 30           # 数据窗口 = track ±(6h,24h)，判定可用性时留 30h 余量

LON0, LON1, LAT0, LAT1 = T.LON0, T.LON1, T.LAT0, T.LAT1
ROOT = T.ROOT                       # 项目根目录（scan_data_500.py 等要用）
SEL = T.SEL                         # data/selected_typhoons
EXPT93_END = T.EXPT93_END
ensure_track_meta_impl = T.ensure_track_meta
window = T.window
sanitize = T.sanitize


def _read_ibtracs():
    return T._read_ibtracs()


def _usable(tmin, tmax):
    a = tmin - pd.Timedelta(hours=PAD_H)
    b = tmax + pd.Timedelta(hours=PAD_H)
    if a < COVER_LO:
        return False
    if (a < GAP1) and (b > GAP0):        # 与 2018 年缺口相交
        return False
    return True


def build_storm_list(n_new=N_NEW):
    """返回**新增**的 n_new 个台风（取可用的、时间上最近的 n_new 个）。"""
    ib = _read_ibtracs()
    nat = {sid: set(g["NATURE"].dropna()) for sid, g in ib.groupby("SID")}
    inb = (ib.LON >= LON0) & (ib.LON <= LON1) & (ib.LAT >= LAT0) & (ib.LAT <= LAT1)
    sub = ib[inb].copy()
    g = sub.groupby("SID").agg(name=("NAME", "first"), season=("SEASON", "first"),
                               tmin=("t", "min"), tmax=("t", "max"), nbox=("LAT", "count"))
    g = g[[sid in nat and bool(nat[sid] & TC_NATURE) for sid in g.index]]
    g = g[[_usable(r.tmin, r.tmax) for r in g.itertuples()]]
    # 排除已有的 200 批（2019-2025 + 2018-12）
    old = g[(g.season >= 2019) | ((g.season == 2018) & (g.tmin >= pd.Timestamp("2018-12-04")))]
    cand = g.drop(index=old.index).sort_values("tmin")
    cand = cand.iloc[-n_new:]                      # 取最近的 n_new 个
    storms = []
    for sid, r in cand.iterrows():
        storms.append({
            "name": sanitize(r["name"]), "raw_name": str(r["name"]),
            "season": int(r["season"]), "sid": sid, "maxwind": 0.0,
            "tmin": pd.Timestamp(r["tmin"]), "tmax": pd.Timestamp(r["tmax"]),
        })
    return storms


# ---------------------------------------------------------------- 唯一命名（沿用 200 批逻辑）
_UNIQUE = None


def _build_tags():
    global _UNIQUE
    if _UNIQUE is not None:
        return
    all_storms = T.build_storm_list() + build_storm_list(10**6)   # 全量（含已有）
    counts = {}
    for s in all_storms:
        b = f"{s['name']}_{s['season']}"
        counts[b] = counts.get(b, 0) + 1
    _UNIQUE = {}
    for s in all_storms:
        b = f"{s['name']}_{s['season']}"
        _UNIQUE[s["sid"]] = b if counts[b] == 1 else f"{b}_{str(s['sid'])[:7]}"


def tag_of(storm):
    _build_tags()
    return _UNIQUE[storm["sid"]]


# ---------------------------------------------------------------- 落盘到第二块 NVMe
STORE_ROOT = os.environ.get(
    "MARL_STORE_ROOT",
    os.path.join(T.ROOT, "data", "typhoon500"),
)
_KIND2LINK = {
    "union": "data/union_region",
    "era5_wind": "data/era5_wind",
    "era5_sst": "data/era5_sst",
    "era5_mslp": "data/era5_mslp",
}


def real_dir(kind, tag):
    """真实目录建在第二块 NVMe，data/<kind>/<tag> 用 symlink 指过去（与 200 批做法一致）。
    返回**真实目录**供写入；若该 tag 已经是真实目录（老台风）就直接用。"""
    real = os.path.join(STORE_ROOT, kind, tag)
    link = os.path.join(T.ROOT, _KIND2LINK[kind], tag)
    os.makedirs(real, exist_ok=True)
    if os.path.islink(link):
        return real
    if os.path.isdir(link):
        return link
    os.makedirs(os.path.dirname(link), exist_ok=True)
    os.symlink(real, link)
    return real


def ensure_track_meta(storm):
    folder = os.path.join(T.SEL, tag_of(storm))
    os.makedirs(folder, exist_ok=True)
    track = os.path.join(folder, "track.csv")
    if not os.path.exists(track):
        ib = _read_ibtracs()
        ib[ib.SID == storm["sid"]].sort_values("t").to_csv(track, index=False)
    t0, t1 = window(storm)
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump({"name": storm["raw_name"], "season": storm["season"], "sid": storm["sid"],
                   "tag": tag_of(storm), "window": [str(t0), str(t1)],
                   "bbox": [LON0, LAT0, LON1, LAT1],
                   "source": "HYCOM GOFS3.1 multi-experiment chain (pre-2018) / expt_93.0 / ESPC"}, f)
    return folder


# ---------------------------------------------------------------- HYCOM 多源候选
# (标签, urlPath, 是否需要 /data/{year} 后缀, 起, 止)
_CHAIN = [
    ("expt_53.X", "GLBv0.08/expt_53.X/data/{year}", True, "2004-07-01", "2016-02-01"),
    ("expt_56.3", "GLBv0.08/expt_56.3", False, "2015-11-01", "2016-10-01"),
    ("expt_57.2", "GLBv0.08/expt_57.2", False, "2016-09-01", "2017-03-01"),
    ("expt_92.8", "GLBv0.08/expt_92.8", False, "2017-02-01", "2017-06-01"),
    # ⚠️ 2026-10-09 修正：expt_57.7 的**实际上界**远不止 2017-08-01。
    #    原先写 2017-08-01，导致 2017-08-10~2017-09-26 的 12 个台风（NORU/HAITANG/
    #    NALGAE/BANYAN/HATO/PAKHAR/SANVU/MAWAR/GUCHOL/TALIM/DOKSURI/UNNAMED）
    #    在候选链里只剩 expt_92.9，而 92.9 对该时段一律 HTTP 400 → 全部下不下来。
    #    实测（probe_hycom_sources.py）57.7 在 2017-06-09~2017-10-08 都有数据
    #    （2017-10-08 起由 expt_92.9 接续），所以上界改到 2017-10-15。
    ("expt_57.7", "GLBv0.08/expt_57.7", False, "2017-05-01", "2017-10-15"),
    ("expt_92.9", "GLBv0.08/expt_92.9", False, "2017-07-01", "2018-04-01"),
    ("expt_93.0", "GLBy0.08/expt_93.0/{agg}/{year}", True, "2018-12-04", str(EXPT93_END)[:10]),
    ("espc", "ESPC-D-V02/{agg}/{year}", True, "2024-08-01", "2030-01-01"),
]


def hycom_candidates(dt, agg="uv3z"):
    """给定时刻，返回**按顺序尝试**的 (标签, urlPath) 列表（时间上可能重叠，故返回多个）。

    agg: uv3z(uv) / ts3z(temp) / u3z / v3z / t3z —— 由下载器决定。
    """
    dt = pd.Timestamp(dt)
    out = []
    for name, path, needs_year, lo, hi in _CHAIN:
        if pd.Timestamp(lo) <= dt <= pd.Timestamp(hi):
            p = path.format(year=dt.year, agg=agg) if needs_year else path
            out.append((name, p))
    return out


# ---------------------------------------------------------------- ERA5 请求区域：紧 bbox
# 背景（2026-10-09）：ERA5 三个下载脚本原来对**每个台风都请求整个并集 bbox**
#   AREA = [56.7, 98.7, 6.8, 171.6]（73°×50°）。一个只占 8°×6° 的台风也要下 50–120 MB，
#   CDS 端子集化又慢，导致 900 个文件要几十小时（实测 ~14 文件/小时）。
#
# 收紧到「轨迹 bbox + margin」后数据量降一个量级，而**取值完全一样**：
#   ERA5 是规则 0.25° 网格，子集不改变格点值；训练侧 `load_storm(crop_margin_km=deploy+200=400)`
#   本来也只保留 track bbox ±400 km，所以两张图的**有效区域完全相同**，不存在新旧数据分布差异。
# 边界向外取整到 0.25°，并夹在并集 bbox 内（与已下载文件同域，绝不超出）。
def track_bbox(tag):
    tr = pd.read_csv(os.path.join(SEL, tag, "track.csv"),
                     parse_dates=["ISO_TIME"]).dropna(subset=["LAT", "LON"])
    return float(tr.LON.min()), float(tr.LON.max()), float(tr.LAT.min()), float(tr.LAT.max())


def tight_area(tag, margin_km=450.0, grid=0.25):
    """返回 CDS 的 [north, west, south, east]（向外取整到 0.25°，且不超出并集 bbox）。"""
    import math
    lo0, lo1, la0, la1 = track_bbox(tag)
    mid = 0.5 * (la0 + la1)
    dlon = margin_km / (111.0 * math.cos(math.radians(mid)))
    dlat = margin_km / 111.0
    W = math.floor((lo0 - dlon) / grid) * grid
    E = math.ceil((lo1 + dlon) / grid) * grid
    S = math.floor((la0 - dlat) / grid) * grid
    N = math.ceil((la1 + dlat) / grid) * grid
    W = max(W, float(LON0)); E = min(E, float(LON1))
    S = max(S, float(LAT0)); N = min(N, float(LAT1))
    return [round(N, 2), round(W, 2), round(S, 2), round(E, 2)]


AREA_MODE = os.environ.get("AREA_MODE", "union")   # union(默认,兼容) | tight


def area_for(storm):
    """按 AREA_MODE 给出该台风的请求区域。"""
    if AREA_MODE != "tight":
        return [float(LAT1), float(LON0), float(LAT0), float(LON1)]
    return tight_area(tag_of(storm))


if __name__ == "__main__":
    s = build_storm_list()
    print(f"新增台风: {len(s)} 个（{s[0]['season']} – {s[-1]['season']}）")
    from collections import Counter
    print("按年:", dict(sorted(Counter(x["season"] for x in s).items())))
    print("前 3 个 tag:", [tag_of(x) for x in s[:3]])
