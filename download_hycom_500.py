"""下载新增 300 个台风（2007–2018）的 HYCOM，格式与现有 200 批**完全一致**：
  data/union_region/<tag>/hycom_uv_{0,50}m.nc      (water_u, water_v, 3-hourly)
  data/union_region/<tag>/hycom_temp_{0,50}m.nc    (water_temp,    3-hourly)

与 download_hycom_200.py 的区别只有一个：**源按时间接力**（200 批只有 expt_93.0 / ESPC 两个源）。
每个 48h 分块会按 hycom_candidates() 给出的候选源**逐个试**，拿到非空就继续 ——
所以对 2016/2017 那几个实验的交接边界不敏感（边界是月级探测定出来的）。

存储：新台风的真实目录由 MARL_STORE_ROOT 指定（默认 data/typhoon500），
并用 symlink 挂到 data/union_region/<tag>，与 200 批的做法一致。

用法: python download_hycom_500.py <worker> <nworkers> <product:uv0|uv50|t0|t50>
"""
import json, os, subprocess, sys, time
import pandas as pd
import xarray as xr

import typhoons_500 as T

ROOT = os.path.dirname(os.path.abspath(__file__))
LINKROOT = os.path.join(ROOT, "data", "union_region")      # 训练代码读这里
STOREROOT = os.environ.get(
    "HYCOM_STORE_ROOT",
    os.path.join(ROOT, "data", "typhoon500", "union"),
)
# ⚠️ 分块大小实测（expt_53.X / 全 union bbox / 单个请求）：
#    48h → 296 s / 16.9 MB (57 KB/s)；72h → 260 s / 25.3 MB (97 KB/s)；
#    120h → 318 s / 42.1 MB (132 KB/s) ← 最快；7 天(168h) 会让服务器 5 分钟后断。
#    所以瓶颈是**每请求的固定开销**，分块越大越快。默认 120h，可用 CHUNK_H 覆盖。
#    96h 那次是瞬时网络错误（rc=56），有重试兜底。
CHUNK_H = int(os.environ.get("CHUNK_H", "120"))
# 自适应回退下限：2009–2017 那批（GLBv0.08/expt_53.X）里有一部分台风，
# 服务器对 120h 大分块会直接断流（curl 56 / HTTP 400），但**同一时段拆成 48h / 24h 就能过**
# （实测 CHOI-WAN_2009：120h → curl 000 失败；48h → 200 / 17.9 MB；24h → 200 / 9.5 MB）。
# 所以分块级失败时上层 _fetch_span 会自动二分，直到这里设的下限。
MIN_CHUNK_H = int(os.environ.get("MIN_CHUNK_H", "24"))
NCSS = "https://ncss.hycom.org/thredds/ncss/grid/"

PRODUCTS = {
    "uv0":  dict(depth=0,  agg="uv3z", varnames=("water_u", "water_v"), espc=("u3z", "v3z"),
                 out="hycom_uv_0m.nc"),
    "uv50": dict(depth=50, agg="uv3z", varnames=("water_u", "water_v"), espc=("u3z", "v3z"),
                 out="hycom_uv_50m.nc"),
    "t0":   dict(depth=0,  agg="ts3z", varnames=("water_temp",),        espc=("t3z",),
                 out="hycom_temp_0m.nc"),
    "t50":  dict(depth=50, agg="ts3z", varnames=("water_temp",),        espc=("t3z",),
                 out="hycom_temp_50m.nc"),
    # 30m 流速（用户 2026-10-09 追加需求：500 个台风都要）
    "uv30": dict(depth=30, agg="uv3z", varnames=("water_u", "water_v"), espc=("u3z", "v3z"),
                 out="hycom_uv_30m.nc"),
    # Additional current-only depths requested for the 500-storm archive.
    "uv10": dict(depth=10, agg="uv3z", varnames=("water_u", "water_v"), espc=("u3z", "v3z"),
                 out="hycom_uv_10m.nc"),
    "uv20": dict(depth=20, agg="uv3z", varnames=("water_u", "water_v"), espc=("u3z", "v3z"),
                 out="hycom_uv_20m.nc"),
    "uv40": dict(depth=40, agg="uv3z", varnames=("water_u", "water_v"), espc=("u3z", "v3z"),
                 out="hycom_uv_40m.nc"),
    "uv60": dict(depth=60, agg="uv3z", varnames=("water_u", "water_v"), espc=("u3z", "v3z"),
                 out="hycom_uv_60m.nc"),
}


def fetch(url, path):
    # ⚠️ 实测：单个 48h 分块要 4–8 分钟；7 天的长请求服务器 5 分钟后直接断，所以分块不能放大。
    #    重试次数也不能多，否则一个坏分块能耗掉半小时。
    #    rc=18/22/28/52/56（半截/HTTP400/超时/空回复/被掐断）说明这个**分块尺寸**服务器不接受，
    #    立刻返回 False 交给上层二分（否则 3 次重试 × 900s 会把 worker 卡死半小时）。
    for attempt in range(3):
        try:
            subprocess.run(["curl", "-sSL", "-f", "--retry", "2", "--retry-delay", "6",
                            "--retry-all-errors", "--max-time", "900", "-o", path, url], check=True)
            return True
        except subprocess.CalledProcessError as e:
            if e.returncode in (18, 22, 28, 52, 56):
                return False          # 分块太大 / 被服务器掐断 → 交给上层二分
            time.sleep(8 * (attempt + 1))
    return False


def time_count(path):
    try:
        with xr.open_dataset(path) as d:
            return int(len(d.time))
    except Exception:
        return -1


def try_claim(folder, outname, stale_s=1800):
    """动态抢单：给 (台风, 产品) 加锁，成功返回 True。

    为什么需要：静态分片（i % N）下，先跑完自己分片的 worker 只能干等，
    尾巴阶段并发会塌到个位数（实测 24 workers 里有 14 个已跑完、只剩 10 个在收尾）。
    动态抢单让空闲 worker 立刻接管别人的剩余台风。锁文件超过 stale_s 视为陈旧可抢。
    """
    lock = os.path.join(folder, f".lock_{outname}")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(lock) > stale_s:
                os.remove(lock)
        except FileNotFoundError:
            pass
        return False


def _url(path, varnames, depth, a):
    q = "".join(f"&var={v}" for v in varnames)
    return (f"{NCSS}{path}?{q[1:]}&vertCoord={depth}&accept=netcdf4"
            f"&north={T.LAT1}&west={T.LON0}&east={T.LON1}&south={T.LAT0}"
            f"&time_start={a['t0']}&time_end={a['t1']}")


def _span_hours(cs, ce):
    return (ce - cs).total_seconds() / 3600.0


def _fetch_span(cs, ce, spec, folder):
    """抓取 [cs, ce) 一段原始数据，返回 xr.Dataset 列表；彻底失败返回 None。

    先试 CHUNK_H 的大分块（吞吐最高）；所有候选源都失败时**二分回退**到更小的分块
    （保持 3h 的整数倍，与 HYCOM 3 小时输出对齐），直到 MIN_CHUNK_H 为止。

    ⚠️ 临时文件名必须**带产品标签**（tmp_<product>_<起始时刻>.nc）：uv0 与 uv50 是同一个
    台风、同一个分块时刻，若共用 tmp_<ctag>.nc 就会互相覆盖（两个产品并发时必炸）。
    """
    a = dict(t0=cs.strftime("%Y-%m-%dT%H:%M:%SZ"), t1=ce.strftime("%Y-%m-%dT%H:%M:%SZ"))
    ctag = cs.strftime("%Y%m%dT%H")
    ptag = spec["out"].replace(".nc", "")          # hycom_uv_0m / hycom_uv_50m / ...
    cp = os.path.join(folder, f"tmp_{ptag}_{ctag}.nc")
    span_h = _span_hours(cs, ce)
    _expect = int(span_h // 3) + 1
    # 断点续跑：分块边界是确定的，tmp_<起始时刻> 可直接复用（要求步数够，避免半截文件）
    if os.path.exists(cp) and time_count(cp) >= int(_expect * 0.9):
        return [xr.open_dataset(cp)]
    for label, path in T.hycom_candidates(cs, agg=spec["agg"]):
        if label == "espc":
            merged, ok = [], True
            for agg in spec["espc"]:
                p = os.path.join(folder, f"tmp_{ptag}_{agg}_{ctag}.nc")
                if not fetch(_url(f"ESPC-D-V02/{agg}/{cs.year}", spec["varnames"], spec["depth"], a), p) \
                   or time_count(p) < 1:
                    ok = False
                    if os.path.exists(p):
                        os.remove(p)
                    break
                merged.append(xr.open_dataset(p))
            if not ok:
                continue
            return [xr.merge(merged)]
        if not fetch(_url(path, spec["varnames"], spec["depth"], a), cp) or time_count(cp) < 1:
            if os.path.exists(cp):
                os.remove(cp)
            continue                      # 换下一个候选源
        return [xr.open_dataset(cp)]
    # 所有候选源在**这个分块尺寸**下都失败 → 二分（步长保持 3h 整数倍）
    n_steps = int(round(span_h / 3.0))
    if span_h > MIN_CHUNK_H and n_steps >= 2:
        mid = cs + pd.Timedelta(hours=3 * (n_steps // 2))
        left = _fetch_span(cs, mid, spec, folder)
        if left is not None:
            right = _fetch_span(mid, ce, spec, folder)
            if right is not None:
                print(f"    [bisect] {ctag}: {span_h:.0f}h 失败 → 拆成 "
                      f"{_span_hours(cs, mid):.0f}h + {_span_hours(mid, ce):.0f}h 成功", flush=True)
                return left + right
    return None


def ensure_dir(tag):
    """实际目录建在 NVMe，data/union_region/<tag> 用 symlink 指过去（与 200 批一致）。"""
    override = os.environ.get("HYCOM_DOWNLOAD_ROOT")
    if override:
        folder = os.path.join(override, tag)
        os.makedirs(folder, exist_ok=True)
        return folder
    return T.real_dir("union", tag)


def download(storm, spec):
    tag = T.tag_of(storm)
    folder = ensure_dir(tag)
    out = os.path.join(folder, spec["out"])
    if os.path.exists(out) and time_count(out) > 0:
        return f"{tag}: {spec['out']} exists, skip"

    t0, t1 = T.window(storm)
    parts, k, cs = [], 0, t0
    while cs < t1:
        ce = min(cs + pd.Timedelta(hours=CHUNK_H), t1)
        got = _fetch_span(cs, ce, spec, folder)
        if got is None:
            raise RuntimeError(f"{tag}: 所有候选源都拿不到 {cs} 的数据"
                               f"（含二分回退到 {MIN_CHUNK_H}h）")
        parts.extend(got)
        k += 1
        cs = ce + pd.Timedelta(hours=3)

    ds = xr.concat(parts, dim="time").drop_duplicates("time").sortby("time")
    enc = {v: {"zlib": True, "complevel": 3} for v in spec["varnames"]}
    ds.to_netcdf(out, encoding=enc)
    n = len(ds.time)
    ds.close()
    for p in os.listdir(folder):
        if p.startswith(f"tmp_{spec['out'].replace('.nc', '')}_"):
            os.remove(os.path.join(folder, p))
    return f"{tag}: {spec['out']} done ({os.path.getsize(out)/1e6:.1f} MB, {n} steps)"


def main():
    worker_arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    prod = sys.argv[3] if len(sys.argv) > 3 else "uv0"
    spec = PRODUCTS[prod]
    storms = T.build_storm_list()
    dyn = (worker_arg == "dyn")
    worker = 0 if dyn else int(worker_arg)
    if worker == 0:
        print(f"[{prod}] 新增台风 {len(storms)} 个, depth={spec['depth']}", flush=True)
    for i, storm in enumerate(storms):
        if not dyn and i % nworkers != worker:
            continue
        T.ensure_track_meta(storm)
        folder = T.real_dir("union", T.tag_of(storm))
        if dyn:
            # 已完成 / 抢不到锁（别人在做）就跳过
            if os.path.exists(os.path.join(folder, spec["out"])) and \
               time_count(os.path.join(folder, spec["out"])) > 0:
                continue
            if not try_claim(folder, spec["out"]):
                continue
        try:
            print(download(storm, spec), flush=True)
        except Exception as e:
            print(f"{T.tag_of(storm)}: FAILED {e}", flush=True)
        finally:
            if dyn:
                lk = os.path.join(folder, f".lock_{spec['out']}")
                if os.path.exists(lk):
                    os.remove(lk)
    print(f"[{prod}] worker {worker} finished", flush=True)


if __name__ == "__main__":
    main()
