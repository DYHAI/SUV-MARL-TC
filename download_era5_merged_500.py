"""ERA5 500 补下载的**合并请求**版本：把「一台风一月一变量一个请求」合并成
「**一个月一个请求（含全部 4 个变量，覆盖该月所有待补台风）**」，然后在本地切给各台风。

为什么要这么干（2026-10-09 实测）：
  · CDS 全局拥堵（Live 页 running_requests ~540 / queued_requests 1 万+），
    而**每个账号只允许约 1–3 个请求排队** → 我们能同时在飞的请求 ≈ 账号数 = 7。
  · 于是总时长 ≈ (请求数 / 7) × 单请求耗时。原方案要 ~478 个请求 → 按 2 h/请求算是**几天**；
    合并成「按月」后只剩 ~60–90 个请求 → **约 5–7 倍提速**。
  · 单个合并请求体积仍可控（月 × 合并 bbox × 全变量 ≈ 100–300 MB）。

切出来的单台风文件与 `download_era5_*_500.py` 产物**格式完全一致**（变量名 u10/v10/sst/msl，
dims time/latitude/longitude，时间轴 = 该台风的窗口），所以训练侧无需任何改动。
空间范围取「该台风轨迹 bbox + 450 km」（与 AREA_MODE=tight 同口径；训练侧本来就只保留 track±400 km）。

用法: python download_era5_merged_500.py            # 只处理"有缺失"的月份
      python download_era5_merged_500.py 2011-05    # 只处理指定月份（调试用）
"""
import os, sys, time, ssl, json
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cdsapi
import typhoons_500 as T

ROOT = os.path.dirname(os.path.abspath(__file__))
MERGED_DIR = os.environ.get(
    "ERA5_MERGED_DIR",
    os.path.join(ROOT, "data", "era5_merged"),
)
VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind",
        "sea_surface_temperature", "mean_sea_level_pressure"]
TIMES = [f"{h:02d}:00" for h in range(24)]
OUTMAP = {"wind": ("era5_wind", "era5_10m_wind_hourly.nc", ("u10", "v10")),
          "sst": ("era5_sst", "era5_sst_hourly.nc", ("sst",)),
          "mslp": ("era5_mslp", "era5_mslp_hourly.nc", ("msl",))}


def out_path(kind, tag):
    d, f, _ = OUTMAP[kind]
    return os.path.join(ROOT, "data", d, tag, f)


def missing_kinds(storm):
    tag = T.tag_of(storm)
    return [k for k in OUTMAP if not (os.path.exists(out_path(k, tag))
                                      and os.path.getsize(out_path(k, tag)) > 1000)]


def group_jobs(only=None):
    """返回 {(起始年, 起始月): [(storm, tag, kinds, win_t0, win_t1), ...]}，只含"有缺失"的分组。

    ⚠️ 关键：台风**只归属到它的起始月**（而不是按月拆开）。
    否则窗口跨月的台风会被两个分片同时处理、**同时写同一个输出文件** → 互相覆盖/写坏。
    跨月的时间范围由该分组的请求自己处理（按自然月发多个请求再拼接，见 fetch_group）。
    """
    groups = {}
    for s in T.build_storm_list():
        miss = missing_kinds(s)
        if not miss:
            continue
        tag = T.tag_of(s)
        t0, t1 = T.window(s)
        key = (t0.year, t0.month)
        if only and f"{key[0]}-{key[1]:02d}" != only:
            continue
        groups.setdefault(key, []).append((s, tag, tuple(miss), t0, t1))
    # 分片：7 个账号各跑一部分月份（避免 7 个进程互相重复做同一个月）。
    shard = os.environ.get("MERGED_SHARD"); nsh = int(os.environ.get("MERGED_NSHARD", "1"))
    if shard is not None:
        i = int(shard)
        groups = {k: v for k, v in groups.items()
                  if ((k[0] * 12 + k[1]) % nsh) == i}
    return groups


def _by_month(t0, t1):
    """把 [t0,t1] 按月切，返回 {((y,m)): (start,end)}。"""
    out = []
    cur = t0
    while cur <= t1:
        y, m = cur.year, cur.month
        nxt = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthBegin(1)
        end = min(t1, nxt - pd.Timedelta(hours=1))
        out.append(((y, m), (cur, end)))
        cur = end + pd.Timedelta(hours=1)
    return out


def merged_area_and_days(items):
    areas = [T.tight_area(tag) for _, tag, _, _, _ in items]
    N = max(a[0] for a in areas); W = min(a[1] for a in areas)
    S = min(a[2] for a in areas); E = max(a[3] for a in areas)
    days = set()
    for _, _, _, a, b in items:
        days |= {d.day for d in pd.date_range(a.normalize(), b.normalize(), freq="D")}
    return [N, W, S, E], sorted(days)


def retrieve_retry(c, req, target, max_attempts=8):
    """⚠️ 只对**任务级拒绝**重试，绝不对连接/轮询类错误重新提交。

    教训（2026-10-09）：原来把 `Connection / Max retries exceeded / timed out` 也当可重试，
    结果是 cdsapi 在**轮询**阶段遇到网络抖动抛错 → 我们**又提交一个新任务**，
    而旧任务仍留在队列里 → 每账号 pending 涨到 2–3 → 命中 CDS 的每账号排队上限 → 再次 `rejected` 死循环。
    正确做法：任务级拒绝（配额/排队上限）→ 重新提交；网络/轮询错误 → 直接抛出，
    由看门狗在**取消该账号陈旧任务之后**重启本分片（避免堆叠）。
    """
    for attempt in range(max_attempts):
        try:
            c.retrieve("reanalysis-era5-single-levels", req, target)
            return
        except Exception as e:
            msg = str(e)
            job_level = any(k in msg for k in ("temporarily limited", "queued requests",
                                               "has been rejected", "rejected"))
            if job_level:
                wait = min(60 * (attempt + 1), 900)
                print(f"    [任务级拒绝] retry {attempt+1}/{max_attempts} in {wait}s: {msg[:110]}",
                      flush=True)
                time.sleep(wait)
                continue
            print(f"    [非任务级错误，交给看门狗重启] {msg[:140]}", flush=True)
            raise
    raise RuntimeError("CDS retries exhausted")



def _status(remote):
    try:
        return str(remote.status)
    except Exception as e:
        return f"unknown({type(e).__name__})"


def retrieve_resumable(dc, req, target, jobfile, poll_s=30, max_wait_h=6.0, stuck_h=3.0):
    """提交/等待/下载 CDS 任务，**可断点续跑 + 自动识别"任务被 CDS 丢掉"**。

    ⚠️ 实测（2026-10-09）：CDS 会**静默丢弃**排队中的任务（再用 jobID 查询直接 404）。
    原来的实现把 404 当普通网络错误无限重试 → 分片永远卡在一个不存在的任务上（我们就这样白等了 1 小时）。
    现在：
      · 连续查询失败 ≥10 次，或**见到 404** → 判定任务已消失，删掉 jobfile 并**重新提交**；
      · 一直停留在 accepted/running 超过 `stuck_h` 小时 → 同样判定为"卡死"，重新提交（CDS 队列不保证公平）。
    """
    for _round in range(8):                     # 最多重投 8 轮，避免无脑循环
        jid = None
        if os.path.exists(jobfile):
            jid = (open(jobfile).read().strip() or None)
        if jid:
            print(f"    [resume] 继续等待已提交任务 {jid[:8]}…", flush=True)
        else:
            remote = dc.submit(collection_id="reanalysis-era5-single-levels", request=req)
            jid = remote.request_id
            with open(jobfile, "w") as f:
                f.write(jid)
            print(f"    [submit] 新任务 {jid[:8]}", flush=True)
        t0 = time.time()
        last, fails, last_change = None, 0, time.time()
        while True:
            try:
                st = _status(dc.get_remote(jid))
                fails = 0
            except Exception as e:
                fails += 1
                st = f"poll_error({type(e).__name__})"
                if "404" in str(e):
                    print(f"    [gone] 任务 {jid[:8]} 返回 404（CDS 已丢弃）→ 重新提交", flush=True)
                    break
                if fails >= 10:
                    print(f"    [gone] 任务 {jid[:8]} 连续 {fails} 次查询失败 → 重新提交", flush=True)
                    break
            if st != last:
                print(f"    [status] {jid[:8]} -> {st}（已等 {(time.time()-t0)/60:.0f} min）", flush=True)
                last, last_change = st, time.time()
            if st in ("successful", "completed"):
                path = dc.download_results(jid, target)
                os.remove(jobfile)
                return path
            if st in ("failed", "rejected", "dismissed", "deleted"):
                print(f"    [terminal] 任务 {jid[:8]} 终态 {st} → 重新提交", flush=True)
                break
            if time.time() - t0 > max_wait_h * 3600 or (time.time() - last_change) > stuck_h * 3600:
                print(f"    [stuck] 任务 {jid[:8]} 卡在 {st} 太久 → 重新提交", flush=True)
                break
            time.sleep(poll_s)
        try:
            os.remove(jobfile)                  # 走到这里说明该任务已不可用
        except FileNotFoundError:
            pass
    raise RuntimeError("CDS 任务反复丢失/卡死，放弃")


def clean(ds):
    ds = ds.drop_vars(["number", "expver"], errors="ignore")
    if "valid_time" in ds.coords or "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    return ds



def _month_days(items, y, m):
    """该分组里所有台风在 (y,m) 这个自然月内需要的"日"集合。"""
    days = set()
    for _, _, _, a, b in items:
        lo = max(a, pd.Timestamp(year=y, month=m, day=1))
        nxt = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthBegin(1)
        hi = min(b, nxt - pd.Timedelta(hours=1))
        if lo <= hi:
            days |= {d.day for d in pd.date_range(lo.normalize(), hi.normalize(), freq="D")}
    return sorted(days)




def _shard_tag():
    """当前分片的标签（用于分块/job 文件名，避免不同分片抢同一个文件）。"""
    return f"_s{os.environ.get('MERGED_SHARD', '0')}"


def _part_ok(path, days):
    """分块文件是否已经存在且覆盖需要的"日"。"""
    if not (os.path.exists(path) and os.path.getsize(path) > 1_000_000):
        return False
    try:
        with xr.open_dataset(path) as d:
            have = set(pd.to_datetime(d["time"].values).day)
        return set(days) <= have
    except Exception:
        return False


def fetch_group(dc, key, items, out_path_merged):
    """抓取一个分组：按自然月逐个请求（跨月台风会产生 2 个请求），拼接成一个合并文件。

    为什么按自然月拆：CDS 的 `day` 列表是**对列表里每个月都生效**的，
    跨月时若把两个月塞进同一个请求，就必须给两个月的并集日列表，
    二月这种月份会直接报错（day=30/31 不存在）。
    """
    y0, m0 = key
    area, _ = merged_area_and_days([(s, t, k, a, b) for s, t, k, a, b in items])
    months = sorted({(d.year, d.month)
                     for _, _, _, a, b in items
                     for d in pd.date_range(a.normalize(), b.normalize(), freq="D")})
    parts = []
    for (y, m) in months:
        days = _month_days(items, y, m)
        if not days:
            continue
        # ⚠️ 分块/job 文件必须**带分片号**：不同分组可能都需要同一个自然月
        #    （例如 group(2011,8) 要 2011-08+2011-09，group(2011,9) 要 2011-09+2011-10），
        #    若共用 `part_201109.nc` / `.job_201109.txt`，两个分片会同时续跑同一个任务、
        #    同时写同一个文件 → 竞态 + 互相覆盖。
        st = _shard_tag()
        tmp = os.path.join(MERGED_DIR, f"part_{y}{m:02d}{st}.nc")
        jobfile = os.path.join(MERGED_DIR, f".job_{y}{m:02d}{st}.txt")
        # 已经下好且覆盖所需天数的分块直接复用（重启/换分片时不要重复下载 100–200 MB）
        if _part_ok(tmp, days):
            print(f"[{y}-{m:02d}] 复用已有分块 {os.path.getsize(tmp)/1e6:.1f} MB", flush=True)
            parts.append(tmp)
            continue
        req = {"product_type": "reanalysis", "variable": VARS,
               "year": str(y), "month": f"{m:02d}",
               "day": [f"{d:02d}" for d in days], "time": TIMES,
               "area": area, "format": "netcdf"}
        print(f"[{y}-{m:02d}] 提交合并请求：{len(items)} 个台风，天 {days[0]}..{days[-1]}"
              f"（{len(days)} 天），area={area}", flush=True)
        t = time.time()
        retrieve_resumable(dc, req, tmp, jobfile)
        print(f"[{y}-{m:02d}] 分块完成 {os.path.getsize(tmp)/1e6:.1f} MB，用时 {time.time()-t:.0f}s",
              flush=True)
        parts.append(tmp)
    if not parts:
        raise RuntimeError(f"分组 {key} 没有任何需要抓取的月份")
    ds = xr.concat([clean(xr.open_dataset(p)) for p in parts], dim="time")
    ds = ds.sortby("time").drop_duplicates("time")
    enc = {v: {"zlib": True, "complevel": 3} for v in ds.data_vars}
    ds.to_netcdf(out_path_merged, encoding=enc)
    ds.close()
    for p in parts:
        os.remove(p)

def split_to_storms(merged_path, items):
    """把合并文件按台风切开，写出与单台风下载器完全一致的三个文件。"""
    wrote = []
    with xr.open_dataset(merged_path) as ds0:
        ds = clean(ds0)
        for s, tag, kinds, a, b in items:
            area = T.tight_area(tag)                     # [N, W, S, E]
            sub = ds.sel(time=slice(a, b),
                         latitude=slice(area[0], area[2]),
                         longitude=slice(area[1], area[3]))
            sub = sub.sortby("time").drop_duplicates("time")
            if sub.sizes.get("time", 0) < 2:
                print(f"    ⚠️ {tag}: 切出来只有 {sub.sizes.get('time',0)} 个时刻，跳过")
                continue
            for kind in kinds:
                d, f, vnames = OUTMAP[kind]
                p = out_path(kind, tag)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                have = [v for v in vnames if v in sub.variables]
                if len(have) != len(vnames):
                    print(f"    ⚠️ {tag}/{kind}: 缺变量 {set(vnames)-set(have)}，跳过")
                    continue
                enc = {v: {"zlib": True, "complevel": 3} for v in vnames}
                # ⚠️ 必须 list(vnames)：xarray 用 tuple 索引会被当成"多键选择"，
                #    单个变量 ('msl',) 会直接 KeyError（这个 bug 曾让所有切分静默失败）。
                sub[list(vnames)].to_netcdf(p, encoding=enc)
                wrote.append(f"{tag}/{kind}")
            print(f"    ✓ {tag}: 切出 {len(kinds)} 个文件（时间 {sub.sizes['time']} 步）", flush=True)
    return wrote


def resplit_existing():
    """把 MERGED_DIR 里已存在的合并文件重新切给各台风（用于修复切分 bug 的历史数据）。"""
    groups = group_jobs()
    n = 0
    for (y, m), items in sorted(groups.items()):
        mp = os.path.join(MERGED_DIR, f"{y}{m:02d}.nc")
        if os.path.exists(mp) and os.path.getsize(mp) > 1_000_000:
            print(f"[{y}-{m:02d}] 重新切分 {os.path.getsize(mp)/1e6:.1f} MB", flush=True)
            split_to_storms(mp, items)
            n += 1
    print(f"重新切分完成：{n} 个月份组", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--resplit":
        resplit_existing()
        return
    only = sys.argv[1] if len(sys.argv) > 1 else None
    key = os.environ.get("CDS_KEY")
    # progress=False：cdsapi 的 tqdm 进度条会把日志刷成几万行（每次下载 500+ 行），
    # 既盖住真正的错误信息，也让排查无从下手。
    c = cdsapi.Client(url=os.environ.get("CDS_URL", "https://cds.climate.copernicus.eu/api"),
                      key=key, progress=False) if key else cdsapi.Client(progress=False)
    dc = c.client          # ecmwf.datastores.Client：submit / get_remote / download_results
    os.makedirs(MERGED_DIR, exist_ok=True)
    groups = group_jobs(only)
    tot = sum(len(v) for v in groups.values())
    print(f"待补月份 {len(groups)} 个，涉及 {tot} 个 (台风,月) — "
          f"合并前需要 {sum(len(set(x[2] for x in v)) and len(v) for v in groups.values())} "
          f"次单台风请求，合并后只需 {len(groups)} 次", flush=True)
    for (y, m), items in sorted(groups.items()):
        mp = os.path.join(MERGED_DIR, f"{y}{m:02d}.nc")
        if not (os.path.exists(mp) and os.path.getsize(mp) > 1_000_000):
            fetch_group(dc, (y, m), items, mp)
        else:
            print(f"[{y}-{m:02d}] 复用已有合并文件", flush=True)
        split_to_storms(mp, items)
    print("全部完成", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
