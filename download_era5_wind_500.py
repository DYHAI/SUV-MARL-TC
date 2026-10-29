"""Auto-generated from download_era5_2020plus.py: 2019-2025 全部热带系统（201 个）版本。

只改了 import 与目录名（用 typhoons_200.tag_of 消歧），URL/变量/文件格式与 download_era5_2020plus.py 完全一致。
用法: python download_era5_wind_200.py <worker> <nworkers>
"""
"""Download ERA5 10m wind (u10, v10) for all 2020+ typhoon-intensity storms.

Matches the format of the original 8 storms:
  data/era5_wind/<NAME>_<YEAR>/era5_10m_wind_hourly.nc  (u10, v10, HOURLY)

Fixed union bbox AREA = [56.7, 98.7, 6.8, 171.6] (N, W, S, E), aligned to HYCOM.
Usage: python download_era5_2020plus.py <worker> <nworkers>
"""
import os, sys, time, time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # 规避 CDS SSL 连接中断
import pandas as pd
import cdsapi
import xarray as xr

import typhoons_500 as T

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(ROOT, "data", "era5_wind")
AREA = [56.7, 98.7, 6.8, 171.6]  # N, W, S, E  (same union bbox)
TIMES = [f"{h:02d}:00" for h in range(24)]  # hourly


def month_chunks(days):
    groups = {}
    for d in days:
        groups.setdefault((d.year, d.month), []).append(d)
    return groups


def retrieve_retry(c, request, target, max_attempts=12):
    """Retry on CDS queue-limit / SSL / connection errors / truncated result files.

    注意：结果文件被截断时 cdsapi 会抛 "File size mismatch"，必须重试而不是直接失败，
    否则会留下「静默缺失」的台风（旧版就没重试，导致 35 个 wind 直接 FAILED）。
    """
    for attempt in range(max_attempts):
        try:
            c.retrieve("reanalysis-era5-single-levels", request, target)
            return
        except Exception as e:
            msg = str(e)
            retryable = ("queued requests" in msg or "temporarily limited" in msg or "rejected" in msg
                         or "SSL" in msg or "EOF" in msg or "Connection" in msg
                         or "File size mismatch" in msg or "size mismatch" in msg
                         or "HDF error" in msg or "NetCDF" in msg or "unpack" in msg
                         or "timed out" in msg.lower() or "timeout" in msg.lower()
                         or "Max retries exceeded" in msg or "ChunkedEncodingError" in msg)
            if retryable:
                wait = min(20 * (attempt + 1), 300)
                print(f"  retry {attempt+1}/{max_attempts} in {wait}s: {msg[:90]}", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("CDS retries exhausted")


def verify_nc(path, var_names):
    """重新打开并检查变量/时间轴，防止截断或损坏的 nc 被当成成功。"""
    with xr.open_dataset(path) as d:
        missing = [v for v in var_names if v not in d.variables]
        n = int(d.sizes.get("time", 0))
    if missing or n == 0:
        raise RuntimeError(f"corrupt netcdf {os.path.basename(path)}: missing={missing} ntime={n}")
    return n


def download(storm):
    outdir = T.real_dir("era5_wind", T.tag_of(storm))   # 落 NVMe + symlink
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "era5_10m_wind_hourly.nc")
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return f"{storm['name']}_{storm['season']}: exists, skip"

    t0, t1 = T.window(storm)
    days = pd.date_range(t0, t1, freq="3h")  # window snap grid (still hourly in TIME list)
    _k = os.environ.get('CDS_KEY'); _u = os.environ.get('CDS_URL', 'https://cds.climate.copernicus.eu/api')
    c = cdsapi.Client(url=_u, key=_k) if _k else cdsapi.Client()
    parts = []
    for (year, month), ds in sorted(month_chunks(days).items()):
        day_list = sorted({f"{d.day:02d}" for d in ds})
        tmp = os.path.join(outdir, f"tmp_{year}{month:02d}.nc")
        retrieve_retry(c, {
                "product_type": "reanalysis",
                "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "year": str(year),
                "month": f"{month:02d}",
                "day": day_list,
                "time": TIMES,
                "area": T.area_for(storm),
                "format": "netcdf",
            }, tmp)
        parts.append(tmp)

    def _clean(p):
        d = xr.open_dataset(p)
        d = d.drop_vars(["number", "expver"], errors="ignore")
        d = d.rename({"valid_time": "time"})
        return d

    try:
        dset = xr.concat([_clean(p) for p in parts], dim="time")
        dset = dset.sortby("time").drop_duplicates("time")
        enc = {v: {"zlib": True, "complevel": 3} for v in ("u10", "v10")}
        dset.to_netcdf(out, encoding=enc)
        nsteps = len(dset.time)
        dset.close()
        verify_nc(out, ("u10", "v10"))
    except Exception:
        for p in parts + [out]:
            if os.path.exists(p):
                os.remove(p)
        raise
    for p in parts:
        os.remove(p)
    return f"{storm['name']}_{storm['season']}: done ({os.path.getsize(out)/1e6:.1f} MB, {nsteps} steps)"


def _out_path(storm):
    return os.path.join(T.real_dir("era5_wind", T.tag_of(storm)), "era5_10m_wind_hourly.nc")


def try_claim(folder, outname, stale_s=900):
    """动态抢单（同 download_hycom_500.py）：空 worker 领下一个未完成台风。
    锁超过 stale_s 视为陈旧可抢；完成/失败都会在 finally 删锁。"""
    lock = os.path.join(folder, f".lock_{outname}")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(lock) > stale_s:
                os.remove(lock)
        except FileNotFoundError:
            pass
        return False


def main():
    worker_arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    dyn = (worker_arg == "dyn")
    worker = 0 if dyn else int(worker_arg)
    storms = T.build_storm_list()
    if worker == 0:
        print(f"total typhoon-level storms 2020+: {len(storms)}", flush=True)
    for i, storm in enumerate(storms):
        if not dyn and i % nworkers != worker:
            continue
        T.ensure_track_meta(storm)
        op = _out_path(storm)
        if dyn:
            if os.path.exists(op) and os.path.getsize(op) > 1000:
                continue
            if not try_claim(os.path.dirname(op), os.path.basename(op)):
                continue
        try:
            print(download(storm), flush=True)
        except Exception as e:
            print(f"{storm['name']}_{storm['season']}: FAILED {e}", flush=True)
        finally:
            if dyn:
                lk = os.path.join(os.path.dirname(op), f".lock_{os.path.basename(op)}")
                if os.path.exists(lk):
                    os.remove(lk)
    print(f"worker {worker} finished", flush=True)


if __name__ == "__main__":
    main()
