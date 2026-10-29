"""Auto-generated from download_hycom_2020plus.py: 2019-2025 全部热带系统（201 个）版本。

只改了 import 与目录名（用 typhoons_200.tag_of 消歧），URL/变量/文件格式与 download_hycom_2020plus.py 完全一致。
用法: python download_hycom_200.py <worker> <nworkers> <depth>
"""
"""Download HYCOM 50m (u,v) for all 2020+ typhoon-intensity storms on the fixed union bbox.

Matches the format of the original 8 storms:
  data/union_region/<NAME>_<YEAR>/hycom_uv_50m.nc  (water_u, water_v, 3-hourly)

Data sources:
  * GLBy0.08/expt_93.0  (GOFS3.1, 2018-12 .. 2024-09-04)  -> 2020..2023 + early 2024
  * ESPC-D-V02 u3z/v3z  (GOFS3.5, 2024-08-10 .. present)   -> late 2024 + 2025

Usage: python download_hycom_2020plus.py <worker> <nworkers>
"""
import json, os, subprocess, sys, time
import pandas as pd
import xarray as xr

import typhoons_200 as T

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTROOT = os.environ.get(
    "HYCOM_DOWNLOAD_ROOT",
    os.path.join(ROOT, "data", "union_region"),
)
DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 50
CHUNK_H = 48

NCSS_93 = ("https://ncss.hycom.org/thredds/ncss/grid/GLBy0.08/expt_93.0/uv3z/{year}"
           "?var=water_u&var=water_v&vertCoord={depth}&accept=netcdf4"
           "&north={north}&west={west}&east={east}&south={south}"
           "&time_start={t0}&time_end={t1}")
NCSS_ESPC_U = ("https://ncss.hycom.org/thredds/ncss/grid/ESPC-D-V02/u3z/{year}"
               "?var=water_u&vertCoord={depth}&accept=netcdf4"
               "&north={north}&west={west}&east={east}&south={south}"
               "&time_start={t0}&time_end={t1}")
NCSS_ESPC_V = ("https://ncss.hycom.org/thredds/ncss/grid/ESPC-D-V02/v3z/{year}"
               "?var=water_v&vertCoord={depth}&accept=netcdf4"
               "&north={north}&west={west}&east={east}&south={south}"
               "&time_start={t0}&time_end={t1}")


def fetch(url, path):
    for attempt in range(5):
        try:
            subprocess.run(["curl", "-sSL", "-f", "--retry", "5", "--retry-delay", "8",
                            "--retry-all-errors", "--max-time", "1800", "-o", path, url],
                           check=True)
            return
        except subprocess.CalledProcessError:
            time.sleep(12 * (attempt + 1))
    raise RuntimeError("curl failed")


def time_count(path):
    try:
        with xr.open_dataset(path) as d:
            return int(len(d.time))
    except Exception:
        return -1


def download(storm):
    folder = os.path.join(OUTROOT, T.tag_of(storm))
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, f"hycom_uv_{DEPTH}m.nc")
    if os.path.exists(out) and time_count(out) > 0:
        return f"{storm['name']}_{storm['season']}:{DEPTH}m exists, skip"
    local_out = os.path.join(
        ROOT, "data", "union_region", T.tag_of(storm), f"hycom_uv_{DEPTH}m.nc"
    )
    if os.path.abspath(local_out) != os.path.abspath(out) and \
       os.path.exists(local_out) and time_count(local_out) > 0:
        return f"{storm['name']}_{storm['season']}:{DEPTH}m local exists, skip"

    t0, t1 = T.window(storm)
    src = T.hycom_source(storm)
    year = t0.year
    args = dict(north=T.LAT1, west=T.LON0, east=T.LON1, south=T.LAT0, depth=DEPTH)

    parts = []
    k = 0
    cs = t0
    while cs < t1:
        ce = min(cs + pd.Timedelta(hours=CHUNK_H), t1)
        a = args.copy()
        a["t0"] = cs.strftime("%Y-%m-%dT%H:%M:%SZ")
        a["t1"] = ce.strftime("%Y-%m-%dT%H:%M:%SZ")
        if src == "espc":
            a["year"] = cs.year
            up = os.path.join(folder, f"tmp_{DEPTH}m_u.nc")
            vp = os.path.join(folder, f"tmp_{DEPTH}m_v.nc")
            fetch(NCSS_ESPC_U.format(**a), up)
            fetch(NCSS_ESPC_V.format(**a), vp)
            if time_count(up) < 1 or time_count(vp) < 1:
                for p in (up, vp):
                    if os.path.exists(p):
                        os.remove(p)
                raise RuntimeError("empty ESPC chunk")
            parts.append(xr.merge([xr.open_dataset(up), xr.open_dataset(vp)]))
            os.remove(up); os.remove(vp)
        else:
            a["year"] = cs.year
            p = os.path.join(folder, f"tmp_{DEPTH}m_{k}.nc")
            fetch(NCSS_93.format(**a), p)
            if time_count(p) < 1:
                os.remove(p)
                raise RuntimeError("empty expt93 chunk")
            parts.append(xr.open_dataset(p))
            os.remove(p)
        k += 1
        cs = ce + pd.Timedelta(hours=3)

    ds = xr.concat(parts, dim="time").drop_duplicates("time").sortby("time")
    enc = {v: {"zlib": True, "complevel": 3} for v in ("water_u", "water_v")}
    ds.to_netcdf(out, encoding=enc)
    ds.close()
    return f"{storm['name']}_{storm['season']}:{DEPTH}m done ({os.path.getsize(out)/1e6:.1f} MB)"


def main():
    worker = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    storms = T.build_storm_list()
    if worker == 0:
        print(f"total typhoon-level storms 2020+: {len(storms)}", flush=True)
    for i, storm in enumerate(storms):
        if i % nworkers != worker:
            continue
        T.ensure_track_meta(storm)
        try:
            print(download(storm), flush=True)
        except Exception as e:
            print(f"{storm['name']}_{storm['season']}:50m FAILED {e}", flush=True)
    print(f"worker {worker} finished", flush=True)


if __name__ == "__main__":
    main()
