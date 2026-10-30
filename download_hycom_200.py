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
import json, os, re, subprocess, sys, time
import pandas as pd
import xarray as xr

import typhoons_200 as T

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTROOT = os.environ.get(
    "HYCOM_DOWNLOAD_ROOT",
    os.path.join(ROOT, "data", "union_region"),
)
DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 50
CHUNK_H = int(os.environ.get("CHUNK_H", "24"))

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
    for attempt in range(3):
        try:
            subprocess.run(["curl", "--http1.1", "-sSL", "-f",
                            "--retry", "2", "--retry-delay", "8",
                            "--retry-all-errors", "--max-time", "600", "-o", path, url],
                           check=True)
            return
        except subprocess.CalledProcessError:
            time.sleep(12 * (attempt + 1))
    raise RuntimeError("curl failed")


def expected_steps(url):
    match = re.search(r"time_start=([^&]+)&time_end=([^&]+)", url)
    if not match:
        return 0
    hours = (
        pd.Timestamp(match.group(2)) - pd.Timestamp(match.group(1))
    ).total_seconds() / 3600.0
    return int(round(hours / 3.0)) + 1


def complete_download(path, url):
    expected = expected_steps(url)
    return expected > 0 and time_count(path) >= expected


def fetch_pair(url_u, path_u, url_v, path_v):
    """Fetch ESPC U/V, falling back to sequential requests if needed.

    The NCSS server occasionally rejects one side of a concurrent U/V pair
    with HTTP 500 even though the same request succeeds when issued alone.
    Retrying sequentially therefore recovers otherwise permanent tail
    stragglers.
    """
    cmd = ["curl", "--http1.1", "-sSL", "-f",
           "--retry", "0", "--retry-delay", "8",
           "--retry-all-errors", "--max-time", "180"]
    # Try the paired request once, then switch to sequential mode. Repeating
    # a failing paired request only wastes the server timeout budget.
    for attempt in range(1):
        processes = [
            subprocess.Popen(cmd + ["-o", path_u, url_u]),
            subprocess.Popen(cmd + ["-o", path_v, url_v]),
        ]
        codes = [process.wait() for process in processes]
        if codes == [0, 0]:
            return
        # Some completed NCSS responses remain open until curl times out. The
        # downloaded files can nevertheless be complete, so validate their
        # dimensions before discarding them and issuing another request.
        if complete_download(path_u, url_u) and complete_download(path_v, url_v):
            return
        time.sleep(12 * (attempt + 1))
    for url, path in ((url_u, path_u), (url_v, path_v)):
        for attempt in range(3):
            proc = subprocess.run(cmd + ["-o", path, url])
            if proc.returncode == 0:
                break
            if complete_download(path, url):
                break
            time.sleep(12 * (attempt + 1))
        else:
            raise RuntimeError("sequential curl failed")
    return


def fetch_espc_span(cs, ce, args, folder):
    """Fetch one ESPC U/V span, bisecting down to 3 h on server errors."""
    ctag = cs.strftime("%Y%m%dT%H")
    up = os.path.join(folder, f"chunk_{DEPTH}m_u_{ctag}.nc")
    vp = os.path.join(folder, f"chunk_{DEPTH}m_v_{ctag}.nc")
    a = args.copy()
    a["t0"] = cs.strftime("%Y-%m-%dT%H:%M:%SZ")
    a["t1"] = ce.strftime("%Y-%m-%dT%H:%M:%SZ")
    a["year"] = cs.year
    url_u = NCSS_ESPC_U.format(**a)
    url_v = NCSS_ESPC_V.format(**a)
    expected = int(round((ce - cs).total_seconds() / 3600.0 / 3.0)) + 1
    if time_count(up) >= expected and time_count(vp) >= expected:
        return [xr.merge([xr.open_dataset(up), xr.open_dataset(vp)],
                         join="outer", compat="override",
                         combine_attrs="override")]
    try:
        fetch_pair(url_u, up, url_v, vp)
        if time_count(up) >= expected and time_count(vp) >= expected:
            return [xr.merge([xr.open_dataset(up), xr.open_dataset(vp)],
                             join="outer", compat="override",
                             combine_attrs="override")]
    except Exception as exc:
        print(f"    [bisect] {ctag} {DEPTH}m failed: {exc}", flush=True)

    span_h = (ce - cs).total_seconds() / 3600.0
    if span_h > 3:
        steps = int(round(span_h / 3.0))
        mid = cs + pd.Timedelta(hours=3 * (steps // 2))
        return fetch_espc_span(cs, mid, args, folder) + \
            fetch_espc_span(mid, ce, args, folder)
    raise RuntimeError(f"could not fetch {DEPTH}m ESPC span starting {cs}")


def time_count(path):
    try:
        with xr.open_dataset(path) as d:
            return int(len(d.time))
    except Exception:
        return -1


def expose_output(storm, out):
    """Expose a NAS product through the canonical data/union_region path."""
    tag = T.tag_of(storm)
    canonical = os.path.join(
        ROOT, "data", "union_region", tag, os.path.basename(out)
    )
    if os.path.abspath(canonical) == os.path.abspath(out):
        return
    os.makedirs(os.path.dirname(canonical), exist_ok=True)
    if not os.path.lexists(canonical):
        os.symlink(out, canonical)


def try_claim(folder, outname, stale_s=1800):
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


def tight_area_for(storm, margin_km=450.0, grid=0.25):
    track = os.path.join(ROOT, "data", "selected_typhoons", T.tag_of(storm),
                         "track.csv")
    df = pd.read_csv(track, usecols=["LAT", "LON"]).dropna()
    lat0, lat1 = float(df.LAT.min()), float(df.LAT.max())
    lon0, lon1 = float(df.LON.min()), float(df.LON.max())
    mid = 0.5 * (lat0 + lat1)
    import math
    dlat = margin_km / 111.0
    dlon = margin_km / (111.0 * math.cos(math.radians(mid)))
    west = max(T.LON0, math.floor((lon0 - dlon) / grid) * grid)
    east = min(T.LON1, math.ceil((lon1 + dlon) / grid) * grid)
    south = max(T.LAT0, math.floor((lat0 - dlat) / grid) * grid)
    north = min(T.LAT1, math.ceil((lat1 + dlat) / grid) * grid)
    return north, west, south, east


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
    if os.environ.get("HYCOM_TIGHT_BBOX", "0") == "1":
        north, west, south, east = tight_area_for(storm)
    else:
        north, west, south, east = T.LAT1, T.LON0, T.LAT0, T.LON1
    args = dict(north=north, west=west, east=east, south=south, depth=DEPTH)

    parts = []
    k = 0
    cs = t0
    while cs < t1:
        ce = min(cs + pd.Timedelta(hours=CHUNK_H), t1)
        a = args.copy()
        a["t0"] = cs.strftime("%Y-%m-%dT%H:%M:%SZ")
        a["t1"] = ce.strftime("%Y-%m-%dT%H:%M:%SZ")
        if src == "espc":
            parts.extend(fetch_espc_span(cs, ce, args, folder))
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
    expose_output(storm, out)
    for p in os.listdir(folder):
        if p.startswith(f"chunk_{DEPTH}m_"):
            os.remove(os.path.join(folder, p))
    return f"{storm['name']}_{storm['season']}:{DEPTH}m done ({os.path.getsize(out)/1e6:.1f} MB)"


def main():
    worker_arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    nworkers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    storms = T.build_storm_list()
    tag_filter = os.environ.get("TAG_FILTER", "").strip()
    if tag_filter:
        storms = [s for s in storms if T.tag_of(s) == tag_filter]
    dynamic = worker_arg == "dyn"
    worker = 0 if dynamic else int(worker_arg)
    if worker == 0:
        print(f"total typhoon-level storms 2020+: {len(storms)}", flush=True)
    for i, storm in enumerate(storms):
        if not dynamic and i % nworkers != worker:
            continue
        T.ensure_track_meta(storm)
        folder = os.path.join(OUTROOT, T.tag_of(storm))
        outname = f"hycom_uv_{DEPTH}m.nc"
        if dynamic:
            if os.path.exists(os.path.join(folder, outname)) and \
               time_count(os.path.join(folder, outname)) > 0:
                continue
            if not try_claim(folder, outname):
                continue
        try:
            print(download(storm), flush=True)
        except Exception as e:
            print(f"{storm['name']}_{storm['season']}:50m FAILED {e}", flush=True)
        finally:
            if dynamic:
                lock = os.path.join(folder, f".lock_{outname}")
                if os.path.exists(lock):
                    os.remove(lock)
    print(f"worker {worker} finished", flush=True)


if __name__ == "__main__":
    main()
