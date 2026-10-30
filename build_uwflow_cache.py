# -*- coding: utf-8 -*-
"""把 HYCOM 台风数据预处理成「磁盘缓存 + 可 mmap」的 npz（2026-10-10）。

动机：矩阵扫描要跑 144 个配置、评测还要再跑几百个 episode。每组都从 netCDF 逐个读
450 个台风要 ~50min，而同一份数组被反复使用。预处理一次后：
    * 载入从 ~50min 降到 ~1min
    * npz 不压缩 -> np.load(mmap_mode="r") 让多个训练/评测进程共享同一份 page cache，
      6 个并行进程的常驻内存从 6x7GB 降到几乎共享

缓存键：depth / grid_deg / crop_margin / native_res（native 或非 0.25° 不做缓存）。

用法：
    python build_uwflow_cache.py <depth_m> [grid_deg] [crop_margin]
"""
import fcntl, os, sys, time, gc
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import underwater_flow_env as ufe

CACHE_ROOT = os.environ.get("UWFLOW_CACHE_DIR", "/home/ding/main/data/uwflow_cache")

FIELDS = ["u", "v", "spd", "dist_to_land", "lon", "lat",
          "track_lon", "track_lat", "track_x", "track_y", "track_cum",
          "h0", "h1", "frac"]


def cache_dir(depth, grid, crop):
    return os.path.join(CACHE_ROOT, "d%g_g%g_c%g" % (depth, grid, crop))


def main():
    depth = float(sys.argv[1])
    grid = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    crop = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    out = cache_dir(depth, grid, crop)
    os.makedirs(out, exist_ok=True)
    shard_index = int(os.environ.get("CACHE_SHARD_INDEX", "0"))
    shard_count = int(os.environ.get("CACHE_SHARD_COUNT", "1"))
    if shard_count < 1 or not (0 <= shard_index < shard_count):
        raise ValueError("invalid cache shard configuration")
    lock_path = os.path.join(out, ".build_s%d_of_%d.lock" % (
        shard_index, shard_count
    ))
    lock_handle = open(lock_path, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("[cache] another builder is active for %s" % out, flush=True)
        return

    storms = sorted(os.listdir(os.path.join(ROOT, "data", "union_region")))
    storms = [s for s in storms
              if os.path.isfile(os.path.join(ROOT, "data", "union_region", s,
                                             "hycom_uv_%dm.nc" % int(depth)))]
    storms = storms[shard_index::shard_count]
    print("[cache] depth=%g grid=%g crop=%g shard=%d/%d storms=%d -> %s"
          % (depth, grid, crop, shard_index, shard_count, len(storms), out),
          flush=True)

    t0 = time.time()
    ok = skip = fail = 0
    for i, s in enumerate(storms, 1):
        p = os.path.join(out, s + ".npz")
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            skip += 1
            continue
        try:
            # 绕过内存缓存，避免 build 阶段把 450 个台风全留在 RAM
            ufe.cache_clear()
            d = ufe.load_storm(s, use_uv=True, depth_m=depth, crop_margin=crop,
                               grid_deg=grid, native_res=False)
            payload = {k: np.ascontiguousarray(d[k]) for k in FIELDS}
            payload["times"] = d["times"]
            payload["T"] = np.int64(d["T"])
            payload["landfall"] = np.int64(d["landfall"])
            payload["in_domain"] = np.asarray(d["in_domain"], dtype=np.int8)
            tmp = p + ".tmp.npz"
            np.savez(tmp, **payload)
            os.replace(tmp, p)
            ok += 1
        except Exception as e:
            fail += 1
            print("  FAIL %s: %s" % (s, repr(e)[:160]), flush=True)
        finally:
            ufe.cache_clear()
            gc.collect()
        if i % 25 == 0 or i == len(storms):
            el = time.time() - t0
            print("  [%d/%d] ok=%d skip=%d fail=%d  %.0fs  (%.2fs/storm)"
                  % (i, len(storms), ok, skip, fail, el, el / i), flush=True)

    size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e9
    print("[cache] DONE depth=%g ok=%d skip=%d fail=%d size=%.2f GB  %.0fs"
          % (depth, ok, skip, fail, size, time.time() - t0), flush=True)
    fcntl.flock(lock_handle, fcntl.LOCK_UN)
    lock_handle.close()


if __name__ == "__main__":
    main()
