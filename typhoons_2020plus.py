"""Shared helper: build the 2020+ typhoon-intensity storm list and ensure track/meta.

The fixed union bbox matches the existing 8-storm downloads:
  lon [98.7, 171.6], lat [6.8, 56.7]
Typhoon intensity = max WMO_WIND >= 64 kt.
Seasons 2020..2025 (complete), WP basin only.

Each storm gets data/selected_typhoons/<NAME>_<YEAR>/{track.csv,meta.json} with the
same format used by the original 8 storms (window = track min-6h .. max+24h snapped 3h).
"""
import json, os, re
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
SEL = os.path.join(DATA, "selected_typhoons")
IB = os.path.join(DATA, "scs_typhoons", "ibtracs_WP.csv")

# fixed union bbox (same as the original 8)
LON0, LON1, LAT0, LAT1 = 98.7, 171.6, 6.8, 56.7
MIN_WIND = 64      # kt, typhoon intensity threshold
MIN_SEASON = 2020
MAX_SEASON = 2025  # complete seasons only (2026 is in-progress)
# expt_93.0 ends 2024-09-04; ESPC-D-V02 starts 2024-08-10 (overlap, no gap)
EXPT93_END = pd.Timestamp("2024-09-04 00:00:00")


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(name))


def _read_ibtracs():
    ib = pd.read_csv(IB, skiprows=[1], low_memory=False)
    for c in ["SEASON", "LAT", "LON", "WMO_WIND"]:
        ib[c] = pd.to_numeric(ib[c], errors="coerce")
    ib["WMO_WIND"] = ib["WMO_WIND"].fillna(0.0)
    ib["t"] = pd.to_datetime(ib["ISO_TIME"])
    return ib


def build_storm_list():
    """Return list of dicts for typhoon-intensity storms 2020+ in the fixed bbox."""
    ib = _read_ibtracs()
    inb = (ib.LON >= LON0) & (ib.LON <= LON1) & (ib.LAT >= LAT0) & (ib.LAT <= LAT1)
    sub = ib[inb & (ib.SEASON >= MIN_SEASON) & (ib.SEASON <= MAX_SEASON)].copy()
    g = sub.groupby("SID").agg(
        name=("NAME", "first"), season=("SEASON", "first"),
        maxwind=("WMO_WIND", "max"), tmin=("t", "min"), tmax=("t", "max"),
        nbox=("LAT", "count"),
    )
    ty = g[g.maxwind >= MIN_WIND].sort_values(["season", "tmin"])
    storms = []
    for sid, r in ty.iterrows():
        storms.append({
            "name": sanitize(r["name"]),
            "raw_name": str(r["name"]),
            "season": int(r["season"]),
            "sid": sid,
            "maxwind": float(r["maxwind"]),
            "tmin": pd.Timestamp(r["tmin"]),
            "tmax": pd.Timestamp(r["tmax"]),
            "nbox": int(r["nbox"]),
        })
    return storms


def snap3(dt, up):
    base = dt.replace(hour=(dt.hour // 3) * 3, minute=0, second=0, microsecond=0)
    if up and (dt.hour % 3 or dt.minute or dt.second):
        base = base + pd.Timedelta(hours=3)
    return base


def window(storm):
    """download window: track min-6h .. max+24h, snapped to 3h."""
    t0 = snap3(storm["tmin"] - pd.Timedelta(hours=6), False)
    t1 = snap3(storm["tmax"] + pd.Timedelta(hours=24), True)
    return t0, t1


def ensure_track_meta(storm):
    """Write track.csv + meta.json for a storm (idempotent; meta always refreshed)."""
    folder = os.path.join(SEL, f"{storm['name']}_{storm['season']}")
    os.makedirs(folder, exist_ok=True)
    track = os.path.join(folder, "track.csv")
    meta = os.path.join(folder, "meta.json")
    if not os.path.exists(track):
        ib = _read_ibtracs()
        g = ib[ib.SID == storm["sid"]].sort_values("t")
        g.to_csv(track, index=False)
    t0, t1 = window(storm)
    src = ("HYCOM GLBy0.08/expt_93.0 uv3z" if t1 <= EXPT93_END
           else "HYCOM ESPC-D-V02 u3z/v3z")
    with open(meta, "w") as f:
        json.dump({
            "name": storm["raw_name"], "season": storm["season"], "sid": storm["sid"],
            "window": [str(t0), str(t1)],
            "bbox": [LON0, LAT0, LON1, LAT1],
            "source": src,
        }, f)
    return folder


def hycom_source(storm):
    """Return 'expt93' or 'espc' for the HYCOM download of a storm."""
    t0, t1 = window(storm)
    if t1 <= EXPT93_END:
        return "expt93"
    return "espc"
