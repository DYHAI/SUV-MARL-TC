#!/usr/bin/env python3
"""Build the primary open-ocean train/test segment files.

Rules:
1. Retain the 200 km offshore filter from segments_open_ocean_200km.csv.
2. Remove segments that extend beyond the final valid IBTrACS timestamp.
3. For both train and test, remove formation-start segments (t0 == 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parent
OPEN = ROOT / "data" / "segments_open_ocean_200km.csv"
SPLIT = ROOT / "data" / "typhoon_split_400.json"
TRACK_ROOT = ROOT / "data" / "selected_typhoons"
ERA_ROOT = ROOT / "data" / "era5_wind"


def valid_last_index(storm: str) -> int:
    track = pd.read_csv(
        TRACK_ROOT / storm / "track.csv",
        parse_dates=["ISO_TIME"],
    ).dropna(subset=["LAT", "LON"])
    with xr.open_dataset(
        ERA_ROOT / storm / "era5_10m_wind_hourly.nc",
        decode_times=True,
    ) as dataset:
        start = pd.Timestamp(dataset.time.values[0])
    return int(
        np.floor(
            (track["ISO_TIME"].max() - start).total_seconds() / 600.0
        )
    )


def main() -> None:
    segments = pd.read_csv(OPEN).reset_index(drop=True)
    split = json.loads(SPLIT.read_text())
    train_tags = set(split["train"])
    test_tags = set(split["test"])

    valid_last = {
        storm: valid_last_index(storm)
        for storm in sorted(set(segments.storm))
    }
    after_track_end = segments.apply(
        lambda row: int(row.t1) > valid_last[row.storm],
        axis=1,
    )
    covered = segments.loc[~after_track_end].copy()

    train = covered[
        covered.storm.isin(train_tags) & (covered.t0 > 0)
    ].reset_index(drop=True)
    test = covered[
        covered.storm.isin(test_tags) & (covered.t0 > 0)
    ].reset_index(drop=True)

    train_path = ROOT / "data" / "segments_primary_open_ocean_train.csv"
    test_path = ROOT / "data" / "segments_primary_open_ocean_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    metadata = {
        "status": "primary_configuration",
        "offshore_criterion_km": 200,
        "tail_filter": "t1 <= final valid IBTrACS index",
        "removed_tail_frozen_segments": int(after_track_end.sum()),
        "training": {
            "path": str(train_path.relative_to(ROOT)),
            "segments": len(train),
            "tcs": int(train.storm.nunique()),
            "exclude_formation_start": True,
        },
        "validation": {
            "path": str(test_path.relative_to(ROOT)),
            "segments": len(test),
            "tcs": int(test.storm.nunique()),
            "exclude_formation_start": True,
        },
    }
    (ROOT / "data" / "segments_primary_open_ocean.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
