#!/usr/bin/env python3
"""Filter TC track segments whose entire path stays offshore by a given buffer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from code.surface.fleet_segment_env import land_interp
from code.data_preparation.slice_segments_500 import load_track10


ROOT = Path(__file__).resolve().parent
SPLIT = ROOT / "data" / "typhoon_split_500.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data" / "segments_500.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "segments_open_ocean_200km.csv",
    )
    parser.add_argument("--buffer-km", type=float, default=200.0)
    parser.add_argument(
        "--exclude-genesis",
        action="store_true",
        help="exclude segments whose deployment starts at t0=0",
    )
    args = parser.parse_args()

    segments = pd.read_csv(args.input)
    split = json.loads(SPLIT.read_text())
    keep_rows: list[dict] = []
    for storm, group in segments.groupby("storm", sort=False):
        lon_t, lat_t, _, _ = load_track10(storm)
        for _, row in group.iterrows():
            t0 = int(row["t0"])
            t1 = int(row["t1"])
            pts = np.stack([lat_t[t0:t1 + 1], lon_t[t0:t1 + 1]], axis=1)
            if len(pts) == 0:
                continue
            offshore = float(np.min(land_interp()(pts)))
            if offshore >= args.buffer_km and (
                not args.exclude_genesis or int(row["t0"]) > 0
            ):
                item = row.to_dict()
                item["min_land_distance_km"] = offshore
                keep_rows.append(item)

    filtered = pd.DataFrame(keep_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output, index=False)
    train = filtered[filtered.storm.isin(split["train"])]
    test = filtered[filtered.storm.isin(split["test"])]
    metadata = {
        "source": str(args.input),
        "output": str(args.output),
        "criterion": (
            "minimum distance to land over every 10 min TC track point "
            f"in the segment is at least {args.buffer_km:g} km"
        ),
        "buffer_km": args.buffer_km,
        "exclude_genesis": bool(args.exclude_genesis),
        "segments": len(filtered),
        "storms_with_segments": int(filtered.storm.nunique()),
        "train_segments": len(train),
        "train_storms_with_segments": int(train.storm.nunique()),
        "test_segments": len(test),
        "test_storms_with_segments": int(test.storm.nunique()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
