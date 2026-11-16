#!/usr/bin/env python3
"""Build the final 400-TC primary surface/subsurface split (360 train / 40 test)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from code.data_preparation.typhoons_500_split import make_split


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EXTRA = DATA / "segments_extra_kajiki_2007_primary.csv"
OUTPUT = DATA / "segments_primary_400.csv"
SPLIT_OUTPUT = DATA / "typhoon_split_400.json"


def season_for(tag: str) -> int:
    track = pd.read_csv(
        DATA / "selected_typhoons" / tag / "track.csv",
        usecols=["SEASON"],
        nrows=1,
    )
    return int(track.SEASON.iloc[0])


def main() -> None:
    train = pd.read_csv(DATA / "segments_primary_open_ocean_train.csv")
    test = pd.read_csv(DATA / "segments_primary_open_ocean_test.csv")
    extra = pd.read_csv(EXTRA)
    segments = pd.concat([train, test, extra], ignore_index=True)

    tags = sorted(segments.storm.unique())
    if len(tags) != 400:
        raise SystemExit(f"expected 400 TCs, found {len(tags)}")
    season = {tag: season_for(tag) for tag in tags}
    train_tags, test_tags = make_split(
        tags,
        season,
        n_test=40,
        seed=0,
    )
    if len(train_tags) != 360 or len(test_tags) != 40:
        raise SystemExit(
            f"bad split: train={len(train_tags)}, test={len(test_tags)}"
        )
    if set(train_tags) & set(test_tags):
        raise SystemExit("TC leakage between train and test")

    segments["split"] = segments.storm.map(
        lambda tag: "train" if tag in set(train_tags) else "test"
    )
    segments.to_csv(OUTPUT, index=False)

    split = {
        "train": train_tags,
        "test": test_tags,
        "split_by": "storm",
        "rule": "season-stratified, seed=0, n_test=40",
        "n_storms_total": 400,
        "segments_csv": str(OUTPUT.relative_to(ROOT)),
    }
    SPLIT_OUTPUT.write_text(json.dumps(split, indent=2) + "\n")

    summary = {
        "tc_count": len(tags),
        "train_tcs": len(train_tags),
        "test_tcs": len(test_tags),
        "segment_count": len(segments),
        "train_segments": int((segments.split == "train").sum()),
        "test_segments": int((segments.split == "test").sum()),
        "train_years": dict(sorted(Counter(season[t] for t in train_tags).items())),
        "test_years": dict(sorted(Counter(season[t] for t in test_tags).items())),
        "storm_overlap": len(set(train_tags) & set(test_tags)),
        "output_segments": str(OUTPUT.relative_to(ROOT)),
        "output_split": str(SPLIT_OUTPUT.relative_to(ROOT)),
    }
    (DATA / "segments_primary_400.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
