#!/usr/bin/env python3
"""Aggregate primary surface and subsurface results using episode medians.

Surface protocol: compute one value per validation episode, take the median
within each training seed and condition, then average seed medians.
Subsurface protocol: compute one value per validation episode, take the median
within each training seed and condition, then average seed medians.

This script intentionally does not reuse the legacy aggregators that call
``DataFrame.mean`` directly on episode rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "runs"
OUT = ROOT / "results" / "primary"

METRICS = {
    "final_centre_dist": "final_tracking_error_km",
    "centre_dist": "mean_tracking_error_km",
    "path": "path_km",
    "vmean": "mean_speed_ms",
    "max_prop_speed_ms": "max_speed_ms",
    "final_wind_speed_ms": "final_wind_speed_ms",
}
UNDERWATER_METRICS = {
    "mean_current_ms": "mean_current_speed_ms",
    "coverage_area_km2": "geometric_coverage_km2",
    "path_km": "path_km",
    "current_weighted_coverage": "current_weighted_coverage",
}


def _metric_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {
        source: target
        for source, target in METRICS.items()
        if source in frame.columns
    }


def seed_medians(
    episodes: pd.DataFrame,
    condition_cols: list[str],
    training_seed_col: str,
    metric_cols: dict[str, str],
) -> pd.DataFrame:
    """Return one median per condition and training seed."""
    if episodes.empty:
        return pd.DataFrame()
    required = set(condition_cols) | {training_seed_col} | set(metric_cols)
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    grouped = (
        episodes.groupby(
            [*condition_cols, training_seed_col],
            dropna=False,
        )[list(metric_cols)]
        .median()
        .reset_index()
        .rename(columns={training_seed_col: "training_seed", **metric_cols})
    )
    return grouped


def aggregate_per_seed_values(
    per_seed: pd.DataFrame,
    condition_cols: list[str],
    metric_cols: dict[str, str],
) -> pd.DataFrame:
    """Mean and sample std of seed medians for each condition."""
    if per_seed.empty:
        return pd.DataFrame()
    metric_names = list(metric_cols.values())
    rows = []
    for keys, group in per_seed.groupby(condition_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(condition_cols, keys))
        row["n_training_seeds"] = int(group["training_seed"].nunique())
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_bundle(
    name: str,
    per_seed: pd.DataFrame,
    summary: pd.DataFrame,
    granularity: str = "medians",
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(OUT / f"{name}_seed_{granularity}.csv", index=False)
    summary.to_csv(OUT / f"{name}_summary.csv", index=False)


def aggregate_classical_surface() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = EXP / "results" / "classical_primary" / "surface"
    paths = sorted(source.glob("*_test_*.csv"))
    if not paths:
        return pd.DataFrame(), pd.DataFrame()
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    episodes = pd.concat(frames, ignore_index=True)
    # Deterministic planners have no training seed. All validation episodes
    # form one distribution and are summarized by a single median.
    episodes["training_seed"] = 0
    condition_cols = ["planner", "n_auv", "vmax_ms", "radius_km"]
    metric_cols = {
        "final_centre_distance_km": "final_tracking_error_km",
        "centre_distance_avg_km": "mean_tracking_error_km",
        "path_km": "path_km",
        "final_wind_speed_ms": "final_wind_speed_ms",
    }
    per_seed = seed_medians(
        episodes, condition_cols, "training_seed", metric_cols
    )
    summary = aggregate_per_seed_values(per_seed, condition_cols, metric_cols)
    _write_bundle("classical_surface", per_seed, summary)

    method_summary = (
        summary.groupby("planner", as_index=False)[
            [
                "final_tracking_error_km_mean",
                "mean_tracking_error_km_mean",
                "path_km_mean",
                "final_wind_speed_ms_mean",
            ]
        ]
        .mean()
    )
    method_summary.to_csv(
        OUT / "classical_surface_by_method.csv", index=False
    )
    return per_seed, summary


def aggregate_classical_underwater() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = Path(
        os.environ.get(
            "CLASSICAL_UNDERWATER_DIR",
            EXP
            / "results"
            / "classical_primary"
            / "underwater_native_hycom_cellwise",
        )
    )
    paths = sorted(source.glob("*_test_d*_v*_s10.csv"))
    if not paths:
        return pd.DataFrame(), pd.DataFrame()
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "current_weighted_coverage" not in frame.columns:
            frame["current_weighted_coverage"] = (
                frame["mean_current_ms"] * frame["coverage_area_km2"]
            )
        frame["source_file"] = path.name
        frames.append(frame)
    episodes = pd.concat(frames, ignore_index=True)
    # Deployment seeds are validation realizations, not trained models.
    episodes["training_seed"] = 0
    condition_cols = ["planner", "depth_m", "vmax_ms"]
    per_seed = seed_medians(
        episodes, condition_cols, "training_seed", UNDERWATER_METRICS
    )
    summary = aggregate_per_seed_values(
        per_seed, condition_cols, UNDERWATER_METRICS
    )
    _write_bundle(
        "classical_underwater", per_seed, summary, granularity="medians"
    )
    return per_seed, summary


def aggregate_underwater_ippo(
    directory: Path,
    name: str = "underwater_ippo",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(
        directory.glob("eval_info_info_d*_start200_20h_v*.csv")
    )
    if not paths:
        return pd.DataFrame(), pd.DataFrame()
    frames = []
    for path in paths:
        match = re.search(
            r"_d(\d+(?:\.\d+)?)_start200_20h_v(\d+(?:\.\d+)?)", path.name
        )
        if not match:
            raise ValueError(f"cannot parse depth/speed from {path.name}")
        frame = pd.read_csv(path)
        active = frame[frame["policy"].astype(str).str.startswith("uw_")].copy()
        if active.empty:
            continue
        active["depth_m"] = float(match.group(1))
        active["vmax_ms"] = float(match.group(2))
        seeds = active["policy"].astype(str).str.extract(r"_s(\d+)$")[0]
        if seeds.isna().any():
            raise ValueError(f"cannot parse training seed in {path.name}")
        active["training_seed"] = seeds.astype(int)
        active["mean_current_ms"] = active["mean_V"]
        active["coverage_area_km2"] = active["cov_area_km2"]
        if "current_weighted_coverage" not in active.columns:
            active["current_weighted_coverage"] = (
                active["mean_current_ms"] * active["coverage_area_km2"]
            )
        active["source_file"] = path.name
        frames.append(active)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    episodes = pd.concat(frames, ignore_index=True)
    condition_cols = ["depth_m", "vmax_ms"]
    per_seed = seed_medians(
        episodes, condition_cols, "training_seed", UNDERWATER_METRICS
    )
    summary = aggregate_per_seed_values(
        per_seed, condition_cols, UNDERWATER_METRICS
    )
    _write_bundle(name, per_seed, summary, granularity="medians")
    return per_seed, summary


def aggregate_underwater_algorithms(
    directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(directory.glob("*algorithm_comparison_primary.csv"))
    if not paths:
        return pd.DataFrame(), pd.DataFrame()
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        labels = frame["policy"].astype(str)
        algorithm = pd.Series("", index=frame.index, dtype=object)
        algorithm[labels.str.startswith("uw_")] = "IPPO"
        algorithm[
            labels.str.contains("_sac_", regex=False)
            | labels.str.startswith("sac_")
        ] = "SAC"
        algorithm[
            labels.str.contains("_td3_", regex=False)
            | labels.str.startswith("td3_")
        ] = "TD3"
        selected = frame[algorithm != ""].copy()
        selected["algorithm"] = algorithm[algorithm != ""]
        seeds = selected["policy"].astype(str).str.extract(r"_s(\d+)$")[0]
        if seeds.isna().any():
            raise ValueError(f"cannot parse algorithm training seed in {path.name}")
        selected["training_seed"] = seeds.astype(int)
        selected["mean_current_ms"] = selected["mean_V"]
        selected["coverage_area_km2"] = selected["cov_area_km2"]
        if "current_weighted_coverage" not in selected.columns:
            selected["current_weighted_coverage"] = (
                selected["mean_current_ms"] * selected["coverage_area_km2"]
            )
        selected["source_file"] = path.name
        frames.append(selected)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    episodes = pd.concat(frames, ignore_index=True)
    condition_cols = ["algorithm", "depth_m", "vmax_ms"]
    per_seed = seed_medians(
        episodes, condition_cols, "training_seed", UNDERWATER_METRICS
    )
    summary = aggregate_per_seed_values(
        per_seed, condition_cols, UNDERWATER_METRICS
    )
    _write_bundle(
        "underwater_algorithms", per_seed, summary, granularity="medians"
    )
    return per_seed, summary


def _parse_xfer_name(path: Path, family: str) -> dict[str, object]:
    stem = path.stem
    metadata: dict[str, object] = {"source_file": path.name}
    seed_match = re.search(r"_s(\d+)_", stem)
    radius_match = re.search(r"_d(\d+)(?:_b[0-9.]+)?_seg$", stem)
    fleet_match = re.search(r"(?:^|_)n(\d+)_", stem)
    speed_match = re.search(r"_v([0-9.]+)_", stem)
    if seed_match:
        metadata["training_seed"] = int(seed_match.group(1))
    if radius_match:
        metadata["radius_km"] = int(radius_match.group(1))
    if fleet_match:
        metadata["n_auv"] = int(fleet_match.group(1))
    if speed_match:
        metadata["vmax_ms"] = float(speed_match.group(1))
    if family == "formation":
        formation_match = re.match(r"(ring|line|grid|v)_", stem)
        if formation_match:
            metadata["formation"] = formation_match.group(1)
        metadata["n_auv"] = 8
    elif family == "algorithm":
        algorithm_match = re.match(r"(ippo|sac|td3)_", stem)
        if algorithm_match:
            metadata["algorithm"] = algorithm_match.group(1).upper()
    elif family == "filter":
        beta_match = re.search(r"_b([0-9.]+)_seg$", stem)
        if beta_match:
            metadata["filter_beta"] = float(beta_match.group(1))
    return metadata


def aggregate_xfer_directory(
    directory: Path,
    name: str,
    family: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(directory.glob("*_seg.csv"))
    if not paths:
        return pd.DataFrame(), pd.DataFrame()
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        for key, value in _parse_xfer_name(path, family).items():
            frame[key] = value
        frames.append(frame)
    episodes = pd.concat(frames, ignore_index=True)
    if "training_seed" not in episodes.columns:
        episodes["training_seed"] = 0
    if family == "formation" and "formation" not in episodes.columns:
        raise ValueError(f"cannot parse formation names in {directory}")
    if family == "algorithm" and "algorithm" not in episodes.columns:
        raise ValueError(f"cannot parse algorithm names in {directory}")
    condition_cols = ["n_auv", "vmax_ms", "radius_km"]
    if family == "formation":
        condition_cols.insert(0, "formation")
    elif family == "algorithm":
        condition_cols.insert(0, "algorithm")
    elif family == "filter":
        condition_cols.insert(0, "filter_beta")
    metric_cols = _metric_columns(episodes)
    per_seed = seed_medians(
        episodes, condition_cols, "training_seed", metric_cols
    )
    summary = aggregate_per_seed_values(per_seed, condition_cols, metric_cols)
    _write_bundle(name, per_seed, summary)
    return per_seed, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=(
            "all",
            "classical",
            "classical_underwater",
            "main",
            "formation",
            "algorithm",
            "filter",
            "underwater",
            "underwater_algorithm",
        ),
        default="all",
    )
    args = parser.parse_args()

    report: dict[str, object] = {}
    if args.only in ("all", "classical"):
        per_seed, summary = aggregate_classical_surface()
        report["classical_surface"] = {
            "seed_rows": len(per_seed),
            "condition_rows": len(summary),
        }
    if args.only in ("all", "classical_underwater"):
        per_seed, summary = aggregate_classical_underwater()
        report["classical_underwater"] = {
            "seed_rows": len(per_seed),
            "condition_rows": len(summary),
        }

    requested = {
        "main": (
            EXP / "results" / "surface_primary_circle" / "segments",
            "surface_main",
            "main",
        ),
        "formation": (
            EXP
            / "results"
            / "surface_rigid_formation_primary"
            / "formation"
            / "segments",
            "surface_formation",
            "formation",
        ),
        "algorithm": (
            EXP / "results" / "algorithms_primary" / "surface" / "segments",
            "surface_algorithm",
            "algorithm",
        ),
        "filter": (
            EXP / "results" / "surface_primary_filter" / "segments",
            "surface_filter",
            "filter",
        ),
    }
    for family, (directory, name, kind) in requested.items():
        if args.only not in ("all", family):
            continue
        per_seed, summary = aggregate_xfer_directory(
            directory, name, kind
        )
        report[name] = {
            "source": str(directory),
            "available": directory.is_dir() and bool(list(directory.glob("*_seg.csv"))),
            "seed_rows": len(per_seed),
            "condition_rows": len(summary),
        }

    if args.only in ("all", "underwater"):
        directory = EXP / "results" / "underwater_primary"
        per_seed, summary = aggregate_underwater_ippo(directory)
        report["underwater_ippo"] = {
            "source": str(directory),
            "available": directory.is_dir()
            and bool(
                list(directory.glob("eval_info_info_d*_start200_20h_v*.csv"))
            ),
            "seed_rows": len(per_seed),
            "condition_rows": len(summary),
        }
    if args.only in ("all", "underwater_algorithm"):
        directory = EXP / "results" / "algorithms_primary" / "underwater"
        per_seed, summary = aggregate_underwater_algorithms(directory)
        report["underwater_algorithms"] = {
            "source": str(directory),
            "available": directory.is_dir()
            and bool(list(directory.glob("*algorithm_comparison_primary.csv"))),
            "seed_rows": len(per_seed),
            "condition_rows": len(summary),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
