#!/usr/bin/env python3
"""Verify the primary 400-TC configuration and paper-facing metric protocol."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "runs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP))

os.environ.update(
    {
        "RIGID_FORMATION": "1",
        "DYNAMICS_MODE": "constrained",
        "DYN_SURFACE_CURRENT": "0",
        "OBS_HEADING": "1",
        "DEPLOY_OCEAN_ONLY": "1",
        "DEPLOY_MIN_COAST_KM": "100",
        "SAFETY_MARGIN_KM": "100",
        "SPLIT_JSON": str(ROOT / "data" / "typhoon_split_400.json"),
        "SEGMENTS_CSV": str(ROOT / "data" / "segments_primary_400.csv"),
    }
)

from code.surface.train_fleet_segment_actfilter import FlatFleetSegmentEnvActFilter
from code.underwater.underwater_fleet_flow_env import UnderwaterFleetFlowEnv
from code.underwater.underwater_coverage import (
    native_hycom_cells,
    native_hycom_coverage_area,
    native_hycom_weighted_coverage,
)
from code.analysis.aggregate_primary_median import (
    aggregate_per_seed_values,
    seed_means,
    seed_medians,
)


def require_text(path: Path, values: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [value for value in values if value not in text]
    if missing:
        raise AssertionError(f"{path}: missing {missing}")


def verify_data_and_config() -> dict[str, object]:
    split = json.loads((ROOT / "data" / "typhoon_split_400.json").read_text())
    segments = pd.read_csv(ROOT / "data" / "segments_primary_400.csv")
    config = json.loads((EXP / "PRIMARY_CONFIGURATION.json").read_text())
    assert len(split["train"]) == 360
    assert len(split["test"]) == 40
    assert segments.storm.nunique() == 400
    assert len(segments) == 3392
    assert segments.groupby("split").size().to_dict() == {
        "test": 262,
        "train": 3130,
    }
    assert not set(split["train"]) & set(split["test"])
    assert float(segments.min_land_distance_km.min()) >= 200.0
    assert int((segments.t0 <= 0).sum()) == 0
    assert "200 km" in config["open_ocean_definition"]["criterion"]
    assert config["traditional_algorithms"]["surface_evaluation"].endswith(
        "all 262 held-out segments"
    )
    assert "subsurface_metric_aggregation" in config["traditional_algorithms"]
    metrics = config["underwater_observation_metrics"]
    assert metrics["coverage_grid_lat_deg"] == 0.04
    assert metrics["coverage_grid_lon_deg"] == 0.08
    assert "mean observed current speed" in metrics[
        "current_weighted_coverage"
    ]
    assert "rigid_ring_n4_v20" not in config
    return {
        "tc_count": 400,
        "train_tcs": 360,
        "validation_tcs": 40,
        "segments": 3392,
        "train_segments": 3130,
        "validation_segments": 262,
        "minimum_land_distance_km": float(
            segments.min_land_distance_km.min()
        ),
    }


def verify_default_paths() -> dict[str, bool]:
    checks = {
        "surface/train_fleet_segment_actfilter.py": (
            "typhoon_split_400.json",
            "segments_primary_400.csv",
        ),
        "underwater/train_uwflow_sb3.py": (
            "typhoon_split_400.json",
            "segments_primary_400.csv",
        ),
        "surface/eval_deploy_xfer.py": (
            "typhoon_split_400.json",
            "segments_primary_400.csv",
        ),
        "underwater/eval_uwflow_info.py": (
            "typhoon_split_400.json",
            "segments_primary_400.csv",
        ),
        "baselines/eval_classical_surface.py": (
            "typhoon_split_400.json",
            "segments_primary_400.csv",
        ),
        "baselines/eval_classical_underwater.py": (
            "typhoon_split_400.json",
            "segments_primary_400.csv",
        ),
    }
    for filename, values in checks.items():
        require_text(ROOT / "code" / filename, values)
    require_text(
        ROOT / "code/official_experiments/run_main_training.py",
        (
            "RIGID_FORMATION",
            "DYN_UNDERWATER_MODEL_SUBDIR",
            "typhoon_split_400.json",
            "segments_primary_400.csv",
        ),
    )
    return {filename: True for filename in checks}


def verify_manuscript() -> dict[str, bool]:
    text_path = ROOT / "paper" / "main.tex"
    require_text(
        text_path,
        (
            "400-TC",
            "360 training TCs",
            "40 held-out",
            "at least 200 km from land",
            "rigid formation",
            "one joint formation action",
            "independent action",
            "median across validation episodes",
            "arithmetic mean over validation episodes",
            "native HYCOM grid cell",
            "0.04^\\circ",
            "0.08^\\circ",
        ),
    )
    text = text_path.read_text(encoding="utf-8")
    forbidden = (
        "500-TC",
        "500 TCs",
        "768 held-out",
        "50 unseen TCs",
        "does not support 400 TCs",
        "formation geometry is not rigidly preserved",
    )
    found = [value for value in forbidden if value in text]
    if found:
        raise AssertionError(f"legacy manuscript phrases remain: {found}")
    return {"required_phrases": True, "legacy_phrases_absent": True}


def verify_pipeline_and_generators() -> dict[str, bool]:
    require_text(
        ROOT / "scripts/run_algorithm_comparison_training.sh",
        (
            "results/surface_rigid_formation_primary/ALL.done",
            "DYN_ALGO_SEEDS=\"${DYN_ALGO_SEEDS:-0 1}\"",
            "INCLUDE_IDLE_RANDOM=0",
            "models/underwater_primary",
        ),
    )
    require_text(
        ROOT / "code/official_experiments/run_main_training.py",
        (
            "ALGORITHM_COMPARISON_TRAINING_PRIMARY.done",
            "DYN_UNDERWATER_MODEL_SUBDIR",
        ),
    )
    plotting = {
        ROOT / "code/analysis/make_surface_distance_figure.py": (
            "primary_median",
            "surface_main_summary.csv",
        ),
        ROOT / "code/analysis/make_trajectory_result_figures.py": (
            "segments_primary_400.csv",
            "surface_primary_circle",
            "underwater_primary",
        ),
        ROOT / "code/formation/make_formation_comparison_figure.py": (
            "segments_primary_400.csv",
            "surface_rigid_formation_primary",
            "formation_rigid_primary",
        ),
        ROOT / "code/analysis/make_filter_path_comparison.py": (
            "segments_primary_400.csv",
            "surface_primary_circle",
        ),
        ROOT / "code/analysis/make_traditional_planner_renderings.py": (
            "segments_primary_400.csv",
            "surface_primary_circle",
            "underwater_primary",
        ),
    }
    for path, values in plotting.items():
        require_text(path, values)
        if "segments_500.csv" in path.read_text(encoding="utf-8"):
            raise AssertionError(f"legacy path remains in {path}")
    require_text(
        ROOT / "code/analysis/rebuild_primary_figures_tables.py",
        (
            "current_weighted_coverage",
            "tab:classical_planners",
            "tab:formation_comparison",
            "tab:underwater_ippo_main",
            "tab:planner_stratified",
        ),
    )
    require_text(
        ROOT / "code/analysis/make_traditional_planner_renderings.py",
        (
            "native_hycom_coverage_area",
            "native_hycom_weighted_coverage",
        ),
    )
    require_text(
        ROOT / "code/analysis/sync_primary_tables_to_main.py",
        ("PRIMARY_TABLE_START", "PRIMARY_TABLE_END"),
    )
    require_text(
        ROOT / "code/underwater/eval_uwflow_info.py",
        (
            'COVERAGE_DEG = float(os.environ.get("COVERAGE_DEG", "0.25"))',
            "native_hycom_cells",
            "native_hycom_weighted_coverage",
            "ocean_valid",
            "np.hypot(final_u, final_v)",
            "INCLUDE_IDLE_RANDOM",
        ),
    )
    require_text(
        ROOT / "code/underwater/underwater_fleet_flow_env.py",
        ("np.hypot(cu, cv)", "np.hypot(cu_new, cv_new)"),
    )
    return {"scheduler": True, "plot_generators": True, "evaluation": True}


def one_segment(storm: str) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "data" / "segments_primary_400.csv")
    return frame[frame.storm == storm].head(1).reset_index(drop=True)


def offsets(env) -> np.ndarray:
    lon0 = float(np.mean(env.lons))
    lat0 = float(np.mean(env.lats))
    return np.stack(
        [
            (env.lons - lon0) * 111.0 * np.cos(np.deg2rad(lat0)),
            (env.lats - lat0) * 111.0,
        ],
        axis=1,
    )


def verify_environment_smoke() -> dict[str, object]:
    split = json.loads((ROOT / "data" / "typhoon_split_400.json").read_text())
    storm = split["train"][0]
    segment = one_segment(storm)

    surface = FlatFleetSegmentEnvActFilter(
        [storm], segment, n_auv=8, beta=0.8
    )
    obs, _ = surface.reset(seed=0)
    before = offsets(surface.inner)
    obs, reward, _term, _trunc, _info = surface.step(
        np.array([0.2, -0.1], dtype=np.float32)
    )
    after = offsets(surface.inner)
    offset_drift = float(np.max(np.abs(after - before)))
    assert obs.shape == (28,)
    assert surface.action_space.shape == (2,)
    assert offset_drift < 1e-9

    underwater = UnderwaterFleetFlowEnv(
        [storm],
        segment,
        n_auv=4,
        vmax=1.5,
        reward_mode="C",
        N=1,
        use_frame_stack=True,
        frame_k=5,
        depth_m=30,
        start_radius=200,
        grid_deg=0.25,
        cov_deg=0.25,
        duration_h=20,
    )
    uw_obs, _ = underwater.reset(seed=0)
    u = underwater._at2m("u", underwater.lats, underwater.lons)
    v = underwater._at2m("v", underwater.lats, underwater.lons)
    uw_obs, _reward, _term, _trunc, info = underwater.step(
        np.zeros((4, 2), dtype=np.float32)
    )
    norm_error = float(
        np.max(np.abs(info["spd_at"] - np.hypot(u, v)))
    )
    assert uw_obs.shape == (4, 19)
    assert underwater.action_space.shape == (4, 2)
    assert abs(underwater._cell_area_km2 - 770.0625) < 1e-9
    native_cells = native_hycom_cells([20.0], [130.0])
    native_area = native_hycom_coverage_area(native_cells)
    native_weighted = native_hycom_weighted_coverage(
        [20.0], [130.0], [0.5]
    )
    assert abs(native_weighted - 0.5 * native_area) < 1e-9
    assert norm_error < 1e-12
    return {
        "surface_observation_shape": list(obs.shape),
        "surface_action_shape": list(surface.action_space.shape),
        "surface_offset_drift_km": offset_drift,
        "underwater_observation_shape": list(uw_obs.shape),
        "underwater_action_shape": list(underwater.action_space.shape),
        "underwater_current_norm_error": norm_error,
        "training_env_grid_cell_area_km2": underwater._cell_area_km2,
        "native_hycom_coverage_cell_area_km2": native_area,
        "native_hycom_cellwise_weighted_coverage": native_weighted,
    }


def verify_aggregation() -> dict[str, object]:
    sample = pd.DataFrame(
        {
            "condition": ["x"] * 6,
            "training_seed": [0, 0, 0, 1, 1, 1],
            "metric": [1.0, 2.0, 9.0, 3.0, 4.0, 8.0],
        }
    )
    mapping = {"metric": "metric"}
    medians = seed_medians(
        sample, ["condition"], "training_seed", mapping
    )
    means = seed_means(sample, ["condition"], "training_seed", mapping)
    assert medians.metric.tolist() == [2.0, 4.0]
    assert means.metric.tolist() == [4.0, 5.0]
    assert aggregate_per_seed_values(
        medians, ["condition"], mapping
    ).metric_mean.tolist() == [3.0]
    assert aggregate_per_seed_values(
        means, ["condition"], mapping
    ).metric_mean.tolist() == [4.5]
    return {
        "surface_seed_medians": medians.metric.tolist(),
        "subsurface_seed_means": means.metric.tolist(),
        "surface_cross_seed_mean": 3.0,
        "subsurface_cross_seed_mean": 4.5,
    }


def main() -> None:
    report = {
        "data_and_config": verify_data_and_config(),
        "default_paths": verify_default_paths(),
        "manuscript": verify_manuscript(),
        "pipeline_and_generators": verify_pipeline_and_generators(),
        "environment_smoke": verify_environment_smoke(),
        "aggregation": verify_aggregation(),
        "status": "pass",
    }
    output = EXP / "results" / "primary_consistency_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
