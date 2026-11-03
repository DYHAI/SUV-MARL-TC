# Multi-AUV Surface--Subsurface Tropical Cyclone Observation

Paper-facing code and processed results for:

**Multi-Agent Reinforcement Learning for Surface--Subsurface Observation of
Tropical Cyclones Using Submersible Unmanned Vessels**

The repository contains only the code used by the final manuscript. Development
scans, superseded experiments, local logs, model checkpoints, and raw
reanalysis files are not included.

## Code

### Data preparation

- `typhoons_2020plus.py`, `typhoons_200.py`, `typhoons_500.py`, and
  `typhoons_500_split.py`: build the 500-TC event list and 450/50 TC-level
  train/test split.
- `download_era5_wind_500.py`: download hourly ERA5 10 m winds.
- `download_hycom_500.py`: download three-hourly HYCOM currents at the seven
  observation depths.
- `audit_hycom_currents.py`, `build_uwflow_cache.py`, and
  `slice_segments_500.py`: audit files, build memory-mapped caches, and create
  300--500 km track segments.

### IPPO training

- `surface_wind_env.py`, `fleet_segment_env.py`, and `fleet_sb3.py`: surface
  observation environment and the shared per-agent actor-critic.
- `train_fleet_segment_actfilter.py`: train the surface IPPO policy with action
  filtering.
- `underwater_flow_env.py`, `underwater_fleet_flow_env.py`, and
  `fleet_sb3.py`: underwater current-observation environment and shared actor.
- `train_uwflow_sb3.py`: train the four-SUV underwater IPPO policy.
- `run_surface_vmax12_seed10.sh` and `run_uwflow_7depth_completion.sh`:
  final multi-seed training launchers.

### Evaluation and comparison

- `eval_deploy_xfer.py` and `run_surface_distance_recompute.sh`: surface
  deployment-error and final-tracking-error evaluation.
- `eval_uwflow_info.py`, `run_uwflow_eval.py`, and
  `summarize_uwflow_500.py`: underwater evaluation and aggregation.
- `classical_planning_baselines.py`, `eval_classical_surface.py`, and
  `eval_classical_underwater.py`: potential-field, A*, lawnmower, and
  information-greedy baselines.
- `aggregate_classical_baselines.py`: aggregate traditional-planner results.
- `run_formation_training.sh`, `run_formation_full_matrix.sh`,
  `eval_surface_robustness.py`, and `aggregate_formation_full_matrix.py`:
  ring/line/grid training and the 12/20 m/s x five-radius evaluation matrix.
- `train_offpolicy_representative.py` and
  `run_algorithm_comparison_training.sh`: matched IPPO/SAC/TD3 runs.

### Figure and table generation

- `aggregate_surface_vmax_comparison.py` and `make_surface_control_ablation_table.py`.
- `aggregate_underwater_method_ablation.py` and
  `analyze_tc_speed_envelope.py`.
- `make_formation_comparison_figure.py`.
- `Latex_Yihui_MARL/tools/` contains the final figure-generation scripts.

### Manuscript

- `Latex_Yihui_MARL/main.tex`: complete manuscript with all tables embedded.
- `Latex_Yihui_MARL/main.bib`: bibliography.
- `Latex_Yihui_MARL/main.pdf`: compiled manuscript.
- `Latex_Yihui_MARL/figures/`: figure files used by the manuscript.

## Data

The data sources are ERA5, HYCOM, and IBTrACS:

- ERA5: hourly 10 m wind, `0.25 deg` grid.
- HYCOM: three-hourly currents at 0/10/20/30/40/50/60 m.
- IBTrACS: best-track position, time, wind, and pressure.

The final 500-TC archive yields 7,248 segments from 495 usable TCs. The split is
at TC level: 450 training TCs (445 with usable segments, 6,480 segments) and 50
held-out TCs (768 segments). Segment lengths are 300--500 km with a 250 km
stride.

Raw NetCDF files are not stored in Git. Download reconstruction scripts are
listed above; CDS credentials must be configured locally and must never be
committed.

## Training

Install the environment:

```bash
python -m pip install -r requirements.txt
```

Surface scenario:

- 4/8/12/16 SUVs on a 100 km ring with `360/N` angular spacing.
- 12 or 20 m/s maximum surface speed.
- Training offsets sampled uniformly inside an 800 km radius.
- Evaluation at fixed 100/200/400/600/800 km offsets.
- Ten training seeds per fleet-size/speed configuration.

Example surface run:

```bash
export SPLIT_JSON="$PWD/data/typhoon_split_500.json"
export SEGMENTS_CSV="$PWD/data/segments_500.csv"
export FLEET_VMAX=12
python train_fleet_segment_actfilter.py \
  2000000 ppo_fleet/surface_ring_n8_v12_s0.zip 0.8 8 800
```

Subsurface scenario:

- Four SUVs and seven depths from 0 to 60 m.
- 1.5 or 2.0 m/s maximum propulsion speed.
- Ten active-policy seeds for each depth/speed configuration.
- The same episodes are used for active IPPO, random control, and passive drift.

Example underwater run:

```bash
export SPLIT_JSON="$PWD/data/typhoon_split_500.json"
export SEGMENTS_CSV="$PWD/data/segments_500.csv"
python train_uwflow_sb3.py \
  2000000 ppo_uwflow_500_d30_v1.5_rC_20h_sb3_s0 1.5 C 30 20 0 0.25 200 4
```

Formation and traditional-planner experiments:

```bash
bash run_formation_training.sh
bash run_formation_full_matrix.sh
bash run_classical_surface_matrix.sh
bash run_classical_underwater_matrix.sh
```

## Results

Processed result files:

- `results/surface/`: 10-seed surface summaries, raw seed results, and paired
  tests for 12 versus 20 m/s.
- `results/underwater/`: final seven-depth underwater evaluation summary and
  report.
- `classical_eval/surface_all_methods_summary.csv` and
  `classical_eval/underwater_all_methods_summary.csv`: traditional-planner
  comparisons.
- `classical_eval/surface_pairwise_tests.csv` and
  `classical_eval/underwater_pairwise_tests.csv`: paired tests.
- `robustness_eval/formation_full_matrix_summary.csv`: ring/line/grid results.
- `Latex_Yihui_MARL/main.pdf`: final manuscript and all tables.

## Main Results

- Surface IPPO obtains the lowest final tracking error and time-averaged
  distance in all 40 matched fleet-size/speed/radius configurations against the
  four traditional planners.
- For the 12-versus-20 m/s surface comparison, 20 m/s has the lower final error
  in all 20 fleet-size/radius combinations; 19 of 20 paired time-averaged
  comparisons are significant at `p < 0.05`.
- Underwater, potential field gives the highest mean experienced current, A*
  gives the shortest path, and IPPO gives the largest nominal and
  current-weighted spatial coverage. No method dominates every metric.
- Across the formation matrix, the ring has the lowest mean final error, while
  the line has the shortest path and the grid preserves a broader footprint.
- Relative to passive drift, active IPPO increases mean experienced current by
  2.1--5.4%, path length by 3.6--7.3 times, and nominal coverage by 3.3--5.7
  times across the evaluated depth/speed settings.

## Citation

See [`CITATION.cff`](CITATION.cff).
