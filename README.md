# Multi-AUV Surface--Subsurface Tropical Cyclone Observation

Paper-facing code, processed results, and manuscript for:

**Multi-Agent Reinforcement Learning for Surface--Subsurface Observation of
Tropical Cyclones Using Submersible Unmanned Vessels**

Repository: <https://github.com/DYHAI/SUV-MARL-TC>

## Repository Layout

```text
code/
  data_preparation/     ERA5/HYCOM/IBTrACS preparation and segmentation
  surface/              surface environment, IPPO training, and evaluation
  underwater/           subsurface environment, IPPO training, and evaluation
  baselines/            traditional planners and comparison aggregation
  formation/            ring/line/grid formation experiments
  algorithm_comparison/ representative IPPO/SAC/TD3 comparison
  analysis/             final tables, figures, and summary calculations
scripts/                runnable experiment launchers
paper/                  LaTeX source, bibliography, figures, and compiled PDF
results/                final processed result tables and reports
runs/                   generated models, logs, and raw evaluation outputs
```

Only paper-facing source and final processed results are tracked. Raw NetCDF
files, model checkpoints, transient logs, and prior development experiments are
not included.

## Code

### Data preparation

- `code/data_preparation/typhoons_2020plus.py`
- `code/data_preparation/typhoons_200.py`
- `code/data_preparation/typhoons_500.py`
- `code/data_preparation/typhoons_500_split.py`
- `code/data_preparation/download_era5_wind_500.py`
- `code/data_preparation/download_hycom_500.py`
- `code/data_preparation/audit_hycom_currents.py`
- `code/data_preparation/slice_segments_500.py`
- `code/data_preparation/build_uwflow_cache.py`

### Surface task

- `code/surface/surface_wind_env.py`: ERA5 10 m wind and coastline handling.
- `code/surface/fleet_segment_env.py`: multi-SUV segmented surface environment.
- `code/surface/fleet_sb3.py`: shared per-agent actor-critic.
- `code/surface/train_fleet_segment_actfilter.py`: surface IPPO training.
- `code/surface/eval_deploy_xfer.py`: deployment-error evaluation.
- `code/surface/eval_surface_robustness.py`: formation and perturbation evaluation.

### Subsurface task

- `code/underwater/underwater_flow_env.py`: HYCOM current loading and caching.
- `code/underwater/underwater_fleet_flow_env.py`: four-SUV subsurface environment.
- `code/underwater/train_uwflow_sb3.py`: subsurface IPPO training.
- `code/underwater/eval_uwflow_info.py`: active/random/drift evaluation.
- `code/underwater/run_uwflow_eval.py`: batch evaluation driver.
- `code/underwater/summarize_uwflow_500.py`: processed result aggregation.
- `code/underwater/eval_underwater_robustness.py`: subsurface perturbation tests.

### Comparisons and analysis

- `code/baselines/`: potential-field, receding-horizon A*, lawnmower, and
  information-greedy planners, plus comparison aggregation.
- `code/formation/`: ring, line, and grid formation training/evaluation tools.
- `code/algorithm_comparison/`: matched IPPO/SAC/TD3 experiments.
- `code/analysis/`: final result tables and manuscript figure generators.

## Data

The final experiments use:

- ERA5 hourly 10 m winds on a 0.25 degree grid.
- HYCOM three-hourly currents at 0, 10, 20, 30, 40, 50, and 60 m.
- IBTrACS best-track position, time, wind, and pressure.

The 500-TC archive contains 495 usable TCs and 7,248 segments. The split is made
at TC level: 450 training TCs, including 445 with usable segments and 6,480
segments, and 50 held-out TCs with 768 segments. Segments are 300--500 km long
with 250 km stride.

Raw reanalysis files are not stored in Git. Configure CDS credentials locally;
never commit credentials. Data roots default to `data/` and can be overridden
through the environment variables used by the environment modules.

## Training

Install the Python environment:

```bash
python -m pip install -r requirements.txt
```

Run a smoke check:

```bash
make check
```

Surface training example:

```bash
export SPLIT_JSON="$PWD/data/typhoon_split_500.json"
export SEGMENTS_CSV="$PWD/data/segments_500.csv"
export FLEET_VMAX=12
python -m code.surface.train_fleet_segment_actfilter \
  2000000 surface_ring_n8_v12_s0.zip 0.8 8 800
```

The model is written under `runs/surface/models/`. Final multi-seed surface
training is launched with:

```bash
make surface-train
```

Subsurface training example:

```bash
export SPLIT_JSON="$PWD/data/typhoon_split_500.json"
export SEGMENTS_CSV="$PWD/data/segments_500.csv"
python -m code.underwater.train_uwflow_sb3 \
  ppo_uwflow_500_d30_v1.5_rC_20h_sb3_s0 1.5 C 30 20 0 0.25 200 4
```

The model is written under `runs/underwater/models/`. The complete seven-depth
matrix is launched with:

```bash
make underwater-train
```

Comparison workflows:

```bash
make classical
make classical-results
make formation
make formation-results
```

## Results

- `results/surface/`: surface summaries, seed-level metrics, and paired tests.
- `results/underwater/`: seven-depth strategy summaries and reports.
- `results/baselines/`: surface/subsurface traditional-planner results and
  paired tests.
- `results/formation/`: ring/line/grid formation matrix summary.
- `paper/main.pdf`: compiled manuscript with all active tables.

## Main Findings

- Surface IPPO obtains the lowest final tracking error and time-averaged
  distance across the 40 matched fleet-size, speed, and deployment-radius
  configurations.
- A 20 m/s surface limit reduces tracking error relative to 12 m/s, while
  12 m/s shortens trajectories and lowers mean speed.
- Underwater, potential field gives the highest mean experienced current, A*
  gives the shortest path, and IPPO gives the largest nominal and
  current-weighted spatial coverage. No method dominates every metric.
- Across the formation matrix, the ring has the lowest mean final error, the
  line has the shortest path, and the grid preserves a broader footprint.
- Relative to passive drift, active IPPO increases experienced current speed,
  path length, and sampled spatial coverage across the evaluated depth/speed
  settings.

## Manuscript

```bash
make paper
```

The final files are under [`paper/`](paper/).

## Citation

See [`CITATION.cff`](CITATION.cff).
