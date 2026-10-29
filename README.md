# Multi-AUV Surface--Subsurface Tropical Cyclone Observation

This public repository contains the final simulation, training, evaluation,
analysis, and manuscript-generation code used in the paper
**Multi-Agent Reinforcement Learning for Surface--Subsurface Observation of
Tropical Cyclones Using Submersible Unmanned Vessels**. The study evaluates a
parameter-sharing IPPO policy for two connected tasks:

1. **Surface tracking:** 4/8/12/16 submersible unmanned vessels (SUVs) observe
   10 m winds and move from a forecast-uncertain deployment position toward the
   TC center and high-wind region.
2. **Subsurface observation:** four communication-free SUVs sample ocean
   currents from 0 to 60 m and compare active IPPO, random control, and passive
   drift.

The environments are event-specific simulations driven by ERA5, HYCOM, and
IBTrACS reanalysis/best-track products. The final manuscript is in
[`Latex_Yihui_MARL/`](Latex_Yihui_MARL/).

## Current Status

- The surface 12/20 m/s matrix is complete for 4/8/12/16 SUVs, ten seeds, and
  deployment radii of 100/200/400/600/800 km.
- The execution-filter ablation includes final-distance and trajectory metrics.
- The final-distance recomputation for the two-panel surface figure is being
  completed.
- The underwater implementation was corrected to enforce unit-disk action
  projection and to update the five-frame current history at every step.
- All seven underwater depths must therefore be retrained. Pre-correction
  weights are excluded from this public repository.
- Corrected training has started for the complete 0/30/50 m layers; the public
  result files will be updated only after the full seven-depth matrix is frozen.
- The seven-depth data archive and retraining pipeline are in progress.
  Manuscript preview figures are explicitly marked and must be replaced by the
  final matrix before submission.

## Key Experimental Design

### Surface task

| Item | Setting |
|---|---|
| Observation | Shared 10 m ERA5 wind speed and `u/v` components |
| Fleet sizes | 4, 8, 12, 16 SUVs |
| Formation | 100 km ring with equal spacing `360°/N` |
| Speed limits | 12 and 20 m/s |
| Training deployment | Random position inside an 800 km disk |
| Validation deployment | Fixed radii 100, 200, 400, 600, 800 km |
| Decision interval | 10 min |
| Seeds | 0–9 |
| Coastal safeguards | Offshore-only deployment, 100 km buffer, 100 km action filter |
| Policy | Parameter-sharing PPO/IPPO with a team-average reward |

### Subsurface task

| Item | Setting |
|---|---|
| Observation | Local current speed plus `u/v`, five causal frames, own velocity |
| Fleet size | Four SUVs per depth |
| Depths | 0, 10, 20, 30, 40, 50, 60 m |
| Speed limits | 1.5 and 2.0 m/s resultant propulsion speed |
| Methods | Active IPPO, random control, passive drift |
| Mission duration | 20 h |
| Initialization | Uniformly inside 200 km of the segment-end TC center |
| Communication | None between underwater vehicles |

## Data

The repository does not include the full reanalysis archive.

| Product | Variables | Native resolution | Source |
|---|---|---|---|
| ERA5 | 10 m `u/v` wind | 0.25°, hourly | [Copernicus CDS](https://cds.climate.copernicus.eu/) |
| HYCOM | `water_u`, `water_v` | ~0.04° × 0.08°, 3-hourly | [HYCOM](https://www.hycom.org/) |
| IBTrACS | Center, time, wind, pressure | 3/6-hourly best track | [NCEI](https://www.ncei.noaa.gov/products/international-best-track-archive) |

The target archive contains 500 western North Pacific TCs, split at event
level into 450 training and 50 test TCs. See
[`docs/DATA.md`](docs/DATA.md) for acquisition, layout, and validation commands.

Create CDS credentials in your home directory; never commit them:

```text
# ~/.cdsapirc
url: https://cds.climate.copernicus.eu/api
key: <YOUR_KEY>
```

## Installation

Python 3.11 is recommended.

```bash
git clone git@github.com:DYHAI/SUV-MARL-TC.git
cd SUV-MARL-TC

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Set environment-specific paths:

```bash
export PYTHON_BIN="$(command -v python)"
export MARL_NAS_ROOT="/path/to/archived/hycom_depth_files"
export UWFLOW_CACHE_DIR="$HOME/.cache/auv_tc/uwflow"
```

## Expected Data Layout

```text
data/
├── union_region/
│   └── CHAMPI_2015_2015286/
│       ├── hycom_uv_0m.nc
│       ├── hycom_uv_10m.nc
│       ├── ...
│       └── hycom_uv_60m.nc
├── era5_wind/<TC>/era5_10m_wind_hourly.nc
├── typhoon_split_500.json
└── segments_500.csv
```

NAS files may be exposed through symlinks in `data/union_region/`. Building an
mmap cache before training is strongly recommended:

```bash
$PYTHON_BIN build_uwflow_cache.py 30 0.25 3
```

## Reproduce the Manuscript

### 1. Audit the current archive

```bash
$PYTHON_BIN audit_hycom_currents.py \
  --local-root data/union_region \
  --nas-root "$MARL_NAS_ROOT"
```

### 2. Train the final surface models

```bash
PYTHON_BIN="$PYTHON_BIN" bash run_surface_vmax12_seed10.sh
```

Evaluation and result aggregation:

```bash
PYTHON_BIN="$PYTHON_BIN" bash run_surface_distance_recompute.sh
$PYTHON_BIN aggregate_surface_vmax_comparison.py
$PYTHON_BIN make_surface_control_ablation_table.py
```

### 3. Train and evaluate the seven-depth underwater matrix

For complete depths that can start before the whole archive is finished:

```bash
DEPTHS="0 30 50" MAXJOBS=3 \
  PYTHON_BIN="$PYTHON_BIN" bash run_uwflow_complete_depths_first.sh
```

After all seven depths are complete:

```bash
PYTHON_BIN="$PYTHON_BIN" bash run_uwflow_7depth_completion.sh
```

The pipeline builds caches, trains the remaining models, evaluates active
IPPO/random/drift policies, and regenerates the final seven-depth outputs.

### 4. Build the manuscript

```bash
cd Latex_Yihui_MARL
tectonic -X compile --keep-intermediates --keep-logs main.tex
```

The generated PDF is `Latex_Yihui_MARL/main.pdf`.

## Main Results Available Now

Across the twenty matched fleet-size/deployment-radius comparisons, the
time-averaged distance is lower for the 20 m/s policy in all 20 cases. The
fleet-averaged 12-minus-20 m/s difference increases from 10.1 km at a 100 km
deployment radius to 84.2 km at 800 km.

The 12 m/s policy uses shorter paths and lower mean speed. In the execution
filter ablation, the four-fleet mean final distance decreases from 346.4 to
292.6 km after filtering, while path length, mean speed, and maximum speed also
decrease.

Final underwater conclusions are intentionally deferred until all seven depths
have been retrained with the corrected implementation.

## Parameters Implemented in Code

| Parameter | Surface | Underwater |
|---|---:|---:|
| Decision interval | 10 min | 10 min |
| Speed limit | 12/20 m/s | 1.5/2.0 m/s |
| Action dimension | 2 | 2 |
| Observation dimension | `3N + 2` | 17 |
| Reward clip | `[-5, 2]` | `[-5, 5]` |
| Action smoothing | `β = 0.8` | None |
| Smoothing penalty | 0.05 | None |
| Current normalization | — | 1.0 m/s |
| Current reward cap | — | 2.0 |
| Propulsion penalty | — | 0.05 |
| Current-alignment weight | — | 0.5 |
| Land penalty | 20 | 20 |
| Nearshore buffer/weight | 100 km / 8 | 100 km / 8 |
| PPO rollout | 2048 × 4 envs | 2048 × 4 envs |
| Mini-batch / epochs | 256 / 10 | 256 / 10 |
| Learning rate | `3e-4` | `3e-4` |
| Discount / GAE | 0.99 / 0.95 | 0.99 / 0.95 |
| PPO clip / value clip | 0.2 / 0.2 | 0.2 / 0.2 |
| Entropy / value loss | 0.01 / 0.5 | 0.01 / 0.5 |
| Max gradient norm | 0.5 | 0.5 |
| Training steps | 2,000,000 | 2,000,000 |

The complete reward equations and implementation notes are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and the manuscript
appendix.

## Repository Layout

```text
.
├── Latex_Yihui_MARL/          Final manuscript, figures, tables, bibliography
├── docs/                      Data and reproducibility documentation
├── results/surface/           Processed ten-seed surface result tables
├── fleet_segment_env.py       Surface multi-SUV environment
├── underwater_fleet_flow_env.py
├── train_fleet_segment_actfilter.py
├── train_uwflow_sb3.py
├── eval_deploy_xfer.py
├── eval_uwflow_info.py
├── aggregate_surface_vmax_comparison.py
├── audit_hycom_currents.py
└── run_*.sh                  Training, evaluation, and completion pipelines
```

The repository is curated for the paper. Development logs, intermediate
ablations, downloaded data, and obsolete model weights are intentionally
excluded.

## Important Limitations

- This is an offline reanalysis simulation. Interpolated fields use adjacent
  reanalysis samples to reconstruct a field at the simulated time; real-time
  forecast latency is not emulated.
- The vehicle model is a horizontal point-mass/kinematic model. It does not
  include 6-DOF attitude, added mass, hydrodynamic drag, wave loads, ballast
  transitions, thruster dynamics, or actuator delay.
- The reported framework has not been validated in field trials and does not
  quantify downstream data-assimilation or operational forecast impact.
- Pre-correction underwater weights are not distributed in this public
  repository.

## Security

`cds_api_keys.json` is no longer tracked and is ignored. If it was previously
committed, rotate the corresponding CDS credential; removing the current file
does not purge it from Git history.

## Citation and License

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). No open-source
license is declared in this repository. Contact the corresponding author for
reuse permissions.
