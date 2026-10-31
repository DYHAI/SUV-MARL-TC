# Reproducibility Guide

## Environment

The reported experiments used Python 3.11, Stable-Baselines3 2.9.0,
Gymnasium 1.2.2, PyTorch, NumPy, pandas, xarray, SciPy, and Matplotlib. Policy
training was CPU-based. See `requirements.txt`.

```bash
export PYTHON_BIN="${PYTHON_BIN:-python}"
export UWFLOW_CACHE_DIR="${UWFLOW_CACHE_DIR:-$HOME/.cache/auv_tc/uwflow}"
export MARL_NAS_ROOT="${MARL_NAS_ROOT:-/path/to/union_current_depths}"
```

## Data Integrity

```bash
$PYTHON_BIN audit_hycom_currents.py \
  --local-root data/union_region \
  --nas-root "$MARL_NAS_ROOT"
```

Do not train before all depths used by a configuration are present unless the
experiment is explicitly restricted to complete depths.

## Surface Protocol

- Fleet sizes: 4, 8, 12, 16.
- Speed limits: 12, 20 m/s.
- Training offset: uniformly sampled inside an 800 km disk.
- Validation offsets: 100, 200, 400, 600, 800 km with random bearings.
- Formation: 100 km ring with equal angular separation `360/N`.
- Seeds: 0–9.
- Coastal safeguards: offshore-only deployment, 100 km nearshore buffer,
  100 km action safety margin.

The surface actor observes only shared 10 m wind features and its own normalized
velocity. The TC center enters the training reward only.

```bash
PYTHON_BIN="$PYTHON_BIN" bash run_surface_vmax12_seed10.sh
PYTHON_BIN="$PYTHON_BIN" bash run_surface_distance_recompute.sh
$PYTHON_BIN aggregate_surface_vmax_comparison.py
```

## Underwater Protocol

- Depths: 0, 10, 20, 30, 40, 50, 60 m.
- Speed limits: 1.5, 2.0 m/s.
- Duration: 20 h.
- Initialization: uniform inside 200 km of the segment-end TC center.
- Methods: active IPPO, random control, passive drift.
- No inter-vehicle communication.

The current environment enforces:

1. component-wise action bounds followed by unit-disk projection;
2. `ground velocity = propulsion + HYCOM_advection`;
3. a causal five-frame local-current history updated every step.

Pre-fix weights are excluded from this public repository. After all seven
depths are available, the final pipeline is:

```bash
PYTHON_BIN="$PYTHON_BIN" bash run_uwflow_7depth_completion.sh
$PYTHON_BIN verify_uwflow_completion.py
```

## Evaluation Units

- Each model is first evaluated over the held-out validation dataset.
- Seed-level means are computed for each configuration.
- Reported uncertainty is the sample standard deviation over ten independent
  training seeds.
- Speed-limit comparisons use two-sided paired t-tests on seed-level values.
- The twenty surface fleet/radius comparisons are not corrected for multiple
  testing and are described as exploratory evidence.

## Manuscript

```bash
cd Latex_Yihui_MARL
tectonic -X compile --keep-intermediates --keep-logs main.tex
```

Expected output: `Latex_Yihui_MARL/main.pdf`.

## Offline-Reanalysis Caveat

Spatial and temporal interpolation uses adjacent reanalysis samples. This is an
offline simulation of the reconstructed field at the simulated time; it does
not emulate forecast latency or provide an operational nowcasting test.
