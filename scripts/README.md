# Experiment Launchers

Set `PYTHON_BIN` when the trained environment is not the active `python`:

```bash
export PYTHON_BIN=/path/to/RL/python
```

Every launcher sources `official_env.sh`, which enables constrained planar
dynamics, disables the surface-current term, and sets the final acceleration,
yaw-rate, and minimum-turn-radius limits.

Main entries:

- `run_surface_vmax12_seed10.sh`: final 12/20 m/s, 4/8/12/16-SUV, ten-seed
  surface matrix.
- `run_surface_distance_recompute.sh`: final radius evaluation and aggregation.
- `run_uwflow_7depth_completion.sh`: seven-depth, two-speed subsurface matrix.
- `run_classical_surface_matrix.sh`: traditional surface planners.
- `run_classical_underwater_matrix.sh`: traditional subsurface planners.
- `run_formation_training.sh`: line/grid/V formation training.
- `run_formation_full_matrix.sh`: formation evaluation matrix.
- `run_algorithm_comparison_training.sh`: representative IPPO/SAC/TD3 runs.

Generated models, logs, and raw evaluation outputs are written under `runs/`.
