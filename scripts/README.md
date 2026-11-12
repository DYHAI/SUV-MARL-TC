# Experiment Launchers

Set `PYTHON_BIN` when the trained environment is not the active `python`:

```bash
export PYTHON_BIN=/path/to/RL/python
```

Every launcher sources `official_env.sh`, which enables constrained planar
dynamics, disables the surface-current term, and sets the final acceleration,
yaw-rate, and minimum-turn-radius limits.

Main entries:

- `run_surface_vmax12_seed10.sh`: final surface multi-seed training.
- `run_surface_distance_recompute.sh`: final-distance evaluation.
- `run_uwflow_7depth_completion.sh`: seven-depth subsurface training/evaluation.
- `run_classical_surface_matrix.sh`: traditional surface baselines.
- `run_classical_underwater_matrix.sh`: traditional subsurface baselines.
- `run_formation_training.sh`: line/grid/V formation training.
- `run_formation_full_matrix.sh`: formation evaluation matrix.
- `run_algorithm_comparison_training.sh`: IPPO/SAC/TD3 representative runs.

Generated models, logs, and raw evaluation outputs are written under `runs/`.
