# Code

- `common/`: constrained planar dynamics shared by all stages.
- `surface/`: surface fleet environment, IPPO training, and evaluation.
- `underwater/`: fixed-depth current-observation environment and IPPO.
- `baselines/`: potential field, receding-horizon A*, lawnmower, and
  information-greedy planners.
- `formation/`: ring, line, grid, and V formation experiments.
- `algorithm_comparison/`: representative IPPO/SAC/TD3 comparison.
- `analysis/`: final aggregation, tables, and figure generation.
- `data_preparation/`: TC selection, segmentation, downloads, and caches.
- `official_experiments/`: final 400-TC configuration, schedulers, constraints,
  and verification records.

Run maintained modules from the repository root, for example:

```bash
python -m code.official_experiments.validate_dynamics
python -m code.analysis.make_surface_distance_figure
python -m code.analysis.make_traditional_planner_renderings
python -m code.analysis.make_underwater_planner_comparison
```

Large datasets and model checkpoints are read from `data/` and `runs/` and are
not tracked.
