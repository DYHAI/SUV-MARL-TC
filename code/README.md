# Code

- `data_preparation/`: ERA5, HYCOM, IBTrACS, segmentation, and cache tools.
- `surface/`: surface environment, IPPO training, and deployment evaluation.
- `underwater/`: subsurface environment, IPPO training, and strategy evaluation.
- `baselines/`: traditional planning baselines and paired comparisons.
- `formation/`: ring, line, and grid formation experiments.
- `algorithm_comparison/`: representative IPPO, SAC, and TD3 comparison.
- `analysis/`: final tables, figures, and result summaries.
- `common/`: shared constrained planar dynamics.
- `official_experiments/`: official configuration, equations audit, and
  verification report.

Run executable modules from the repository root with `python -m ...`; the
launchers in `scripts/` do this automatically.
