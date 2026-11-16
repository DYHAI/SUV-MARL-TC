# Results

The `primary/` directory contains the processed outputs used by the final
400-TC manuscript.

- `surface_main_summary.csv`: surface IPPO summaries by fleet size, speed, and
  deployment radius.
- `surface_filter_summary.csv`: matched action-filter comparison.
- `surface_formation_summary.csv`: ring/line/grid/V formation comparison.
- `classical_surface_summary.csv`: surface traditional-planner results.
- `underwater_ippo_summary.csv`: seven-depth, two-speed IPPO results.
- `classical_underwater_summary.csv`: traditional-planner subsurface results.
- `surface_algorithm_summary.csv`: representative IPPO/SAC/TD3 surface
  comparison.
- `underwater_algorithms_summary.csv`: representative IPPO/SAC/TD3 subsurface
  comparison.
- `*_seed_medians.csv`: the per-training-seed episode medians from which each
  IPPO or algorithm cross-seed summary is calculated.
- `primary_consistency_report.json`: final data, aggregation, environment, and
  manuscript consistency audit.

The final protocol uses 400 TCs, split into 360 training and 40 validation
TCs, with 262 open-ocean validation segments. Surface metrics are episode
medians within each seed followed by cross-seed means. Subsurface current,
geometric-coverage, and current-weighted-coverage metrics use the same
episode-median protocol. Traditional planners use one episode median over the
paired validation episodes.

Model checkpoints, raw evaluation rows, logs, and intermediate files are not
tracked.
