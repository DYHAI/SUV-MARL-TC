# Paper

This directory contains the final manuscript source, bibliography, rendered
PDF, and all figures referenced by the manuscript.

## Main files

- `main.tex`: 400-TC manuscript source.
- `main.pdf`: rendered manuscript.
- `main.bib`: bibliography used by `main.tex`.
- `figures/`: all manuscript figures in vector PDF and PNG preview form.

All result tables are embedded directly in `main.tex`.

## Compile

```bash
tectonic main.tex
```

From the repository root:

```bash
make paper
```

## Result provenance

The tables in the manuscript are generated from:

- `results/primary/surface_main_summary.csv`
- `results/primary/surface_filter_summary.csv`
- `results/primary/surface_formation_summary.csv`
- `results/primary/classical_surface_summary.csv`
- `results/primary/underwater_ippo_summary.csv`
- `results/primary/classical_underwater_summary.csv`
- `results/primary/surface_algorithm_summary.csv`
- `results/primary/underwater_algorithms_summary.csv`

The consistency record is
`results/primary/primary_consistency_report.json`.

The final surface matrix contains 80 trained models and 400 primary
radius evaluation conditions. The final subsurface matrix contains 140
trained models over seven depths, two propulsion limits, and ten seeds.

## Final figure set

- `Fig1_Surface_Subsurface_Observation`
- `Fig2_RL_Agent_Environment_Loop`
- `Fig3_Surface_Trajectory_Evolution`
- `Fig4_Surface_Vmax_Distance_Comparison`
- `Fig5_Formation_Comparison`
- `Fig6_Filter_Path_Comparison`
- `FigA1_Surface_Trajectory_Matrix`
- `FigB1_Traditional_Planner_Rendering`
- `FigB2_Limitations_Cases`
- `FigC1_Underwater_Strategy_Trajectories`
- `FigC2_Underwater_Methods_7Depth`

Regeneration scripts are under `../code/analysis/`.
