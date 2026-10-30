# Manuscript Work Status (2026-10-28)

## Completed

- Final working directory is this folder.
- Figure 1 regenerated with transparent cropped vessel images; all four vessel
  icons have the same leftward heading and no rectangular image background.
- Figure 1 panel labels are equal-sized and enlarged; colorbars use a separate
  row and no longer overlap their labels.
- Table 2 shortened:
  - `High speed scenario`
  - `Current speed scenario`
  - `0--60 m (7-layer)`
- ERA5, HYCOM, and IBTrACS source portals are line-broken footnotes in the Data
  section.
- Wenlei Peng added as third author with affiliations a/b; Jingsong Yang remains
  last and corresponding.
- Training supervision is explicit: center distance appears only in the reward;
  validation removes reward and center coordinate.
- Code availability lists the project repository, software versions, and
  workstation configuration (CPU-only policy training).
- Underwater three-method ten-seed figure is generated. The underwater Table 3
  was deliberately removed from the manuscript on request.
- Low-pass numerical paragraph is present.
- The combined low-pass/on-land-penalty table is generated and included:
  6.60% on-land steps without the on-land reward versus 5.94% with
  `P_land=20`.
- Figure 1 is moved into the Introduction; its caption notes that four vehicles
  are illustrative and that experiments extend to 8/12/16.
- The method explains why SAC/TD3 were not compared: replay-buffer memory grows
  with joint multi-agent state/action size and parallel environments.

## Background runs

- `run_surface_vmax12_seed10.sh` completed all 4/8/12/16 fleets, seeds 0--9,
  and radii 100/200/400/600/800 km.
- The no-on-land-penalty run completed at 6.60% on-land AUV-steps, compared
  with 5.94% for `P_land=20`.
- All 200 evaluations completed and the aggregator parsed 400/400
  seed-level 12/20 m/s evaluations.
- When `logs_vmax12_seed10/ALL.done` appears, the watcher runs
  `aggregate_surface_vmax_comparison.py` and writes:
  - `tables/surface_vmax_dynamic.tex`
  - `tables/surface_vmax_operational.tex`
  - `figures/surface_vmax_10seed_metrics.pdf`
  - `surface_vmax_10seed_summary.csv`
  - `surface_vmax_10seed_paired_tests.csv`

## Finalization status

1. Final compile and visual audit are complete.
2. The clean final-only GitHub commit is being prepared.
