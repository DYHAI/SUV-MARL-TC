# Classical planner comparison results

## Evaluation scope

Both planners use the same 500-TC dataset and 450/50 TC-level split as the
manuscript. Classical planners require no training. Their fixed causal rules
were evaluated on the same held-out physical scenarios:

- Surface: 4/8/12/16 SUVs x 12/20 m/s x
  100/200/400/600/800 km, with all 768 test segments.
- Subsurface: 0/10/20/30/40/50/60 m x 1.5/2.0 m/s, with 500 paired episodes
  per setting (50 test TCs x 10 deployment indices).
- Subsurface deployment uses NumPy seeds 100--109, exactly matching
  `eval_uwflow_info.py`.
- Neither planner receives the TC center or the complete future environmental
  field.

## Surface aggregate

Values are averaged over the 40 surface configurations.

| Method | Final error (km) | Time-averaged distance (km) | Path per SUV (km) |
|---|---:|---:|---:|
| Parameter-sharing IPPO | **341.7** | **349.1** | **463.6** |
| Potential field | 416.5 | 405.9 | 727.6 |
| Receding-horizon A* | 459.0 | 430.4 | 824.9 |

IPPO has the lower final error, time-averaged distance, and path length in all
40 comparisons against each baseline. Mean final-error reductions are 74.8 km
relative to the potential-field planner and 117.3 km relative to A*.

## Subsurface aggregate

Values are averaged over the 14 depth-speed configurations.

| Method | Mean current (m/s) | Nominal coverage (km2) | Path (km) |
|---|---:|---:|---:|
| Parameter-sharing IPPO | 0.409 | **3630** | 126.4 |
| Potential field | **0.449** | 1976 | 112.5 |
| Receding-horizon A* | 0.428 | 2275 | 94.8 |

The conventional planners obtain higher local current exposure in every
setting, but their coverage is only 54--63% of the IPPO coverage after
aggregation. Using the per-episode current-weighted coverage
`mean_current * coverage`, IPPO is 1.40--1.71 times higher than the two
baselines in all 14 settings. IPPO paths are 7--39% longer.

The two baseline CSV sets were generated with the same deployment seeds as the
MARL evaluation. Paired tests over the 500 underwater episodes are stored in
`underwater_pairwise_tests.csv`.

## Manuscript use

The manuscript contains only one compact comparison table
(`tables/classical_planner_comparison.tex`) and one short results paragraph.
The diagnostic comparison figures remain in this directory and are not included
in the paper.

## Verification

Run:

```bash
/home/ding/main/miniconda3/envs/RL/bin/python verify_classical_baselines.py
```

Expected:

- 16/16 surface raw files, 61,440 episodes.
- 28/28 underwater raw files, 14,000 episodes.
- 120 surface summary rows.
- 42 underwater summary rows.
- 140 underwater paired-test rows.
