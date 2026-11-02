# Classical planning baseline evaluation

The baselines are deliberately causal: they do not receive the TC center,
future reanalysis states, or the complete environmental field.

## Surface protocol

- Dataset: the same 500-TC dataset and 450/50 TC-level split as the manuscript.
- Validation: all 768 held-out segments.
- Fleets: 4, 8, 12, and 16 SUVs.
- Speed limits: 12 and 20 m/s.
- Deployment-center radii: 100, 200, 400, 600, and 800 km.
- Deployment is the same 100 km ring with a segment-dependent random bearing
  used by `eval_deploy_xfer.py` with deployment seed index 0.
- Coastal rejection sampling, the 100 km soft nearshore buffer, the 100 km
  surface safety filter, circular action projection, and the 0.8 action
  low-pass filter are applied exactly as in the final MARL surface evaluation.

## Subsurface protocol

- Dataset: the same 50 held-out TCs.
- Episodes: 50 TCs x 10 deployment seeds = 500 paired episodes for each
  depth and speed.
- Depths: 0, 10, 20, 30, 40, 50, and 60 m.
- Speed limits: 1.5 and 2.0 m/s.
- NumPy deployment seeds are 100--109, matching `eval_uwflow_info.py`; the
  CSV retains both the actual seed and the 0--9 deployment index.
- There is no inter-vehicle communication. The same circular action
  projection and landfall termination as the MARL evaluation are used.

## Planners

1. `PotentialFieldPlanner`: local gradient ascent on observed wind/current
   speed plus offshore repulsion inside the coastal buffer.
2. `RecedingHorizonAStarPlanner`: online causal value map, ridge-regularized
   local-gradient target selection, and a three-step A* route. The first action
   is executed, with the route refreshed every 15 decision intervals.

The planners are not trained. Their parameters are fixed before evaluation and
no test episode or true TC state is used for tuning.

## Commands

```bash
MAX_JOBS=6 ./run_classical_surface_matrix.sh
PLANNER=potential DEPTHS='0 10 20' MAX_JOBS=2 BATCH_STORMS=5 \
  ./run_classical_underwater_tail.sh
PLANNER=astar DEPTHS='0 10 20 30' MAX_JOBS=1 BATCH_STORMS=5 \
  ./run_classical_underwater_tail.sh
./aggregate_classical_baselines.py
```
