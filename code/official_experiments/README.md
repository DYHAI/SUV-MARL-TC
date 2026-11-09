# Official constrained-dynamics experiments

This directory records the official experimental configuration and audit
status for the manuscript. Every reported result uses the constrained dynamics
in `FINAL_CONFIGURATION.md`.

## Dynamics

The policy issues a two-dimensional unit-disk command at each 10 min control
interval. The executed propulsion state follows:

1. bounded acceleration toward the commanded speed;
2. bounded deceleration when the commanded speed decreases;
3. a continuous heading state;
4. a bounded yaw rate;
5. a minimum turn radius.

The equations are integrated with 60 s substeps. The actor receives two
additional observation values, `cos(heading)` and `sin(heading)`, so the
heading state is fully observable.

The official surface configuration treats the policy command as a desired
ground velocity and does not add HYCOM current. Acceleration and heading
dynamics determine the effective surface velocity. The official subsurface
configuration retains HYCOM advection.

| Parameter | Surface | Subsurface |
|---|---:|---:|
| acceleration | 0.02 m/s^2 | 0.002 m/s^2 |
| deceleration | 0.03 m/s^2 | 0.003 m/s^2 |
| maximum yaw rate | 1 deg/s | 2 deg/s |
| minimum turn radius | 200 m | 100 m |
| internal integration step | 60 s | 60 s |

The final experiments set:

```text
DYNAMICS_MODE=constrained
DYN_SURFACE_CURRENT=0
OBS_HEADING=1
```

The instantaneous kinematic mode remains available only for development and
is not a manuscript result.

## Main matrices

- Surface: 4/8/12/16 SUVs, 12/20 m/s, ten seeds, and 100/200/400/600/800 km
  evaluation radii.
- Subsurface: four SUVs, depths 0/10/20/30/40/50/60 m, 1.5/2.0 m/s, and ten
  seeds.
- Training: the same 450/50 TC split and approximately two million environment
  steps per model as the manuscript.
- Surface comparisons: ring/line/grid formations, action filtering,
  IPPO/SAC/TD3, and four traditional planners.
- Subsurface comparisons: active IPPO, random control, passive drift,
  IPPO/SAC/TD3, and four traditional planners.

## Completion status

The official matrices are complete:

- 80 surface IPPO models;
- 140 subsurface IPPO models;
- four surface formation models;
- two surface and two subsurface SAC/TD3 representative baselines;
- all surface and subsurface traditional-planner evaluations.

`validate_dynamics.py` performs direct numerical checks of the motion model.
`verification_report.json` records the completed evaluation counts and observed
speed, acceleration, and yaw-rate limits. Model checkpoints and raw evaluation
rows are retained in the private development archive.
