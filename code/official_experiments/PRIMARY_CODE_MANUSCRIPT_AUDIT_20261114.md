# Primary code, formula, and manuscript consistency audit

Date: 2026-11-14

## Data

- 400 TCs: 360 training and 40 validation TCs.
- 3,392 segments: 3,130 training and 262 validation.
- No TC overlap between train and validation.
- Open ocean: every retained 10 min TC-track point is at least 200 km from
  land. The realized minimum is 200.4418 km.
- Formation-start segments (`t0=0`) are excluded. The smallest retained `t0`
  is 38.
- Segments extending beyond the final valid IBTrACS timestamp are excluded.
- Realized segment lengths are 292.1--505.5 km (median 396.05 km).
- Durations are 2.17--97.5 h (median 15.33 h, mean 17.73 h).

## Surface

- One PPO MLP policy controls the whole formation.
- Observation dimension: `3N+4`.
- Action: one joint two-dimensional command, broadcast to all SUVs.
- Relative offsets are projected back exactly after every step. A smoke test
  measured a maximum offset drift of approximately `2.5e-13 km`.
- Main geometry: ring with mean radius 100 km.
- Fleet sizes: 4/8/12/16.
- Speed limits: 12/20 m/s.
- Training deployment: uniform point in the 0--800 km disk around the initial
  TC center.
- Evaluation radii: 100/200/400/600/800 km with random bearing.
- Main matrix: ten training seeds.
- Formation comparison: one matched seed, explicitly a sensitivity study.
- SAC/TD3 algorithm comparison: two representative training seeds.
- Surface tracking statistics: median over validation episodes within each
  seed, then arithmetic mean over seeds.
- Deterministic planners: one episode median over the paired 400-TC validation
  set, with no training-seed variance.

## Subsurface

- Four independent SUVs; no inter-vehicle communication.
- Observation dimension: 19 per SUV.
- Observation: five-frame local current history, normalized propulsion
  velocity, and heading encoded as `[cos(psi), sin(psi)]`.
- Current-speed magnitude is computed from interpolated `u,v` components:
  `sqrt(u^2+v^2)`.
- Main matrix: seven depths, two propulsion limits, ten training seeds.
- Coverage grid: 0.25 degrees, cell area 770.0625 square km.
- Only valid ocean positions contribute to experienced current, coverage, and
  current-weighted coverage.
- Subsurface statistics: arithmetic mean over validation episodes within each
  seed, then arithmetic mean over seeds.

## Manuscript formulas checked

- Platform speed constraints and constrained planar dynamics.
- Surface no-current motion and subsurface current advection.
- Observation dimensions and communication assumptions.
- Unit-disk action projection and rigid formation action sharing.
- PPO clipped objective and surface/subsurface ratio definitions.
- Surface reward, subsurface reward, coastal reward, and action filter.
- Surface fleet-center tracking error, path, mean speed, and peak speed.
- Experienced current, geometric coverage, and current-weighted coverage.

## Corrections made during audit

- Coverage grid changed from an accidental 0.05 degrees to the manuscript's
  0.25 degrees.
- Current speed changed from interpolation of a precomputed speed field to
  interpolation of `u,v` followed by their vector norm.
- Land positions are excluded from underwater current and coverage metrics.
- Current-weighted coverage is computed per episode before aggregation.
- Surface and subsurface aggregation were separated into median and mean
  protocols respectively.
- Primary underwater models are isolated in `models/underwater_primary` so the
  scheduler cannot mistake old 500-TC models for completed 400-TC runs.
- Random control and passive drift were removed from primary underwater
  evaluation because they are not part of the current paper comparison.

## Remaining work

The formula, protocol, and generation-code paths are synchronized. Numerical
paper tables and figures still require replacement after the active 400-TC
training and evaluation matrices complete.

## Executed smoke and numerical checks

- Surface rigid smoke test: observation `(28,)`, action `(2,)`, maximum
  relative-offset drift approximately `2.5e-13 km`.
- Underwater smoke test: observation `(4,19)`, action space `(4,2)`, and
  component-interpolated current-speed error `0.0`.
- Surface median and subsurface mean aggregation pass synthetic two-seed tests.
- Regenerated 0.25-degree traditional underwater outputs have coverage areas
  that are exact integer multiples of 770.0625 square km, and current-weighted
  coverage equals the per-episode product of fleet-mean current and coverage.
