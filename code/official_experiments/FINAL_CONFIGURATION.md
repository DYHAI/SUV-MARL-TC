# Official constrained-dynamics configuration

This file records the official submission protocol. All manuscript results,
tables, figures, and statistical comparisons use this configuration.

## Surface

- `DYNAMICS_MODE=constrained`
- `DYN_SURFACE_CURRENT=0`
- `OBS_HEADING=1`
- The policy command specifies a desired ground velocity.
- Acceleration and heading dynamics determine the effective ground velocity.
- No separate surface-ocean-current term is added to the position update.
- Acceleration: `0.02 m/s^2`
- Deceleration: `0.03 m/s^2`
- Maximum yaw rate: `1 deg/s`
- Minimum turn radius: `200 m`

## Subsurface

- `DYNAMICS_MODE=constrained`
- HYCOM current advection is retained.
- Acceleration: `0.002 m/s^2`
- Deceleration: `0.003 m/s^2`
- Maximum yaw rate: `2 deg/s`
- Minimum turn radius: `100 m`

## Manuscript rule

The additional acceleration, deceleration, heading, yaw-rate, and
minimum-turn-radius equations are described only in the motion-equation
subsection. They are not promoted elsewhere as a separate contribution.

All final decision tables, figures, paired statistics, and traditional-planner
comparisons use the constrained configuration above. Historical
`current_only` and `current_constrained` surface runs are supporting development
data and are not the final manuscript protocol.
