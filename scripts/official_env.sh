#!/bin/bash
# Official constrained-dynamics configuration for the final manuscript.

export DYNAMICS_MODE=constrained
export DYN_SURFACE_CURRENT=0
export OBS_HEADING=1

export DYN_SURFACE_ACC_MS2=0.02
export DYN_SURFACE_DEC_MS2=0.03
export DYN_SURFACE_YAW_DEG_S=1.0
export DYN_SURFACE_MIN_TURN_RADIUS_M=200

export DYN_SUB_ACC_MS2=0.002
export DYN_SUB_DEC_MS2=0.003
export DYN_SUB_YAW_DEG_S=2.0
export DYN_SUB_MIN_TURN_RADIUS_M=100
