#!/bin/bash
set -ex
CWD=$(pwd)
export PYTHONPATH=$PWD

# INVALID first attempt — kept for methodology record (see report.md).
# R=20 without --scale-override lets the calibrated neutron scale shrink
# in sim-units too (scale is tied to R when uncalibrated), leaving only
# 27 leaf cells / ~2 cells across the diameter resolving the object.
# The resulting H/H0 swings (250-700x) are IC under-resolution noise,
# not a real AMR signal. Do not reuse this invocation for anything.
python3 -u tst/substrate/adaptive/refined_neutron_decay.py \
    --R 20 --max-level 8 --n-steps 5 --report 1 --cross-scales 1e-3 --m0 2.49 --n-threads 10 \
    > "$CWD/L8_INVALID_no_scale_override.log" 2>&1

# Valid matched-footprint runs: --scale-override/--scale-ref-level holds the
# spheromak at the same physical (sim-unit) size regardless of --max-level,
# per refined_neutron_decay.py's own calibration mechanism. R=20 (not the
# calibrated R=100 production domain) keeps the mesh cheap -- R=100/L8 alone
# took >8min to not even finish one step (5.58M leaves); R=20 finishes
# 10 steps in well under a minute (~144k-180k leaves).

python3 -u tst/substrate/adaptive/refined_neutron_decay.py \
    --R 20 --max-level 6 --n-steps 10 --report 1 --cross-scales 1e-3 --m0 2.49 --n-threads 10 \
    --scale-override 1.3606e-3 --scale-ref-level 8 \
    > "$CWD/L6_matched_footprint.log" 2>&1

python3 -u tst/substrate/adaptive/refined_neutron_decay.py \
    --R 20 --max-level 8 --n-steps 10 --report 1 --cross-scales 1e-3 --m0 2.49 --n-threads 10 \
    --scale-override 1.3606e-3 --scale-ref-level 8 \
    > "$CWD/L8_matched_footprint.log" 2>&1
