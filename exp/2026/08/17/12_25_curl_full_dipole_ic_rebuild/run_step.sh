#!/usr/bin/env bash
# Step both curl-full-dipole ICs 100 steps in parallel (L1R60: 6 threads,
# uniform R180: 3 threads, sum=9, same convention as the 08-16 14:50 rebuild's
# run.sh), plotting every 25 steps (u, log10|u|, poloidal-field-lines +
# toroidal-vy panels).
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -e

(
    export PYTHONPATH=$PWD

    date
    echo "=== step L1R60 + uniform R180 curl-full-dipole ICs in parallel, 100 steps, plot every 25 ==="
    python3 -u "$CWD/step_curlfull_dipole_l1_wider_margin.py" \
        --n-steps 100 --plot-every 25 --n-threads 6 \
        > "$CWD/step_curlfull_dipole_l1_wider_margin.log" 2>&1 &
    PID1=$!
    python3 -u "$CWD/step_curlfull_dipole_uniform_r180.py" \
        --n-steps 100 --plot-every 25 --n-threads 3 \
        > "$CWD/step_curlfull_dipole_uniform_r180.log" 2>&1 &
    PID2=$!

    RC1=0; RC2=0
    wait "$PID1" || RC1=$?
    wait "$PID2" || RC2=$?
    date
    echo "=== stepping done: l1_wider_margin rc=$RC1  uniform_r180 rc=$RC2 ==="

) 1> >(tee "$CWD/run_step.sh.out") 2> >(tee "$CWD/run_step.sh.err" >&2)
