#!/usr/bin/env bash
# Continue both curl-full-dipole ICs another 100 steps (100->200), resuming
# from step_100_final.h5, same cross_scale=1e-3, same thread split as
# run_step.sh.
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -e

(
    export PYTHONPATH=$PWD

    date
    echo "=== step L1R60 + uniform R180 curl-full-dipole ICs, 100->200, plot every 25 ==="
    python3 -u "$CWD/step_curlfull_dipole_l1_wider_margin.py" \
        --snapshot /bk/curlfull_dipole_l1_wider_margin_stepped/step_100_final.h5 \
        --n-steps 100 --plot-every 25 --n-threads 6 \
        > "$CWD/step_curlfull_dipole_l1_wider_margin_2.log" 2>&1 &
    PID1=$!
    python3 -u "$CWD/step_curlfull_dipole_uniform_r180.py" \
        --snapshot /bk/curlfull_dipole_uniform_r180_stepped/step_100_final.h5 \
        --n-steps 100 --plot-every 25 --n-threads 3 \
        > "$CWD/step_curlfull_dipole_uniform_r180_2.log" 2>&1 &
    PID2=$!

    RC1=0; RC2=0
    wait "$PID1" || RC1=$?
    wait "$PID2" || RC2=$?
    date
    echo "=== stepping done: l1_wider_margin rc=$RC1  uniform_r180 rc=$RC2 ==="

) 1> >(tee "$CWD/run_step2.sh.out") 2> >(tee "$CWD/run_step2.sh.err" >&2)
