#!/usr/bin/env bash
# Stress test: cross_scale=1000 (vs production 1e-3) for 10 steps from t=0,
# plotting every step, to see whether step_cross produces a visibly large
# effect (merging, or blowup -- either is informative) on a short timescale.
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -e

(
    export PYTHONPATH=$PWD

    date
    echo "=== cross_scale=1000 stress test, 10 steps, plot every step ==="
    python3 -u "$CWD/step_curlfull_dipole_l1_wider_margin.py" \
        --cross-scale 1000 --n-steps 10 --plot-every 1 --n-threads 6 \
        --out-dir /bk/curlfull_dipole_l1_wider_margin_crossscale1000 \
        --plot-dir "$CWD/plots/l1_wider_margin_crossscale1000" \
        > "$CWD/step_curlfull_dipole_l1_wider_margin_crossscale1000.log" 2>&1 &
    PID1=$!
    python3 -u "$CWD/step_curlfull_dipole_uniform_r180.py" \
        --cross-scale 1000 --n-steps 10 --plot-every 1 --n-threads 3 \
        --out-dir /bk/curlfull_dipole_uniform_r180_crossscale1000 \
        --plot-dir "$CWD/plots/uniform_r180_crossscale1000" \
        > "$CWD/step_curlfull_dipole_uniform_r180_crossscale1000.log" 2>&1 &
    PID2=$!

    RC1=0; RC2=0
    wait "$PID1" || RC1=$?
    wait "$PID2" || RC2=$?
    date
    echo "=== stress test done: l1_wider_margin rc=$RC1  uniform_r180 rc=$RC2 ==="

) 1> >(tee "$CWD/run_step_crossscale1000.sh.out") 2> >(tee "$CWD/run_step_crossscale1000.sh.err" >&2)
