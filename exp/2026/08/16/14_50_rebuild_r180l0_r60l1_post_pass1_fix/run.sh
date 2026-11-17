#!/usr/bin/env bash
# Rebuild the L1R60 (l1_wider_margin_active) and uniform-R180 ICs from
# scratch under the current code -- now including 2026-08-14/15's
# pass1/pass2 coefficient-parity fix (see `now`, doc/dst_amr.md) on top of
# the 2026-08-13 stale-proxy fix these two checkpoints last baked in --
# then step both 100 steps (first look -- shorter than the original 300,
# see report.md), plotting every 25. Both configs run IN PARALLEL, L1R60 (the
# bigger/junction-bearing config) gets 6 threads, uniform R180 gets 3 (sum=9,
# user-specified split, 8-core/16-thread machine). Each parallel pair's own
# stdout/stderr goes to its own log file (interleaving two processes in one
# stream would be unreadable); this script's own date/section markers go to
# run.sh.out.
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -e

(
    export PYTHONPATH=$PWD

    date
    echo "=== build L1R60 + uniform R180 ICs in parallel (L1R60: 6 threads, uniform R180: 3 threads) ==="
    python3 -u "$CWD/build_l1_wider_margin_active.py" \
        --out /bk/l1_wider_margin_active_ic220_pp1fix.h5 --n-threads 6 \
        > "$CWD/build_l1_wider_margin_active.log" 2>&1 &
    PID1=$!
    python3 -u "$CWD/build_uniform_r180_active.py" \
        --out /bk/uniform_r180_active_ic220_pp1fix.h5 --n-threads 3 \
        > "$CWD/build_uniform_r180_active.log" 2>&1 &
    PID2=$!

    RC1=0; RC2=0
    wait "$PID1" || RC1=$?
    wait "$PID2" || RC2=$?
    date
    echo "=== builds done: l1_wider_margin_active rc=$RC1  uniform_r180_active rc=$RC2 ==="
    if [ "$RC1" -ne 0 ] || [ "$RC2" -ne 0 ]; then
        echo "a build failed, aborting before stepping"
        exit 1
    fi

    echo "=== step both in parallel (L1R60: 6 threads, uniform R180: 3 threads), 100 steps, plot every 25 ==="
    python3 -u "$CWD/step_l1_wider_margin_active.py" \
        --ic /bk/l1_wider_margin_active_ic220_pp1fix.h5 \
        --out-dir /bk/l1_wider_margin_active_pp1fix_stepped \
        --plot-dir "$CWD/plots/L1_R60" \
        --n-steps 100 --plot-every 25 --n-threads 6 \
        > "$CWD/step_l1_wider_margin_active.log" 2>&1 &
    PID3=$!
    python3 -u "$CWD/step_uniform_r180_active.py" \
        --ic /bk/uniform_r180_active_ic220_pp1fix.h5 \
        --out-dir /bk/uniform_r180_active_pp1fix_stepped \
        --plot-dir "$CWD/plots/uniform_r180" \
        --n-steps 100 --plot-every 25 --n-threads 3 \
        > "$CWD/step_uniform_r180_active.log" 2>&1 &
    PID4=$!

    RC3=0; RC4=0
    wait "$PID3" || RC3=$?
    wait "$PID4" || RC4=$?
    date
    echo "=== stepping done: l1_wider_margin_active rc=$RC3  uniform_r180_active rc=$RC4 ==="

    date
    echo "=== done ==="

) 1> >(tee "$CWD/run.sh.out") 2> >(tee "$CWD/run.sh.err" >&2)
