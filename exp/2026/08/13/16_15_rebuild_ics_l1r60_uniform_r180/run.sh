#!/usr/bin/env bash
# Rebuild the L1R60 (l1_wider_margin_active) and uniform-R180 ICs from
# scratch under the current code (stale-proxy fix baked in at construction,
# not just self-healed on load), then step both 300 steps, plotting every
# 100. Both configs run IN PARALLEL, 4 threads each (8-core/16-thread
# machine -- sum stays at 8, not 16). Each parallel pair's own stdout/stderr
# goes to its own log file (interleaving two processes in one stream would
# be unreadable); this script's own date/section markers go to run.sh.out.
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -e

(
    export PYTHONPATH=$PWD

    date
    echo "=== build L1R60 + uniform R180 ICs in parallel (4 threads each) ==="
    python3 -u "$CWD/build_l1_wider_margin_active.py" \
        --out /bk/l1_wider_margin_active_ic220_v2.h5 --n-threads 4 \
        > "$CWD/build_l1_wider_margin_active.log" 2>&1 &
    PID1=$!
    python3 -u "$CWD/build_uniform_r180_active.py" \
        --out /bk/uniform_r180_active_ic220_v2.h5 --n-threads 4 \
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

    echo "=== step both in parallel (4 threads each), 300 steps, plot every 100 ==="
    python3 -u "$CWD/step_l1_wider_margin_active.py" \
        --ic /bk/l1_wider_margin_active_ic220_v2.h5 \
        --out-dir /bk/l1_wider_margin_active_v2_stepped \
        --plot-dir "$CWD/plots/L1_R60" \
        --n-steps 300 --plot-every 100 --n-threads 4 \
        > "$CWD/step_l1_wider_margin_active.log" 2>&1 &
    PID3=$!
    python3 -u "$CWD/step_uniform_r180_active.py" \
        --ic /bk/uniform_r180_active_ic220_v2.h5 \
        --out-dir /bk/uniform_r180_active_v2_stepped \
        --plot-dir "$CWD/plots/uniform_r180" \
        --n-steps 300 --plot-every 100 --n-threads 4 \
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
