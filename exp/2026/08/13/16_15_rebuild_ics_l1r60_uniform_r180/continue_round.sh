#!/usr/bin/env bash
# Resume both configs from their latest checkpoint and advance N_STEPS further
# (default 30), plotting once at the end of the round. Both run in parallel,
# 4 threads each. Auto-detects the latest step_*_final.h5 in each --out-dir.
# Re-run this script as many times as needed to build up a longer trajectory
# a round at a time.
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -e

N_STEPS="${1:-30}"

(
    export PYTHONPATH=$PWD

    latest_snapshot() { ls -1v "$1"/step_*_final.h5 2>/dev/null | tail -1; }

    L1_SNAP="$(latest_snapshot /bk/l1_wider_margin_active_stepped)"
    UNI_SNAP="$(latest_snapshot /bk/uniform_r180_active_stepped)"

    date
    echo "=== round: N_STEPS=$N_STEPS  L1R60 resuming from $L1_SNAP  uniformR180 resuming from $UNI_SNAP ==="

    python3 -u "$CWD/step_l1_wider_margin_active.py" \
        --snapshot "$L1_SNAP" \
        --out-dir /bk/l1_wider_margin_active_stepped \
        --plot-dir "$CWD/plots/L1_R60" \
        --n-steps "$N_STEPS" --plot-every "$N_STEPS" --n-threads 4 \
        >> "$CWD/step_l1_wider_margin_active.log" 2>&1 &
    PID1=$!
    python3 -u "$CWD/step_uniform_r180_active.py" \
        --snapshot "$UNI_SNAP" \
        --out-dir /bk/uniform_r180_active_stepped \
        --plot-dir "$CWD/plots/uniform_r180" \
        --n-steps "$N_STEPS" --plot-every "$N_STEPS" --n-threads 4 \
        >> "$CWD/step_uniform_r180_active.log" 2>&1 &
    PID2=$!

    RC1=0; RC2=0
    wait "$PID1" || RC1=$?
    wait "$PID2" || RC2=$?
    date
    echo "=== round done: L1R60 rc=$RC1  uniformR180 rc=$RC2 ==="

    # Keep only the 2 most recent checkpoints per config (safety margin of 1)
    # -- each round adds a new step_*_final.h5 and never removes the old one,
    # and these are multi-GB each, so left unchecked they accumulate fast.
    if [ "$RC1" -eq 0 ] && [ "$RC2" -eq 0 ]; then
        for d in /bk/l1_wider_margin_active_stepped /bk/uniform_r180_active_stepped; do
            ls -1v "$d"/step_*_final.h5 2>/dev/null | head -n -2 | xargs -r rm -v
        done
    fi

) 1>> "$CWD/continue_round.log" 2>> "$CWD/continue_round.err"
