#!/usr/bin/env bash
# Resume both configs from their latest checkpoint and advance N_STEPS further
# (default 200), plotting every 25. Both run in parallel, L1R60 on 6 threads,
# uniform R180 on 3 (per user request, sum=9 on the 8c/16t machine).
# Auto-detects the latest step_*_final.h5 in each --out-dir. NDEBUG/release
# build (same as the first 100 steps -- see report.md).
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -e

N_STEPS="${1:-200}"

(
    export PYTHONPATH=$PWD

    latest_snapshot() { ls -1v "$1"/step_*_final.h5 2>/dev/null | tail -1; }

    L1_SNAP="$(latest_snapshot /bk/l1_wider_margin_active_pp1fix_stepped)"
    UNI_SNAP="$(latest_snapshot /bk/uniform_r180_active_pp1fix_stepped)"

    date
    echo "=== round: N_STEPS=$N_STEPS  L1R60 resuming from $L1_SNAP  uniformR180 resuming from $UNI_SNAP ==="

    python3 -u "$CWD/step_l1_wider_margin_active.py" \
        --snapshot "$L1_SNAP" \
        --out-dir /bk/l1_wider_margin_active_pp1fix_stepped \
        --plot-dir "$CWD/plots/L1_R60" \
        --n-steps "$N_STEPS" --plot-every 25 --n-threads 6 \
        >> "$CWD/step_l1_wider_margin_active.log" 2>&1 &
    PID1=$!
    python3 -u "$CWD/step_uniform_r180_active.py" \
        --snapshot "$UNI_SNAP" \
        --out-dir /bk/uniform_r180_active_pp1fix_stepped \
        --plot-dir "$CWD/plots/uniform_r180" \
        --n-steps "$N_STEPS" --plot-every 25 --n-threads 3 \
        >> "$CWD/step_uniform_r180_active.log" 2>&1 &
    PID2=$!

    RC1=0; RC2=0
    wait "$PID1" || RC1=$?
    wait "$PID2" || RC2=$?
    date
    echo "=== round done: L1R60 rc=$RC1  uniformR180 rc=$RC2 ==="

) 1>> "$CWD/continue_round.log" 2>> "$CWD/continue_round.err"
