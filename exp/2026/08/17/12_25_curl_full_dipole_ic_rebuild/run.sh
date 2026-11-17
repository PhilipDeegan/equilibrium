#!/usr/bin/env bash
# Rebuild L1R60 and uniform-R180 with the NEW volume-filling curl-full
# spheromak IC (equilibrium.substrate.curl_full_dipole_ic), replacing both
# the 2026-08-16 14:50 curl-free dipole ICs and the 2026-08-16 18:56
# wire-loop Hopf-pair ICs for these two configs. Same parallel/thread-split
# convention as the 14:50 rebuild (L1R60 bigger/junction-bearing config gets
# 6 threads, uniform R180 gets 3, sum=9 on the 8-core/16-thread machine).
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -e

(
    export PYTHONPATH=$PWD

    date
    echo "=== build L1R60 + uniform R180 curl-full-dipole ICs in parallel (L1R60: 6 threads, uniform R180: 3 threads) ==="
    python3 -u "$CWD/build_curlfull_dipole_l1_wider_margin.py" \
        --out /bk/curlfull_dipole_l1_wider_margin_ic220.h5 --n-threads 6 \
        > "$CWD/build_curlfull_dipole_l1_wider_margin.log" 2>&1 &
    PID1=$!
    python3 -u "$CWD/build_curlfull_dipole_uniform_r180.py" \
        --out /bk/curlfull_dipole_uniform_r180_ic220.h5 --n-threads 3 \
        > "$CWD/build_curlfull_dipole_uniform_r180.log" 2>&1 &
    PID2=$!

    RC1=0; RC2=0
    wait "$PID1" || RC1=$?
    wait "$PID2" || RC2=$?
    date
    echo "=== builds done: l1_wider_margin rc=$RC1  uniform_r180 rc=$RC2 ==="

) 1> >(tee "$CWD/run.sh.out") 2> >(tee "$CWD/run.sh.err" >&2)
