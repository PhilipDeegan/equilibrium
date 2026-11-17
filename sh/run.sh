#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

(
    export PYTHONPATH=$PWD

    # Pair 3 of the R=50 barrier-gate resolution check (killed before it could
    # auto-start from sh's own 3-pair loop, see exp/2026/08/24/17_40_barrier_r50_check/).
    # Pairs 1 (m0=2.49, cs=0.1) and 2 (m0=3.0, cs=0.1) already completed there --
    # both confirm the R=20 separation (surv_e/surv_n) holds up at R=50:
    #
    #   m0    cs    R=20 sep   R=50 sep
    #   2.49  0.1   2.550      2.558
    #   3.0   0.1   2.217      2.293
    #
    OUT_DIR="$PWD/exp/2026/08/24/17_40_barrier_r50_check"
    OUT="$OUT_DIR/m0_2.49_cs_0.03.log"

    date
    echo "=== pair 3: m0=2.49 cs=0.03, scales 4,10 ==="
    python3 -u run/substrate/adaptive/probe_resolution_scan.py \
        --scales 4,10 --gate barrier --m0 2.49 --cross-scale 0.03 \
        --base-steps 3000 --base-scale 4 --report 1000 --n-threads 8 \
        > "$OUT" 2>&1

    date
    echo "=== done ==="

) 1> >(tee $CWD/.run.sh.out) 2> >(tee $CWD/.run.sh.err >&2)
