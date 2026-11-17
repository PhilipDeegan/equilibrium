#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -ex

(
    export PYTHONPATH=$PWD

    # Follow-on to exp/2026/08/04/12_46_l8_etah_continued_r20_r30 -- L8 eta_h=0
    # and eta_h=1e-14 gave near-identical divergence at both R=20 and R=30
    # (H/H0 ~599-606 in all four combinations). R itself also barely moved the
    # result (1.2%) despite inner_cells going 27->113. Both knobs tried so far
    # are flat against a ~600x blowup -- testing a much larger L8 eta_h
    # (1e-13, same as L7, no tightening at all) to see if ANY magnitude in
    # this parameter has a visible effect before concluding eta_h isn't the
    # lever and the cause is structural (dt/subcycling schedule, precision/
    # encoding limit, etc. specific to reaching max_level=8).
    OUT_DIR="$PWD/exp/2026/08/04/13_25_l8_etah_1e-13_r20_r30"
    mkdir -p "$OUT_DIR"

    for R in 20 30; do
        date
        echo "=== L8 eta_h=1e-13 probe, R=${R}: cross_scale=1e-3, 1 coarse step ==="
        python3 -u run/substrate/adaptive/refined_neutron_decay.py \
            --R "${R}" --max-level 8 --region-r0 5.0 \
            --lam-target 1.0 --lam-scale 0.05 --lam-mode dynamic \
            --cross-scales 1e-3 --damp-max 0 \
            --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 1e-13 \
            --n-steps 1 --report 1 --n-threads 8 \
            --save-snapshot-dir "$OUT_DIR/snapshots_R${R}" \
            > "$OUT_DIR/l8_etah1e-13_R${R}.log" 2>&1
        date
        echo "=== finished R=${R} ==="
        tail -40 "$OUT_DIR/l8_etah1e-13_R${R}.log"
    done

) 1> >(tee "$CWD"/.run.sh.out) 2> >(tee "$CWD"/.run.sh.err >&2)
