#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -ex

(
    export PYTHONPATH=$PWD

    # Follow-on to exp/2026/08/04/11_53_l8_first_look_1step (R=20) and
    # exp/2026/08/04/12_33_l8_r30_first_look_1step (R=30) -- both diverged at
    # step=1 with near-identical severity (H/H0=599 and 606) despite R=30
    # resolving the core 4x better (inner_cells 27->113). That similarity
    # points at eta_h=0 (unset at L8 in both) as the cause, not resolution --
    # but it was confounded until now. This run gives L8 a non-zero eta_h
    # guess (1e-14) to remove "no damping at all" as a variable: the L6->L7
    # ladder transition tightened by 10x (1e-12->1e-13), so extrapolating that
    # same ratio gives 1e-14 at L8 -- NOT itself earned, just a better-motivated
    # placeholder than flat-continuing 1e-13. Repeats both R=20 and R=30,
    # sequentially (not parallel -- 8 threads each, avoid oversubscribing the
    # 8-core/16-thread machine).
    OUT_DIR="$PWD/exp/2026/08/04/12_46_l8_etah_continued_r20_r30"
    mkdir -p "$OUT_DIR"

    for R in 20 30; do
        date
        echo "=== L8 eta_h-guess probe, R=${R}: cross_scale=1e-3, 1 coarse step, L8 eta_h=1e-14 (L6->L7 ratio extrapolated, not earned) ==="
        python3 -u run/substrate/adaptive/refined_neutron_decay.py \
            --R "${R}" --max-level 8 --region-r0 5.0 \
            --lam-target 1.0 --lam-scale 0.05 --lam-mode dynamic \
            --cross-scales 1e-3 --damp-max 0 \
            --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 1e-14 \
            --n-steps 1 --report 1 --n-threads 8 \
            --save-snapshot-dir "$OUT_DIR/snapshots_R${R}" \
            > "$OUT_DIR/l8_etah_guess_R${R}.log" 2>&1
        date
        echo "=== finished R=${R} ==="
        tail -40 "$OUT_DIR/l8_etah_guess_R${R}.log"
    done

) 1> >(tee "$CWD"/.run.sh.out) 2> >(tee "$CWD"/.run.sh.err >&2)
