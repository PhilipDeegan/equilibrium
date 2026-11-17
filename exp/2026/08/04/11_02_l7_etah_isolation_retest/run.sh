#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -ex

(
    export PYTHONPATH=$PWD

    # L7 dynamic-lam eta_h isolation retest (follow-on to
    # exp/2026/08/03/15_15_l7_dynamic_lam_ladder_test, Part 23):
    # that run's L6->L7 jump in H/H0 (peak 14.14->58.47, @150 4.05->18.58) was
    # much sharper than the L4->L5->L6 progression (~2-2.6x/level vs ~4-4.5x
    # here). Question: is the jump driven by the L7 eta_h value dropping to
    # 1e-13 in the ladder (0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13), or intrinsic
    # to running one level deeper regardless of eta_h? This retest substitutes
    # L6's eta_h (1e-12) into the L7 slot, keeping everything else identical --
    # if H/H0 comes back down near L6's numbers, the ladder's eta_h tail is the
    # cause; if it stays near the previous L7 run's numbers, it's intrinsic to
    # the level.
    OUT_DIR="$PWD/exp/2026/08/04/11_02_l7_etah_isolation_retest"
    mkdir -p "$OUT_DIR"

    date
    echo "=== L7 dynamic-lam eta_h isolation retest: cross_scale=1e-3, L7 slot eta_h=1e-12 (L6 value, not ladder's 1e-13) ==="
    python3 -u run/substrate/adaptive/refined_neutron_decay.py \
        --R 20 --max-level 7 --region-r0 5.0 \
        --scale-override 1.3606e-3 --scale-ref-level 8 \
        --lam-target 1.0 --lam-scale 0.05 --lam-mode dynamic \
        --cross-scales 1e-3 --damp-max 0 \
        --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-12 \
        --n-steps 150 --report 15 --n-threads 8 \
        > "$OUT_DIR/dynamic_lam_L7_etah1e-12.log" 2>&1
    date
    echo "=== finished L7 eta_h isolation retest ==="
    tail -40 "$OUT_DIR/dynamic_lam_L7_etah1e-12.log"

) 1> >(tee "$CWD"/.run.sh.out) 2> >(tee "$CWD"/.run.sh.err >&2)
