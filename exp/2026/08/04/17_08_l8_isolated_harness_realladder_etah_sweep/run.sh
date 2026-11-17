#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -ex

(
    export PYTHONPATH=$PWD

    # Corrected follow-on to the (killed, discarded) 16_56_* attempt, which
    # wrongly zeroed L0-L7's eta_h instead of holding the earned ladder fixed
    # -- that changed two variables at once (cross_scale/lam_mode AND the
    # L0-L7 ladder) instead of isolating cross_scale/lam_mode alone as the
    # single new variable versus every earlier L8 test today. This run keeps
    # L0-L7 at the earned ladder values (0 0 1e-5 1e-7 3e-9 1e-10 1e-12
    # 1e-13, same as exp/2026/08/04/13_25_l8_etah_1e-13_r20_r30 etc.) and
    # changes ONLY --cross-scales/--lam-mode/--damp-max to the fully-isolated
    # harness (matching project_odd27_amr_junction_leak Parts 6-11's original
    # eta_h calibration convention), sweeping L8's own eta_h value.
    OUT_DIR="$PWD/exp/2026/08/04/17_08_l8_isolated_harness_realladder_etah_sweep"
    mkdir -p "$OUT_DIR"

    for ETAH in 1e-5 1e-7 1e-10 1e-12; do
        date
        echo "=== L8 ISOLATED harness (cross_scale=0, lam frozen, damp_max=0), earned L0-L7 ladder, eta_h[8]=${ETAH} ==="
        python3 -u run/substrate/adaptive/refined_neutron_decay.py \
            --R 20 --max-level 8 --region-r0 5.0 \
            --scale-override 1.3606e-3 --scale-ref-level 8 \
            --lam-mode frozen --cross-scales 0 --damp-max 0 \
            --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 "${ETAH}" \
            --n-steps 10 --report 1 --n-threads 8 \
            --save-snapshot-dir "$OUT_DIR/snapshots_etah${ETAH}" \
            > "$OUT_DIR/l8_isolated_realladder_etah${ETAH}.log" 2>&1
        date
        echo "=== finished eta_h[8]=${ETAH} ==="
        tail -20 "$OUT_DIR/l8_isolated_realladder_etah${ETAH}.log"
    done

) 1> >(tee "$CWD"/.run.sh.out) 2> >(tee "$CWD"/.run.sh.err >&2)
