#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -ex

(
    export PYTHONPATH=$PWD

    # Follow-on to exp/2026/08/04/13_25_l8_etah_1e-13_r20_r30 -- L8 eta_h
    # tested at 0, 1e-14, and 1e-13 all gave near-identical step=1 divergence
    # (H/H0 ~599-606 regardless), and R=20 vs R=30 barely moved it either
    # (1.2%) despite 4x better core resolution. Both knobs flat against a
    # ~600x blowup rules out eta_h and coarse resolution as the driver.
    # Testing --gate v1 (continuous step_cross) instead of the default
    # "barrier" (quantized all-or-nothing threshold switch) -- step_cross's
    # barrier gate previously had a real runaway bug (project_step_cross,
    # fixed via drain+mirror) and is resolution-sensitive by nature (a
    # per-cell pooled threshold crossing differently at different cell
    # sizes), unlike eta_h which is a boundary-only correction term. Same
    # eta_h ladder as the last probe (L8=1e-13) held fixed so gate is the
    # only new variable.
    OUT_DIR="$PWD/exp/2026/08/04/13_57_l8_gate_v1_r20_r30"
    mkdir -p "$OUT_DIR"

    for R in 20 30; do
        date
        echo "=== L8 gate=v1 probe, R=${R}: cross_scale=1e-3, 1 coarse step ==="
        python3 -u run/substrate/adaptive/refined_neutron_decay.py \
            --R "${R}" --max-level 8 --region-r0 5.0 \
            --lam-target 1.0 --lam-scale 0.05 --lam-mode dynamic \
            --cross-scales 1e-3 --damp-max 0 --gate v1 \
            --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 1e-13 \
            --n-steps 1 --report 1 --n-threads 8 \
            --save-snapshot-dir "$OUT_DIR/snapshots_R${R}" \
            > "$OUT_DIR/l8_gatev1_R${R}.log" 2>&1
        date
        echo "=== finished R=${R} ==="
        tail -40 "$OUT_DIR/l8_gatev1_R${R}.log"
    done

) 1> >(tee "$CWD"/.run.sh.out) 2> >(tee "$CWD"/.run.sh.err >&2)
