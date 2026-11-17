#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -ex

(
    export PYTHONPATH=$PWD

    # Follow-on to exp/2026/08/04/11_53_l8_first_look_1step (R=20, diverged at
    # step=1, H/H0=599 -- real calibration only resolved the core with 27
    # cells, and L8's own eta_h defaulted to 0/unset). The eventual real test
    # is R=100 (the script's own default, matching the actual Bohr-radius/
    # margin calibration this whole thing is built on) -- stepping there via
    # R=30 first rather than jumping straight to R=100's cost. Same 1-step,
    # real-calibration (no --scale-override), L8-eta_h-unset probe as before;
    # only R changed.
    OUT_DIR="$PWD/exp/2026/08/04/12_33_l8_r30_first_look_1step"
    mkdir -p "$OUT_DIR"

    date
    echo "=== L8 first look, R=30: cross_scale=1e-3, 1 coarse step, real calibration, L8 eta_h defaults to 0 (unset) ==="
    python3 -u run/substrate/adaptive/refined_neutron_decay.py \
        --R 30 --max-level 8 --region-r0 5.0 \
        --lam-target 1.0 --lam-scale 0.05 --lam-mode dynamic \
        --cross-scales 1e-3 --damp-max 0 \
        --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 \
        --n-steps 1 --report 1 --n-threads 8 \
        --save-snapshot-dir "$OUT_DIR/snapshots" \
        > "$OUT_DIR/l8_r30_1step.log" 2>&1
    date
    echo "=== finished L8 R=30 first look ==="
    tail -40 "$OUT_DIR/l8_r30_1step.log"

) 1> >(tee "$CWD"/.run.sh.out) 2> >(tee "$CWD"/.run.sh.err >&2)
