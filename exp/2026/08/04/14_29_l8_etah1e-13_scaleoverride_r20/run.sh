#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -ex

(
    export PYTHONPATH=$PWD

    # Redo of exp/2026/08/04/13_25_l8_etah_1e-13_r20_r30's R=20 case, this time
    # WITH --scale-override 1.3606e-3 --scale-ref-level 8 (dropped earlier
    # this session on the mistaken assumption it was a no-op at max_level=8 --
    # it isn't: it's what makes R=20 represent the SAME real-calibrated object
    # size every other rung of this ladder (L2/L4/L5/L6/L7) was tested
    # against, matching project_odd27_amr_junction_leak Parts 10/11/15's own
    # established methodology. Without it, R=20 self-calibrates to a ~5x
    # smaller, non-comparable object (inner_cells=27 instead of what the
    # override gives). L8 eta_h combined ladder still =1e-13 (continuing L7's
    # value, same as before) -- if this also diverges, the base-update L8
    # blowup is real and independent of both the scale mistake and eta_h, per
    # doc/dst_amr_scaling.md's fp64-wall analysis (which only implicates the
    # eta_h/biharmonic term, not the base update).
    OUT_DIR="$PWD/exp/2026/08/04/14_29_l8_etah1e-13_scaleoverride_r20"
    mkdir -p "$OUT_DIR"

    date
    echo "=== L8 eta_h=1e-13, WITH scale override, R=20: cross_scale=1e-3, 1 coarse step ==="
    python3 -u run/substrate/adaptive/refined_neutron_decay.py \
        --R 20 --max-level 8 --region-r0 5.0 \
        --scale-override 1.3606e-3 --scale-ref-level 8 \
        --lam-target 1.0 --lam-scale 0.05 --lam-mode dynamic \
        --cross-scales 1e-3 --damp-max 0 \
        --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 1e-13 \
        --n-steps 1 --report 1 --n-threads 8 \
        --save-snapshot-dir "$OUT_DIR/snapshots" \
        > "$OUT_DIR/l8_scaleoverride.log" 2>&1
    date
    echo "=== finished ==="
    tail -40 "$OUT_DIR/l8_scaleoverride.log"

) 1> >(tee "$CWD"/.run.sh.out) 2> >(tee "$CWD"/.run.sh.err >&2)
