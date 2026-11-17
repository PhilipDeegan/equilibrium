#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -ex

(
    export PYTHONPATH=$PWD

    # Exact reproduction of exp/2026/08/04/14_29_l8_etah1e-13_scaleoverride_r20's
    # run (the properly-calibrated L8 repro, H/H0=134.1 at step 1, matching
    # project_odd27_amr_junction_leak Part 2's 130.8 finding) -- rerun today
    # against the freshly rebuilt mod_lattice.so containing the
    # accumulate_to_parent catastrophic-cancellation fix (doc/dst_algo.md,
    # 2026-08-04) to see whether that fix changes the base-update L8 blowup
    # at all. Same flags, same eta_h ladder, same everything -- only the
    # underlying .so differs.
    OUT_DIR="$PWD/exp/2026/08/04/21_37_l8_accumulate_to_parent_fix_validation"
    mkdir -p "$OUT_DIR"

    date
    echo "=== L8 accumulate_to_parent fix validation: R=20, same params as 14_29's repro ==="
    python3 -u run/substrate/adaptive/refined_neutron_decay.py \
        --R 20 --max-level 8 --region-r0 5.0 \
        --scale-override 1.3606e-3 --scale-ref-level 8 \
        --lam-target 1.0 --lam-scale 0.05 --lam-mode dynamic \
        --cross-scales 1e-3 --damp-max 0 \
        --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 1e-13 \
        --n-steps 1 --report 1 --n-threads 8 \
        --save-snapshot-dir "$OUT_DIR/snapshots" \
        > "$OUT_DIR/l8_fix_validation.log" 2>&1
    date
    echo "=== finished ==="
    tail -40 "$OUT_DIR/l8_fix_validation.log"

) 1> >(tee "$CWD"/.run.sh.out) 2> >(tee "$CWD"/.run.sh.err >&2)
