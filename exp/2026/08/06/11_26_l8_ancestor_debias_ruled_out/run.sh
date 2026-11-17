#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -ex

(
    export PYTHONPATH=$PWD

    # L8 revalidation of enable_ancestor_debias_correction (see now / Part 26
    # of project_odd27_amr_junction_leak.md) against the real base-update L8
    # blowup (exp/2026/08/04/14_29_l8_etah1e-13_scaleoverride_r20, H/H0=134.06).
    # Requires the mod_lattice .so rebuilt AFTER adding
    # set/get_enable_ancestor_debias_correction bindings to src/eq/lattice.cpp
    # (they were not exposed to Python before this session).
    OUT_DIR="$PWD/exp/2026/08/06/11_26_l8_ancestor_debias_ruled_out"
    mkdir -p "$OUT_DIR"

    date
    echo "=== baseline (no debias correction) ==="
    python3 -u run/substrate/adaptive/refined_neutron_decay.py \
        --R 20 --max-level 8 --region-r0 5.0 \
        --scale-override 1.3606e-3 --scale-ref-level 8 \
        --lam-target 1.0 --lam-scale 0.05 --lam-mode dynamic \
        --cross-scales 1e-3 --damp-max 0 \
        --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 1e-13 \
        --n-steps 1 --report 1 --n-threads 8 \
        > "$OUT_DIR/baseline.log" 2>&1
    date

    echo "=== with --enable-ancestor-debias ==="
    python3 -u run/substrate/adaptive/refined_neutron_decay.py \
        --R 20 --max-level 8 --region-r0 5.0 \
        --scale-override 1.3606e-3 --scale-ref-level 8 \
        --lam-target 1.0 --lam-scale 0.05 --lam-mode dynamic \
        --cross-scales 1e-3 --damp-max 0 \
        --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 1e-13 \
        --n-steps 1 --report 1 --n-threads 8 \
        --enable-ancestor-debias \
        > "$OUT_DIR/debias.log" 2>&1
    date

    echo "=== raw per-cell array diff (bare AdaptiveLattice, no eta_h/cross_scale/lam) ==="
    python3 -u "$OUT_DIR/diff_check_l8.py" 2>&1 | grep -v "reflux-probe\|matched\|init_spheromak" \
        > "$OUT_DIR/diff_check_l8.log"
    cat "$OUT_DIR/diff_check_l8.log"

    echo "=== H/H0 comparison ==="
    grep "step=\|H/H0" "$OUT_DIR/baseline.log" "$OUT_DIR/debias.log"

) 1> >(tee "$CWD"/.run.sh.out) 2> >(tee "$CWD"/.run.sh.err >&2)
