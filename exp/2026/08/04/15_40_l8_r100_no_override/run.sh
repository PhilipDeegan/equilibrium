#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -ex

(
    export PYTHONPATH=$PWD

    # Follow-on to exp/2026/08/04/14_29_l8_etah1e-13_scaleoverride_r20 (H/H0=
    # 134.06 at step 1, matching the historical L8 finding almost exactly).
    # That test used R=20 + --scale-override to represent the real L8 object
    # size cheaply. This is the genuine apples-to-apples check: R=100 (the
    # script's own real default, no override needed) -- real self-calibration
    # at R=100 gives scale=1.3606e-3 natively, IDENTICAL to what the override
    # forced at R=20, so the object size is exactly matched; the only
    # difference is the actual domain/mesh size (R=20 vs R=100's much bigger
    # base grid). If this also blows up the same way, the R=20 proxy domain
    # is confirmed innocent -- it's a genuine L8-depth issue, not a small-
    # domain artifact. If it behaves DIFFERENTLY, that's a new, important
    # finding. Expect this to be noticeably slower to build than the R=20
    # runs (much bigger base mesh) -- historical note (project_odd27_amr_
    # junction_leak Part 3) flagged R=100/L8 as taking >8min for even one
    # step in a related config.
    OUT_DIR="$PWD/exp/2026/08/04/15_40_l8_r100_no_override"
    mkdir -p "$OUT_DIR"

    date
    echo "=== L8, R=100, no scale-override (real self-calibration): cross_scale=1e-3, 1 coarse step ==="
    python3 -u run/substrate/adaptive/refined_neutron_decay.py \
        --R 100 --max-level 8 --region-r0 5.0 \
        --lam-target 1.0 --lam-scale 0.05 --lam-mode dynamic \
        --cross-scales 1e-3 --damp-max 0 \
        --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 1e-13 \
        --n-steps 1 --report 1 --n-threads 8 \
        --save-snapshot-dir "$OUT_DIR/snapshots" \
        > "$OUT_DIR/l8_r100.log" 2>&1
    date
    echo "=== finished ==="
    tail -40 "$OUT_DIR/l8_r100.log"

) 1> >(tee "$CWD"/.run.sh.out) 2> >(tee "$CWD"/.run.sh.err >&2)
