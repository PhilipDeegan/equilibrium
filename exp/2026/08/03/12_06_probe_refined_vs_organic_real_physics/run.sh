#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$CWD"/../../../../..

set -ex

export PYTHONPATH=$PWD
OUT_DIR="$CWD"

# Correctness check (not an eta_h stability check): does refined_neutron_decay.py's
# fixed-telescope AMR reproduce sweep_neutron_decay_cross_scale.py's organic/growing
# AMR under REAL physics (lam_mode=dynamic, damp_max=0.003, cross_scale=1e-3 -- all
# script defaults, "everything active")? Previously only validated at max_level=0
# (trivial, both flat/uniform). Testing 0 (re-check against current .so), 1, 2 (the
# script's own documented example) before trusting L7/L8 eta_h-ladder work further.

for LEVEL in 0 1 2; do
    date
    echo "=== probe_refined_vs_organic_amr: max_level=$LEVEL, everything active (defaults) ==="
    python3 -u run/substrate/adaptive/probe_refined_vs_organic_amr.py \
        --max-level "$LEVEL" --split-hi 0.02 --n-threads 8 \
        > "$OUT_DIR/probe_L${LEVEL}.log" 2>&1
    date
    echo "=== finished max_level=$LEVEL ==="
    tail -30 "$OUT_DIR/probe_L${LEVEL}.log"
done
