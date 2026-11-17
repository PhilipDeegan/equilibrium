#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$CWD"/../../../../..

set -ex

export PYTHONPATH=$PWD
OUT_DIR="$CWD"

# First-ever test of the eta_h ladder with lam_mode=dynamic (all prior sweeps used
# lam_mode=frozen). Cheap levels first (L2/L4/L5/L6) since they're fast on this
# machine; L7 deliberately deferred until these confirm no dynamic-lam-specific
# blowup. cross_scale fixed at the already-validated 1e-3 (Part 17/18) so this test
# isolates the dynamic-lam change alone, not combined with a cross_scale sweep.
# eta_h ladder truncated per max-level from the fixed provisional L0-L7 ladder:
#   0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13

declare -A LADDERS=(
  [2]="0 0 1e-5"
  [4]="0 0 1e-5 1e-7 3e-9"
  [5]="0 0 1e-5 1e-7 3e-9 1e-10"
  [6]="0 0 1e-5 1e-7 3e-9 1e-10 1e-12"
)

for LEVEL in 2 4 5 6; do
    date
    echo "=== dynamic-lam ladder test: max-level=$LEVEL, cross_scale=1e-3, eta_h=${LADDERS[$LEVEL]} ==="
    python3 -u run/substrate/adaptive/refined_neutron_decay.py \
        --R 20 --max-level "$LEVEL" --region-r0 5.0 \
        --scale-override 1.3606e-3 --scale-ref-level 8 \
        --lam-target 1.0 --lam-scale 0.05 --lam-mode dynamic \
        --cross-scales 1e-3 --damp-max 0 \
        --eta-h-per-level ${LADDERS[$LEVEL]} \
        --n-steps 150 --report 15 --n-threads 8 \
        > "$OUT_DIR/dynamic_lam_L${LEVEL}.log" 2>&1
    date
    echo "=== finished max-level=$LEVEL ==="
    tail -25 "$OUT_DIR/dynamic_lam_L${LEVEL}.log"
done
