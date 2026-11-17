#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$CWD"/../../../../..

set -ex

export PYTHONPATH=$PWD
OUT_DIR="$CWD"

# Last rung before L8: same fixed provisional L0-L7 ladder that passed cleanly
# at L2/L4/L5/L6 under lam_mode=dynamic (project_odd27_amr_junction_leak Part 19),
# now at the full max_level=7. cross_scale held at the already-validated 1e-3,
# damp_max=0, matching every prior test in this line so only max_level changes.
# This is the expensive rung (~7h/candidate per Part 16's costing at L7 under the
# isolated harness) -- --blowup-factor default (100) aborts early at the next
# --report checkpoint if it diverges, so it won't necessarily burn the full
# --n-steps if something goes wrong.

date
echo "=== L7 dynamic-lam ladder test: cross_scale=1e-3, eta_h ladder full L0-L7 ==="
python3 -u run/substrate/adaptive/refined_neutron_decay.py \
    --R 20 --max-level 7 --region-r0 5.0 \
    --scale-override 1.3606e-3 --scale-ref-level 8 \
    --lam-target 1.0 --lam-scale 0.05 --lam-mode dynamic \
    --cross-scales 1e-3 --damp-max 0 \
    --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 \
    --n-steps 150 --report 15 --n-threads 8 \
    > "$OUT_DIR/dynamic_lam_L7.log" 2>&1
date
echo "=== finished L7 dynamic-lam test ==="
tail -40 "$OUT_DIR/dynamic_lam_L7.log"
