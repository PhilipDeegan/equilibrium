#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$CWD"/../../../../..

set -ex

export PYTHONPATH=$PWD
OUT_DIR="$CWD"

# Extend the refined-vs-organic correctness check (project_odd27_amr_junction_leak
# Parts 20-21, exp/2026/08/03/12_06_probe_refined_vs_organic_real_physics) to L6/L7 --
# only checked at L1/L2 so far. Using --R 20 (not the probe script's own default R=80)
# to keep the base mesh affordable (R=80 starts at ~2.78M L0-only cells before any
# refinement; R=20 matches the much cheaper convention already used throughout the
# eta_h-ladder work). --n-steps is scaled up to compensate for the script's own
# time_factor convention (SCALE_REF_LEVEL=1, time_factor=3^(1-max_level)), targeting
# ~15 actual recorded steps at each level. "Everything active" = script defaults
# (lam_mode=dynamic, damp_max=0.003, cross_scale=1e-3).

date
echo "=== probe_refined_vs_organic_amr: R=20, max_level=6, n_steps=3645 (~15 actual steps) ==="
python3 -u run/substrate/adaptive/probe_refined_vs_organic_amr.py \
    --R 20 --max-level 6 --n-steps 3645 --split-hi 0.02 --n-threads 8 \
    > "$OUT_DIR/probe_L6.log" 2>&1
date
echo "=== finished max_level=6 ==="
tail -30 "$OUT_DIR/probe_L6.log"
