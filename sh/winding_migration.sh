#!/usr/bin/env bash
# DST hydrogen ground state — BC comparison + domain size study.
#
# Tests whether sponge BC (matching 1/r profile at boundary) stabilises
# the orbit at small R, and confirms it agrees with larger R absorbing BC.
# If BC-fix and larger-R give the same orbit_r, the orbit is real.
#
# chi=0.9 sweep: alpha=0.05, u_central=0.1 (product=0.005)
# chi=1.0 sweep: maximum compression limit — does spontaneous moat radius shift?

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..
set -ex

export PYTHONPATH=$PWD

BASE="--alpha 0.05 --u-central 0.1 --n-warmup 5000 --n-ring 20000 --n-probe 2000"
BASE="$BASE --amplitude 1.0 --report 1000 --threads 10"

# --- chi=1.0 (maximum compression) ---

# 1. R=40  absorbing
date && python3 -Ou run/substrate/winding_migration.py $BASE --chi 1.0 \
    --R 40  --bc absorbing --probe-rmax 20 --save-prefix winding_R40_abs_chi1.0

# 2. R=80  absorbing
date && python3 -Ou run/substrate/winding_migration.py $BASE --chi 1.0 \
    --R 80  --bc absorbing --probe-rmax 40 --save-prefix winding_R80_abs_chi1.0

# 3. R=80  sponge
date && python3 -Ou run/substrate/winding_migration.py $BASE --chi 1.0 \
    --R 80  --bc sponge --probe-rmax 40 --save-prefix winding_R80_sponge_chi1.0
