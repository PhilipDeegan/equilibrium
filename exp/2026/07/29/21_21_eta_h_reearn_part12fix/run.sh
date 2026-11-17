#!/bin/bash
set -e
cd /home/p/git/eq
export PYTHONPATH=$PWD

BASE="python3 -u tst/substrate/adaptive/refined_neutron_decay.py \
  --R 20 --region-r0 5.0 \
  --scale-override 1.3606e-3 --scale-ref-level 8 \
  --lam-target 1.0 --lam-scale 0.05 --lam-mode frozen \
  --cross-scales 0 --damp-max 0 \
  --n-threads 8"

# L2 (re-earned: 1e-5, unchanged from pre-fix)
$BASE --max-level 2 --eta-h-per-level 0 0 1e-5 --n-steps 500 --report 25

# L3 (re-earned: 1e-7, shifted slightly from pre-fix 3e-7)
$BASE --max-level 3 --eta-h-per-level 0 0 1e-5 1e-7 --n-steps 500 --report 25

# L4 (re-earned: 3e-9, unchanged from pre-fix)
$BASE --max-level 4 --eta-h-per-level 0 0 1e-5 1e-7 3e-9 --n-steps 500 --report 25

# L5 (re-earned: 1e-10, shifted ~3x from pre-fix 3e-11) — see l5_run_sweep.sh
# for the full 150-step candidate sweep that picked this value.
$BASE --max-level 5 --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 --n-steps 500 --report 25

# L6 (re-earned: 1e-12, unchanged from pre-fix) — see l6_run_sweep.sh for the
# full 150-step candidate sweep. 500-step durability check not yet run.
$BASE --max-level 6 --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 --n-steps 150 --report 15
