#!/usr/bin/env bash
# Exact commands used for the original spheromak decay chain.
# lam on ring tiles (not poles), absorbing BC (u_bg=0).
# Snapshots: proton_candidate_R40.npz → decay2 → decay3 → decay4 → decay5

set -ex
export PYTHONPATH=$PWD

# Stage 0 — fresh IC
date && python3 tst/substrate/test_spheromak_decay.py \
    --R 40 --rings both --lam-rings 1.0 \
    --cross-scale 0.005 --lam-scale 0.05 \
    --n-steps 0 --report 2000 \
    --save-snapshot proton_candidate_R40_IC.npz \
    --plot-out tmp/spheromak_R40.png

# Stage 1
date && python3 tst/substrate/test_spheromak_decay.py \
    --R 40 --rings both --lam-rings 1.0 \
    --load proton_candidate_R40_IC.npz \
    --cross-scale 0.005 --lam-scale 0.05 \
    --n-steps 20000 --report 2000 \
    --save-snapshot proton_candidate_R40_decay1.npz \
    --plot-out tmp/spheromak_R40_decay1.png

# Stage 2
date && python3 tst/substrate/test_spheromak_decay.py \
    --R 40 --rings both \
    --load proton_candidate_R40_decay1.npz \
    --cross-scale 0.005 --lam-scale 0.05 \
    --n-steps 20000 --report 2000 \
    --save-snapshot proton_candidate_R40_decay2.npz \
    --plot-out tmp/spheromak_R40_decay2.png

# Stage 3
date && python3 tst/substrate/test_spheromak_decay.py \
    --R 40 --rings both \
    --load proton_candidate_R40_decay2.npz \
    --cross-scale 0.005 --lam-scale 0.05 \
    --n-steps 20000 --report 2000 \
    --save-snapshot proton_candidate_R40_decay3.npz \
    --plot-out tmp/spheromak_R40_decay3.png

# Stage 4 — circ_B≈0, circ_A=+0.007 at start, pure dipole
date && python3 tst/substrate/test_spheromak_decay.py \
    --R 40 --rings both \
    --load proton_candidate_R40_decay3.npz \
    --cross-scale 0.005 --lam-scale 0.05 \
    --n-steps 20000 --report 2000 \
    --save-snapshot proton_candidate_R40_decay4.npz \
    --plot-out tmp/spheromak_R40_decay4.png

# Stage 5 — electron moat visible in final snapshot
date && python3 tst/substrate/test_spheromak_decay.py \
    --R 40 --rings both \
    --load proton_candidate_R40_decay4.npz \
    --cross-scale 0.005 --lam-scale 0.05 \
    --n-steps 50000 --report 2000 \
    --save-snapshot proton_candidate_R40_decay5.npz \
    --plot-out tmp/spheromak_R40_decay5.png

# Comparison plot across all snapshots
python3 tst/substrate/probe_decay_compare.py \
    --R 40 \
    proton_candidate_R40_IC.npz \
    proton_candidate_R40_decay2.npz \
    proton_candidate_R40_decay3.npz \
    proton_candidate_R40_decay4.npz \
    proton_candidate_R40_decay5.npz \
    --plot-dir tmp
