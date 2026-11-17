#!/bin/bash
# 10-08: same-sense partner BESIDE (same z, lateral s / sqrt2) at P 300: superflow predicts a slow-down -0.00062 (biot_exact.py --qp -1 --dz 0, sign flipped)
cd "$(dirname "$0")"
ROOT=$(cd ../../../../.. && pwd)
date
PYTHONPATH=$ROOT python3 -u pair_box.py --partner 0.5,0.5,0,1 --N 2 --a 10 --r0 30 --P 300 --m 0.8 --xi 3.9 \
  --walk_xi --T 8000 --n_plot 8 --recentre 500 --threads 8 --parity --probe_every 200 --project_plus \
  --profile gp_profile_xi3.9.npz --tail 20,40 --slosh_every 20 --slosh_e_every 100 --tag _side > pair_side_q1_N2_a10_r30_P300.log 2>&1
date
