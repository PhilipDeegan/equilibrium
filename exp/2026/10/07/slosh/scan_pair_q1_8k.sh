#!/bin/bash
# 10-07 evening: q +1 pair control rerun to 8k (the fedora run stopped at ~5.5k) -- is its ~4 % faster drift (fitted) real or a breathing-phase window effect?
cd "$(dirname "$0")"
ROOT=$(cd ../../../../.. && pwd)
date
PYTHONPATH=$ROOT python3 -u pair_box.py --partner 0.5,0.5,0.5,1 --N 2 --a 10 --r0 30 --P 300 --m 0.8 --xi 3.9 \
  --walk_xi --T 8000 --n_plot 8 --recentre 500 --threads 8 --parity --probe_every 200 --project_plus \
  --profile gp_profile_xi3.9.npz --tail 20,40 --slosh_every 20 --slosh_e_every 100 --tag _gp8 > pair_q1_N2_a10_r30_P300_T8k.log 2>&1
date
