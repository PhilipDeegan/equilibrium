#!/bin/bash
# 10-07 pair: GP clean seed N 2 r0 30 a 10 + a body-centre partner, q -1 (CsCl, opposite sense) vs q +1 (BCC control)
cd "$(dirname "$0")"
ROOT=$(cd ../../../../.. && pwd)
for q in -1 1; do
  date
  PYTHONPATH=$ROOT python3 -u pair_box.py --partner 0.5,0.5,0.5,$q --N 2 --a 10 --r0 30 --P 300 --m 0.8 --xi 3.9 \
    --walk_xi --T 8000 --n_plot 8 --recentre 500 --threads 8 --parity --probe_every 200 --project_plus \
    --profile gp_profile_xi3.9.npz --tail 20,40 --slosh_every 20 --slosh_e_every 100 --tag _gp > pair_q${q}_N2_a10_r30_P300.log 2>&1
done
date
