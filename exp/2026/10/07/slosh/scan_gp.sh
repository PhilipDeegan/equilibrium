#!/bin/bash
# 10-07: GP clean seed (core/gp_vortex.py, tail -> 1 by d 40, + projected) N 2 r0 30 a 10, slosh probe every 20 ticks
cd "$(dirname "$0")"
ROOT=$(cd ../../../../.. && pwd)
for P in 220 180 340; do
  date
  PYTHONPATH=$ROOT python3 -u spring_box.py --N 2 --a 10 --r0 30 --P $P --m 0.8 --xi 3.9 --walk_xi --T 8000 \
    --n_plot 8 --recentre 500 --threads 8 --parity --probe_every 200 --project_plus --profile gp_profile_xi3.9.npz \
    --tail 20,40 --slosh_every 20 --slosh_e_every 100 --tag _gp > box_gp_N2_a10_r30_P${P}.log 2>&1
done
date
