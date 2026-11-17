#!/bin/bash
# 10-07 evening: N 4 at the spacing floor to 20k -- r0 30 (inner spacing 8 xi) drops a turn or keeps leaking? r0 40 (11 xi): does the leak go?
cd "$(dirname "$0")"
ROOT=$(cd ../../../../.. && pwd)
for R in 30 40; do
  date
  PYTHONPATH=$ROOT python3 -u spring_box.py --N 4 --a 10 --r0 $R --P 220 --m 0.8 --xi 3.9 --walk_xi --T 20000 \
    --n_plot 20 --recentre 500 --threads 8 --parity --probe_every 200 --project_plus --profile gp_profile_xi3.9.npz \
    --tail 20,40 --slosh_every 20 --slosh_e_every 100 --slosh_dt 38 --tag _gpL > quant_gp_N4_a10_r${R}_P220_T20k.log 2>&1
done
date
