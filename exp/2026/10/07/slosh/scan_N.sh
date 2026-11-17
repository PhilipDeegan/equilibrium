#!/bin/bash
# 10-07 quant in N: GP clean seed (+ projected) at fixed geometry r0 30 a 10, P 220 4k; Et within a + 28 (quant_table.py)
cd "$(dirname "$0")"
ROOT=$(cd ../../../../.. && pwd)
while pgrep -f "scan_pair.sh" > /dev/null; do sleep 20; done
for N in 1 3 4; do
  date
  PYTHONPATH=$ROOT python3 -u spring_box.py --N $N --a 10 --r0 30 --P 220 --m 0.8 --xi 3.9 --walk_xi --T 4000 \
    --n_plot 4 --recentre 500 --threads 8 --parity --probe_every 200 --project_plus --profile gp_profile_xi3.9.npz \
    --tail 20,40 --slosh_every 20 --slosh_e_every 100 --slosh_dt 38 --tag _gpN > quant_gp_N${N}_a10_r30_P220.log 2>&1
done
date
