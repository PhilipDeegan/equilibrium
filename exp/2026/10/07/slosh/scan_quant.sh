#!/bin/bash
# 10-07 quant: r0 / a scan with the GP clean seed (+ projected), N 2 P 220 4k; Et_in = E_in + g/2 D_in within a + 28
cd "$(dirname "$0")"
ROOT=$(cd ../../../../.. && pwd)
while pgrep -f "scan_gp.sh|scan_pair.sh" > /dev/null; do sleep 20; done
for ra in "20 10" "25 10" "35 10" "40 10" "30 6" "30 8" "30 12" "30 14"; do
  set -- $ra
  date
  PYTHONPATH=$ROOT python3 -u spring_box.py --N 2 --a $2 --r0 $1 --P 220 --m 0.8 --xi 3.9 --walk_xi --T 4000 \
    --n_plot 4 --recentre 500 --threads 8 --parity --probe_every 200 --project_plus --profile gp_profile_xi3.9.npz \
    --tail 20,40 --slosh_every 20 --slosh_e_every 100 --slosh_dt $(( $2 + 28 )) --tag _gpq > quant_gp_N2_a$2_r$1_P220.log 2>&1
done
date
