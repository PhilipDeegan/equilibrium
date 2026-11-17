#!/bin/bash
# 10-07 quant in N: resume N 4 / N 3 (r0 30 a 10 P 220, still losing Et at 4k) to 10k -- do they shed turns to a steady N?
cd "$(dirname "$0")"
ROOT=$(cd ../../../../.. && pwd)
SNAP=$([ -d /bk/dst ] && echo /bk/dst || echo $HOME/tmp/dst)/2026/10/07/slosh
for N in 4 3; do
  date
  PYTHONPATH=$ROOT python3 -u spring_box.py --N $N --a 10 --r0 30 --P 220 --m 0.8 --xi 3.9 --walk_xi --T 10000 \
    --n_plot 10 --recentre 500 --threads 8 --parity --probe_every 200 --project_plus --profile gp_profile_xi3.9.npz \
    --tail 20,40 --slosh_every 20 --slosh_e_every 100 --slosh_dt 38 --tag _gpN \
    --resume $SNAP/prof_pp_springbox_N${N}_S1_q1_a10_r30_xi3.9_m0.8_P220_z0_wxi_rc_par_gpN/state_t04000.npz \
    > quant_gp_N${N}_a10_r30_P220_from4000.log 2>&1
done
date
