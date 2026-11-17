#!/bin/bash
# 10-07 evening: N 4 r0 30 a 10 P 220 resumed 20k -> 40k -- does the helix contraction stop at a finite rad (twisted ring) or unwind?
cd "$(dirname "$0")"
ROOT=$(cd ../../../../.. && pwd)
SNAP=$([ -d /bk/dst ] && echo /bk/dst || echo $HOME/tmp/dst)/2026/10/07/slosh
date
PYTHONPATH=$ROOT python3 -u spring_box.py --N 4 --a 10 --r0 30 --P 220 --m 0.8 --xi 3.9 --walk_xi --T 40000 \
  --n_plot 20 --recentre 500 --threads 8 --parity --probe_every 200 --project_plus --profile gp_profile_xi3.9.npz \
  --tail 20,40 --slosh_every 20 --slosh_e_every 100 --slosh_dt 38 --tag _gpL \
  --resume $SNAP/prof_pp_springbox_N4_S1_q1_a10_r30_xi3.9_m0.8_P220_z0_wxi_rc_par_gpL/state_t20000.npz \
  > quant_gp_N4_a10_r30_P220_from20000.log 2>&1
date
