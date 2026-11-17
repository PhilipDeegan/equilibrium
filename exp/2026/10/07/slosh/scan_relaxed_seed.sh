#!/bin/bash
# 10-07 evening: relaxed-profile seed (10-06 branch/ profile_pp_N2_t8000_tail1) vs GP seed, N 2 r0 30 a 10 P 220 8k,
# same flags as scan_gp.sh -> ringing ~ seed mismatch? the ~2k oscillation / ~5k breathing should not change (coil-local)
cd "$(dirname "$0")"
ROOT=$(cd ../../../../.. && pwd)
date
PYTHONPATH=$ROOT python3 -u spring_box.py --N 2 --a 10 --r0 30 --P 220 --m 0.8 --xi 3.9 --walk_xi --T 8000 \
  --n_plot 8 --recentre 500 --threads 2 --parity --probe_every 200 --project_plus --profile profile_pp_N2_t8000_tail1.npz \
  --tail 20,40 --slosh_every 20 --slosh_e_every 100 --tag _rp > box_rp_N2_a10_r30_P220.log 2>&1
date
