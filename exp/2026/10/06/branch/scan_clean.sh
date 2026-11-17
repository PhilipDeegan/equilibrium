#!/bin/bash
# next 10-07 (A): clean seed (+ projected, relaxed core, tail -> 1 by d 40) across the spacing: box vs seed tail separated
cd /home/p/git/eq/exp/2026/10/06/branch
for P in 220 340; do
  date
  PYTHONPATH=/home/p/git/eq python3 -u spring_box.py --N 2 --a 10 --r0 30 --P $P --m 0.8 --xi 3.9 --walk_xi --T 8000 \
    --n_plot 8 --recentre 500 --threads 5 --parity --probe_every 200 --project_plus --profile profile_pp_N2_t8000_tail1.npz \
    --tag _tail1 > box_clean_N2_a10_r30_P${P}.log 2>&1
done
date
