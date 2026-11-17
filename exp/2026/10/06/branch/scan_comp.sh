#!/bin/bash
# next 10-07 item 0c: compensated (D_tot = 0), + projected seeds across the spacing; does the halo close locally?
cd /home/p/git/eq/exp/2026/10/06/branch
for P in 180 220 260; do
  date
  PYTHONPATH=/home/p/git/eq python3 -u spring_box.py --N 2 --a 10 --r0 30 --P $P --m 0.8 --xi 3.9 --walk_xi --T 8000 \
    --n_plot 8 --recentre 500 --threads 3 --parity --probe_every 200 --project_plus --compensate > box_comp_pp_N2_a10_r30_P${P}.log 2>&1
done
date
