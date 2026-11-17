#!/bin/bash
# spacing scan (next 2026-10-07 item 0b): N 2 r0 30 a 10 at box period P; does the bound extent follow the spacing?
cd /home/p/git/eq/exp/2026/10/06/spacing
for P in 180 220 260 300 340; do
  date
  PYTHONPATH=/home/p/git/eq python3 -u spring_box.py --N 2 --a 10 --r0 30 --P $P --m 0.8 --xi 3.9 --walk_xi --T 8000 \
    --n_plot 8 --recentre 500 --threads 6 --parity --probe_every 200 > box_N2_a10_r30_P${P}.log 2>&1
done
date
