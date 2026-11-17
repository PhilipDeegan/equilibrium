#!/bin/bash
# quantization scan: shed fraction vs continuous r0 at fixed N 2, a 10, P 220, 8000 ticks
cd /home/p/git/eq/exp/2026/10/06/quant
for r0 in 30 20 24 27 33 36 40; do
  date
  PYTHONPATH=/home/p/git/eq python3 -u spring_box.py --N 2 --a 10 --r0 $r0 --P 220 --m 0.8 --xi 3.9 --walk_xi --T 8000 \
    --n_plot 4 --recentre 500 --threads 8 --parity --snap_root /bk/dst/2026/10/06/quant > box_N2_a10_r${r0}_P220.log 2>&1
done
date
