#!/bin/bash
# quantization scan: shed fraction vs continuous a at fixed N 2, r0 30, P 220, 8000 ticks
cd /home/p/git/eq/exp/2026/10/06/quant
for a in 6 8 12 14; do
  date
  PYTHONPATH=/home/p/git/eq python3 -u spring_box.py --N 2 --a $a --r0 30 --P 220 --m 0.8 --xi 3.9 --walk_xi --T 8000 \
    --n_plot 4 --recentre 500 --threads 8 --parity --snap_root /bk/dst/2026/10/06/quant > box_N2_a${a}_r30_P220.log 2>&1
done
date
