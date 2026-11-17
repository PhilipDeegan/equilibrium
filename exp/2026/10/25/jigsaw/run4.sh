#!/usr/bin/env bash
# 10-25 deformation / translation under an external pressure gradient: held co-moving well 0.4 + feedback 0.5,
# background depletion gradient along x (contracted toward +x) on until t 150, off after; T 300
cd /home/p/git/eq/exp/2026/10/25/jigsaw
export OMP_NUM_THREADS=1
for q in 3 4 5; do for g in 0 0.005; do echo "$q $g"; done; done |
  xargs -P 6 -L 1 sh -c 'python3 -u ring.py --n 64 --R 10 --a 4 --q $0 --kappa 0.5 --preseed --well 0.4 --follow --grad $1 --toff 150 --T 300 > r4_q$0_g$1.txt 2>&1'
