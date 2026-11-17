#!/usr/bin/env bash
# 10-25 signed depletion (contract u > 0, expand u < 0) on top of held well 0.4 + |throughput| kappa 0.5; dt 0.05, every 4
cd /home/p/git/eq/exp/2026/10/25/jigsaw
export OMP_NUM_THREADS=1
for q in 3 4 4.5 5; do for s in 0.1 0.3; do echo "$q $s"; done; done |
  xargs -P 8 -L 1 sh -c 'python3 -u ring.py --n 64 --R 10 --a 4 --q $0 --kappa 0.5 --ksig $1 --preseed --well 0.4 --dt 0.05 --every 4 --T 300 > r3_q$0_s$1.txt 2>&1'
