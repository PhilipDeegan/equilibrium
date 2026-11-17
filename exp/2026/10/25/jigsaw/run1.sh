#!/usr/bin/env bash
# 10-25 q x kappa matrix, preseeded depression, n 64 R 10 a 4 T 300 (~10 transits)
cd /home/p/git/eq/exp/2026/10/25/jigsaw
export OMP_NUM_THREADS=1
for q in 3 4 5; do for k in 0 0.1 0.5; do echo "$q $k"; done; done |
  xargs -P 9 -L 1 sh -c 'python3 -u ring.py --n 64 --R 10 --a 4 --q $0 --kappa $1 --preseed --T 300 > r1_q$0_k$1.txt 2>&1'
