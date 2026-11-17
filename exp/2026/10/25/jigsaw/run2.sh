#!/usr/bin/env bash
# 10-25 lopsided seeds in a held depression (well 0.4) +- throughput feedback; does the lobe count move to even?
cd /home/p/git/eq/exp/2026/10/25/jigsaw
export OMP_NUM_THREADS=1
for q in 3 4 4.5 5; do for k in 0 0.5; do echo "$q $k"; done; done |
  xargs -P 8 -L 1 sh -c 'python3 -u ring.py --n 64 --R 10 --a 4 --q $0 --kappa $1 --preseed --well 0.4 --T 300 > r2_q$0_k$1.txt 2>&1'
