#!/usr/bin/env bash
# 10-25 depletion source = flux per LOCAL tick, and an energy + flux mix (q 4 depth 0.8)
cd /home/p/git/eq/exp/2026/10/25/jigsaw
export OMP_NUM_THREADS=1
C="--n 64 --R 10 --a 4 --tpre 60 --dmax 0.9 --T 400 --q 4 --depth 0.8"
printf "%s\n" "local --src local" "mix --src mix" |
  xargs -P 2 -L 1 sh -c 'python3 -u ring2.py '"$C"' "$@" > r7_$0.txt 2>&1'
