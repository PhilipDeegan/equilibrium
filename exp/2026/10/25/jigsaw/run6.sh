#!/usr/bin/env bash
# 10-25 leak diagnosis for v5 A (q 4 depth 0.8): frozen well vs dynamic (flux source) vs dynamic (energy source)
cd /home/p/git/eq/exp/2026/10/25/jigsaw
export OMP_NUM_THREADS=1
C="--n 64 --R 10 --a 4 --tpre 60 --dmax 0.9 --T 400 --q 4 --depth 0.8"
printf "%s\n" "frozen --freeze" "flux --src flux" "energy --src energy" |
  xargs -P 3 -L 1 sh -c 'python3 -u ring2.py '"$C"' "$@" > r6_$0.txt 2>&1'
