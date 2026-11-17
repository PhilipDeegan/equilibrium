#!/usr/bin/env bash
# 10-25 requirement test, propagating depletion carrier P (ring2.py), self-consistent preseed, depth 0.8 (dmax 0.9)
cd /home/p/git/eq/exp/2026/10/25/jigsaw
export OMP_NUM_THREADS=1
C="--n 64 --R 10 --a 4 --tpre 60 --dmax 0.9 --T 400"
printf "%s\n" \
 "A --q 4 --depth 0.8" \
 "B --q 4 --depth 0.8 --grad 0.005 --ton 50 --toff 200" \
 "C --q 4 --depth 0.8 --grad 0.005 --ton 50 --toff 200 --sense -1" \
 "D --q 5 --depth 0.8" \
 "E --q 5 --depth 0.8 --grad 0.005 --ton 50 --toff 200" \
 "F --q 3 --depth 0.8" \
 "G --q 4 --depth 0.6" |
  xargs -P 7 -L 1 sh -c 'python3 -u ring2.py '"$C"' "$@" > r5_$0.txt 2>&1'
