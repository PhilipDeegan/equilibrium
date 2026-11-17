#!/usr/bin/env bash
# 10-25 requirements + counts on the local-tick-flux hold (ring2.py --src local, depth 0.8)
cd /home/p/git/eq/exp/2026/10/25/jigsaw
export OMP_NUM_THREADS=1
C="--n 64 --R 10 --a 4 --tpre 60 --dmax 0.9 --T 400 --depth 0.8 --src local"
printf "%s\n" \
 "push005p --q 4 --grad 0.005 --ton 50 --toff 250" \
 "push005m --q 4 --grad 0.005 --ton 50 --toff 250 --sense -1" \
 "push02p --q 4 --grad 0.02 --ton 50 --toff 250" \
 "push02m --q 4 --grad 0.02 --ton 50 --toff 250 --sense -1" \
 "q3 --q 3" "q4 --q 4" "q5 --q 5" "q4.5 --q 4.5" |
  xargs -P 8 -L 1 sh -c 'python3 -u ring2.py '"$C"' "$@" > r8_$0.txt 2>&1'
