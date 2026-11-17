#!/usr/bin/env bash
# 10-25 ramped push (cosine on / off over 50) on the local-flux hold, dt 0.05; q 4 both senses, grad 0.005 / 0.01
cd /home/p/git/eq/exp/2026/10/25/jigsaw
export OMP_NUM_THREADS=1
C="--n 64 --R 10 --a 4 --tpre 60 --dmax 0.9 --T 400 --depth 0.8 --src local --q 4 --dt 0.05 --every 4 --ton 50 --toff 300 --ramp 50"
printf "%s\n" "g005p --grad 0.005" "g005m --grad 0.005 --sense -1" "g01p --grad 0.01" "g01m --grad 0.01 --sense -1" |
  xargs -P 4 -L 1 sh -c 'python3 -u ring2.py '"$C"' "$@" > r9_$0.txt 2>&1'
