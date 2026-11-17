#!/usr/bin/env bash
# 10-25 mass scaling / universality of fall: same ramped depletion gradient (0.01, on t 50-300) on bodies of different
# size (R 8 / 10 / 13, depth-normalised), count (3 / 5) and energy at fixed coupling (A 1 / 1.15, kfix from r9)
cd /home/p/git/eq/exp/2026/10/25/jigsaw
export OMP_NUM_THREADS=1
C="--tpre 60 --dmax 0.9 --T 400 --depth 0.8 --src local --dt 0.05 --every 4 --ton 50 --toff 300 --ramp 50 --grad 0.01"
printf "%s\n" \
 "R8 --n 80 --R 8 --a 3.2 --q 4" \
 "R10 --n 80 --R 10 --a 4 --q 4" \
 "R13 --n 80 --R 13 --a 5.2 --q 4" \
 "q3 --n 80 --R 10 --a 4 --q 3" \
 "q5 --n 80 --R 10 --a 4 --q 5" \
 "A1.00 --n 64 --R 10 --a 4 --q 4 --A 1.0 --kfix 1.319e-2" \
 "A1.15 --n 64 --R 10 --a 4 --q 4 --A 1.15 --kfix 1.319e-2" |
  xargs -P 7 -L 1 sh -c 'python3 -u ring2.py '"$C"' "$@" > r10_$0.txt 2>&1'
