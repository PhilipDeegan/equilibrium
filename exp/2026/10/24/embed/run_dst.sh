#!/usr/bin/env bash
# 10-24 embedded spheromak, DST a 6 P 96, T 150 (7.5 turnovers), 4 at a time x 2 threads
cd /home/p/git/eq/exp/2026/10/24/embed
export PYTHONPATH=/home/p/git/eq OMP_NUM_THREADS=1
B=/bk/dst/2026/10/24/embed
mkdir -p $B
printf "%s\n" "0 aligned" "0.1 aligned" "0.1 opposed" "0.1 tilt45" "0.3 aligned" "0.3 opposed" "0.3 tilt45" |
  xargs -P 4 -L 1 sh -c 'python3 -u dst_embed.py --frac $0 --orient $1 --threads 2 --save '$B'/dst_$0_$1.npz > dst_$0_$1.txt 2>&1'
