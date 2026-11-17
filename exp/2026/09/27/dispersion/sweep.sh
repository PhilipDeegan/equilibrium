#!/bin/bash
cd "$(dirname "$0")"
for h in 0.5 0.25 0.125; do
  for d in 1,0,0 1,1,1; do
    for m in T L; do
      PYTHONPATH=/home/p/git/eq python3 -u plane_wave.py --h $h --dir $d --mode $m --threads 8
    done
  done
done
