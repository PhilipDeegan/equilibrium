#!/usr/bin/env bash
cd /home/p/git/eq
export PYTHONPATH=$PWD
D=exp/2026/10/14/cartan
for c in 1 1.003 0.997 1.01 0.99 1.03 0.97 1.1 0.9; do
  date
  python3 -u $D/vill.py --P 72 --R0 9 --T 20 --every 2 --smooth 1 --off 0.13 0.07 0.11 --flow cartan --kappa 1 --threads 1 --cpp --midpoint 2 --circ $c 2>/dev/null > $D/valve/R9_circ$c.log
done
