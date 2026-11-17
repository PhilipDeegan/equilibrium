#!/usr/bin/env bash
# 10-22 spheromak baseline, step() + Cartan, a 9 / 20 (smooth scaled (a / 9)^2), T 200, 8 threads, sequential
cd /home/p/git/eq
export PYTHONPATH=/home/p/git/eq OMP_NUM_THREADS=1
B=/bk/dst/2026/10/22/sph
mkdir -p $B
python3 -u exp/2026/10/22/spheromak/sph.py --P 96 --a 9 --smooth 1 --T 200 --every 5 --threads 8 \
  --seed $B/seed_a9.npz --save $B/end_a9.npz > exp/2026/10/22/spheromak/a9.txt 2>&1
python3 -u exp/2026/10/22/spheromak/sph.py --P 192 --a 20 --smooth 4.938 --T 200 --every 5 --threads 8 \
  --seed $B/seed_a20.npz --save $B/end_a20.npz > exp/2026/10/22/spheromak/a20.txt 2>&1
