#!/usr/bin/env bash
# 10-21 valve: tilted ring pair (ang 45, gap 0.25 R), linear store, R 20 P 192 -- how long until contact / reconnection
cd /home/p/git/eq
export PYTHONPATH=/home/p/git/eq OMP_NUM_THREADS=1
B=/bk/dst/2026/10/21/valve
mkdir -p $B
python3 -u exp/2026/10/18/zoo/zoo.py --struct side_tilt --R 20 --P 192 --smooth 4.938 --T 120 --every 2 --threads 8 \
  --seed $B/seed_tilt_R20.npz --save $B/end_tilt_R20.npz --ckpt_every 3 > exp/2026/10/21/valve/r20_tilt.txt 2>&1
