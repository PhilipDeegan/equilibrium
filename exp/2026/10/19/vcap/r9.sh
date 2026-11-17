#!/usr/bin/env bash
# 10-19 vcap R 9: lone ring, smooth carrier bound vcap (core max |a| 0.61, C0 2) = ring above capacity; P 96, T 20, 4 x 2 threads
cd /home/p/git/eq
export PYTHONPATH=/home/p/git/eq
B=/bk/dst/2026/10/19/vcap
for v in 0.6 0.3 0.15 0.075; do
  python3 -u exp/2026/10/18/zoo/zoo.py --R 9 --P 96 --T 20 --threads 2 --vcap $v --seed $B/seed_ring_R9_P96.npz \
    --ckpt $B/ckpt_v$v.npz --save $B/end_v$v.npz > exp/2026/10/19/vcap/r9/v$v.txt 2>&1 &
done
wait
