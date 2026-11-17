#!/usr/bin/env bash
# 10-20 cap R 9, steeper store: (p, scap) = (2, 0.5) (4, 0.7) (4, 0.5); P 96, T 20, 3 x 2 threads
cd /home/p/git/eq
export PYTHONPATH=/home/p/git/eq
B=/bk/dst/2026/10/20/cap
S=/bk/dst/2026/10/19/vcap/seed_ring_R9_P96.npz
for pc in "2 0.5" "4 0.7" "4 0.5"; do
  set -- $pc
  python3 -u exp/2026/10/18/zoo/zoo.py --R 9 --P 96 --T 20 --threads 2 --scap $2 --cap_p $1 --seed $S \
    --ckpt $B/ckpt_p$1_s$2.npz --save $B/end_p$1_s$2.npz > exp/2026/10/20/cap/r9/p$1_s$2.txt 2>&1 &
done
wait
