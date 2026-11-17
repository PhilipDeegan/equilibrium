#!/usr/bin/env bash
# 10-20 cap R 9: lone ring, cell holding capacity S_c = scap x seed max S_j (p 1); scap < 1 = core above capacity; P 96, T 20, 4 x 2 threads
cd /home/p/git/eq
export PYTHONPATH=/home/p/git/eq
B=/bk/dst/2026/10/20/cap
S=/bk/dst/2026/10/19/vcap/seed_ring_R9_P96.npz
for c in 1.0 0.5 0.25 0.1; do
  python3 -u exp/2026/10/18/zoo/zoo.py --R 9 --P 96 --T 20 --threads 2 --scap $c --seed $S \
    --ckpt $B/ckpt_s$c.npz --save $B/end_s$c.npz > exp/2026/10/20/cap/r9/s$c.txt 2>&1 &
done
wait
