#!/usr/bin/env bash
# 10-19 recv R 20: equal rings side by side (same / opposite sense) + one alone, P 192 (SEP = side / 2 = 48 lattice periods); 3 + 3 + 2 threads
cd /home/p/git/eq
export PYTHONPATH=/home/p/git/eq
B=/bk/dst/2026/10/19/recv O=exp/2026/10/19/recv/r20
run() {
  n=$1 th=$2; shift 2
  python3 -u exp/2026/10/18/zoo/zoo.py --R 20 --P 192 --T 30 --smooth 4.938 --threads $th --seed $B/seed_$n.npz --ckpt $B/ckpt_$n.npz \
    --save $B/end_$n.npz "$@" > $O/$n.txt 2>&1
}
run pair_same 3 --struct side &
run pair_opp 3 --struct side_opp &
run solo_a 2 --struct side_a &
wait
