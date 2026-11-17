#!/usr/bin/env bash
# 10-20 cap push: ring held at the ceiling (relaxed_A, S_c 0.38626, p 4) hit by a plane u pulse (DIST 15, WIDTH 3)
# AMP 0.1 / 0.3 / 0.5 with the store, AMP 0.3 linear control; P 96, T 25, 4 x 2 threads
cd /home/p/git/eq
export PYTHONPATH=/home/p/git/eq
B=/bk/dst/2026/10/20/cap
O=exp/2026/10/20/cap/push
mkdir -p $O
Z="python3 -u exp/2026/10/18/zoo/zoo.py --R 9 --P 96 --T 25 --threads 2 --ediv --seed $B/relaxed_A.npz"
$Z --Sc 0.38626 --cap_p 4 --smax 1.15 --pulse 0.1 15 3 --ckpt $B/ckpt_push0.1.npz --save $B/end_push0.1.npz > $O/a0.1.txt 2>&1 &
$Z --Sc 0.38626 --cap_p 4 --smax 1.4 --pulse 0.3 15 3 --ckpt $B/ckpt_push0.3.npz --save $B/end_push0.3.npz > $O/a0.3.txt 2>&1 &
$Z --Sc 0.38626 --cap_p 4 --smax 1.65 --pulse 0.5 15 3 --ckpt $B/ckpt_push0.5.npz --save $B/end_push0.5.npz > $O/a0.5.txt 2>&1 &
$Z --pulse 0.3 15 3 --ckpt $B/ckpt_push0.3_lin.npz --save $B/end_push0.3_lin.npz > $O/a0.3_linear.txt 2>&1 &
wait
