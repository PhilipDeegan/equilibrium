#!/usr/bin/env bash
# 10-20 cap: rest-relaxed under the store (p 4, scap 0.7, relax 15): lone P 96 (A), lone P 144 T 12 (B), equal side pair P 96 (C) + linear pair control (D); 4 x 2 threads
cd /home/p/git/eq
export PYTHONPATH=/home/p/git/eq
B=/bk/dst/2026/10/20/cap
O=exp/2026/10/20/cap/relax
mkdir -p $O
Z="python3 -u exp/2026/10/18/zoo/zoo.py --R 9 --threads 2"
CAP="--scap 0.7 --cap_p 4 --relax 15"
$Z --P 96 --T 20 $CAP --seed /bk/dst/2026/10/19/vcap/seed_ring_R9_P96.npz --ckpt $B/ckpt_A.npz --save $B/end_A.npz --save_relaxed $B/relaxed_A.npz > $O/A_lone_P96.txt 2>&1 &
$Z --P 144 --T 12 $CAP --seed $B/seed_ring_R9_P144.npz --ckpt $B/ckpt_B.npz --save $B/end_B.npz --save_relaxed $B/relaxed_B.npz > $O/B_lone_P144.txt 2>&1 &
$Z --P 96 --T 20 --struct side $CAP --seed $B/seed_side_R9_P96.npz --ckpt $B/ckpt_C.npz --save $B/end_C.npz --save_relaxed $B/relaxed_C.npz > $O/C_side_P96.txt 2>&1 &
$Z --P 96 --T 20 --struct side --seed $B/seed_side_R9_P96_lin.npz --ckpt $B/ckpt_D.npz --save $B/end_D.npz > $O/D_side_P96_linear.txt 2>&1 &
wait
