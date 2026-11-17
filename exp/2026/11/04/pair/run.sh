#!/usr/bin/env bash
# 11-04 pair regime map: separation +-24 (n 128), +-30 (n 144), opposite senses at +-18 (n 112); each with solos.
set -u
cd "$(dirname "$0")"
O=/bk/dst/2026/11/04/pair
mkdir -p "$O"
C="--R 8 --a 3.2 --q 4 --depth 0.8 --src local --dmax 0.9 --dt 0.05 --T 300 --snap 100,200,300"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run() { name=$1; shift; python3 -u pair.py $C "$@" --out $O/$name.npz > $name.log 2>&1 & }
run pair24   --n 128 --cx=-24,24
run solo_m24 --n 128 --cx=-24
run solo_p24 --n 128 --cx=24
run pair30   --n 144 --cx=-30,30
run solo_m30 --n 144 --cx=-30
run solo_p30 --n 144 --cx=30
run opp18    --n 112 --cx=-18,18 --senses 1,-1
run solo_p18n --n 112 --cx=18 --senses -1
wait
