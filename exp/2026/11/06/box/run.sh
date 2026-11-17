#!/usr/bin/env bash
# 11-06 box check: +-24 at n 144 (same box as 11-04 +-30) -> is the non-monotonic approach a box effect?
set -u
cd "$(dirname "$0")"
O=/bk/dst/2026/11/06/box
mkdir -p "$O"
C="--R 8 --a 3.2 --q 4 --depth 0.8 --src local --dmax 0.9 --dt 0.05 --T 300 --snap 100,200,300"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run() { name=$1; shift; taskset -c 0-3,8-11 python3 -u pair.py $C "$@" --out $O/$name.npz > $name.log 2>&1 & }
run pair24   --n 144 --cx=-24,24
run solo_m24 --n 144 --cx=-24
run solo_p24 --n 144 --cx=24
wait
# 4th core: reading b (delayed bridge). First launched from t 0 (pair30_T600_killed.log, matched 11-04 to t 200, killed
# 13:08); now continued from the 11-04 t 300 state (pair.py --restart, bit-exact vs a straight run), checkpoint every 25:
# taskset -c 0-3,8-11 python3 -u pair.py <C without T/snap> --n 144 --cx=-30,30 --T 600 --snap 400,500,600 \
#   --restart /bk/dst/2026/11/04/pair/pair30.npz --t0 300 --eratio 0.9345 --ckpt 25 --out $O/pair30_T600.npz
# resume after a kill: --restart $O/pair30_T600_ckpt.npz --t0 <t in ckpt> (E0 stored, no --eratio)
