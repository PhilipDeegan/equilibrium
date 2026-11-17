#!/usr/bin/env bash
# 10-19 csig R 20: sigma in the Cartan transport; 4 x 1 thread (4-core cap)
cd /home/p/git/eq
export PYTHONPATH=/home/p/git/eq
Z="python3 -u exp/2026/10/18/zoo/zoo.py --R 20 --P 140 --T 30 --smooth 4.938 --seed /bk/dst/2026/10/18/zoo/seed_ring_R20.npz"
O=exp/2026/10/19/csig/r20 B=/bk/dst/2026/10/19/csig
run() { n=$1; shift; $Z "$@" --ckpt $B/ckpt_$n.npz --save $B/end_$n.npz > $O/$n.txt 2>&1; }
run tilt0.03_csig --tilt 0.03 --csig &
run tilt0.1_csig --tilt 0.1 --csig &
run dip_off --dip 0.1 25 6 &
run dip_csig --dip 0.1 25 6 --csig &
wait
