#!/usr/bin/env bash
# 10-20 well: does a sigma (lam) well around the ring give a wave <-> circulation channel? linear store, relaxed_A seed,
# sigma dip DEPTH 0.3 WIDTH 6 centred on the ring; with / without the plane pulse (AMP 0.3); bump (DEPTH -0.3) + pulse
cd /home/p/git/eq
export PYTHONPATH=/home/p/git/eq
B=/bk/dst/2026/10/20/cap
O=exp/2026/10/20/cap/well
Z="python3 -u exp/2026/10/18/zoo/zoo.py --R 9 --P 96 --T 25 --threads 2 --ediv --seed $B/relaxed_A.npz"
$Z --dip 0.3 0 6 --pulse 0.3 15 3 --ckpt $B/ckpt_well_p.npz > $O/dip_pulse.txt 2>&1 &
$Z --dip 0.3 0 6 --ckpt $B/ckpt_well.npz > $O/dip.txt 2>&1 &
$Z --dip -0.3 0 6 --pulse 0.3 15 3 --ckpt $B/ckpt_bump_p.npz > $O/bump_pulse.txt 2>&1 &
$Z --dip -0.3 0 6 --ckpt $B/ckpt_bump.npz > $O/bump.txt 2>&1 &
wait
