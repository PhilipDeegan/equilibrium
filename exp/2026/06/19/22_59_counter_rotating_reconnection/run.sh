#!/usr/bin/env bash
# Counter-rotating spheromak reconnection — omega=0.1, lam-target=1.0
# Run date: 19 Jun 2026 22:59 CEST

PYTHONPATH=$PWD python3 tst/substrate/adaptive/test_counter_rotating_spheromak_reconnection.py \
    --R 60 --scale 12 --z-sep 16 \
    --omega 0.1 --n-steps 20000 --report 500 \
    --cross-scale 0.005 --lam-scale 0.05 \
    --plot-out tmp/reconnection.png
