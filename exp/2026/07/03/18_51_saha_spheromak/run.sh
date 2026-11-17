#!/bin/bash
# Short diagnostic run
date && PYTHONPATH=$PWD python3 -u tst/substrate/probe_saha_spheromak.py

# Long run (logged)
date && PYTHONPATH=$PWD python3 -u tst/substrate/probe_saha_spheromak.py \
    --R 25 --n-steps 20000 --report 1000 --out saha_spheromak_long.png
