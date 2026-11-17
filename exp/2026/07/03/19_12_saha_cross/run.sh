#!/bin/bash
date && PYTHONPATH=$PWD python3 -u tst/substrate/probe_saha_cross.py \
    --R 20 --offset 6 --n-steps 8000 --report 800 --out saha_cross.png
