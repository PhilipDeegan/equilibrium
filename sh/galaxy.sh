#!/usr/bin/env bash
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..
set -ex
:> CLAWD
export PYTHONPATH=$PWD

python3 -Ou run/substrate/galaxy_sandbox.py \
    --R 30 --r-bulge 3 --r-disk 10 --h-disk 1.5 \
    --chi-center 0.8 --chi-disk 0.15 \
    --v-rot 0.05 --alpha 0.005 \
    --n-steps 100000 --report 5000 --threads 6 \
    --save-prefix galaxy
