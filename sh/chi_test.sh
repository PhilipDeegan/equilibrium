#!/usr/bin/env bash
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..
set -ex
export PYTHONPATH=$PWD

BASE="--R 30 --r-bulge 3 --r-disk 10 --h-disk 1.5
      --chi-center 0.8 --chi-disk 0.15
      --alpha 0.005 --n-steps 50000 --report 5000 --threads 6"

echo "=== A: no phi ==="
python3 -Ou run/substrate/galaxy_sandbox.py $BASE --v-rot 0.0   --save-prefix /tmp/chi_A

echo "=== B: tiny flat phi ==="
python3 -Ou run/substrate/galaxy_sandbox.py $BASE --v-rot 0.001 --save-prefix /tmp/chi_B

echo "=== C: tiny Keplerian phi ==="
python3 -Ou run/substrate/galaxy_sandbox.py $BASE --v-rot 0.001 --keplerian --save-prefix /tmp/chi_C
