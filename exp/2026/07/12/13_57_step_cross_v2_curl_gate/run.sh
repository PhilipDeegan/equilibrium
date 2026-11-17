#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

(
    export PYTHONPATH=$PWD

    mkdir -p "$PWD/tmp/electron_dipole_R80"

    date && python3 -Ou tst/substrate/adaptive/test_electron_dipole.py \
        --R 80 --n-steps 5000 --report 500 --n-threads 10 \
        --data-out "$PWD/tmp/electron_dipole_R80/edip_r80.npz" \
        --save-snapshot "$PWD/tmp/electron_dipole_R80/edip_r80_final.npz" \
        --plot-out "$PWD/tmp/electron_dipole_R80/electron_dipole_r80.png"

) 1> >(tee $CWD/.run.sh.out) 2> >(tee $CWD/.run.sh.err >&2)
