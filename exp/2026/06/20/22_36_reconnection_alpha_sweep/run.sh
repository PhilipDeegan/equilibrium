#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

(
    export PYTHONPATH=$PWD

    # Reconnection alpha sweep — calibrate phi→lam coupling before harris_ic_sweep
    # Sweeps lam_alpha ∈ {1,10,30,50}: find where sigma_mid drops below ~0.9 at J_mid peak
    date && python3 tst/substrate/adaptive/test_reconnection_alpha_sweep.py \
        --alpha-values 1,10,30,50 --n-steps 10000 \
        --plot-out tmp/reconnection_alpha_sweep.png

) 1> >(tee $CWD/.run.sh.out) 2> >(tee $CWD/.run.sh.err >&2)
