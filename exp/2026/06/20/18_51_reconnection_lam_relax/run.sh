#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

(
    export PYTHONPATH=$PWD

    # Counter-rotating spheromak reconnection — local lam relaxation toward alpha*|phi|
    # lam_target_i = alpha * |phi_i|, dlam/dt = rate * (lam_target - lam)
    # No transport between cells — physically correct for substrate displacement proxy
    date && python3 tst/substrate/adaptive/test_counter_rotating_spheromak_reconnection.py \
        --n-steps 10000 --lam-relax-rate 0.1 --lam-alpha 1.0 \
        --plot-out tmp/reconnection_lam_relax.png

) 1> >(tee $CWD/.run.sh.out) 2> >(tee $CWD/.run.sh.err >&2)
