#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

(
    export PYTHONPATH=$PWD

    # Stern-Gerlach: spin +1 vs spin -1 deflection under phi_z gradient.
    # Chirality = rotation direction of the whole winding structure (right-hand rule).
    # spin +1: counterclockwise about z (omega > 0), spin -1: clockwise (omega < 0).
    date && python3 tst/substrate/adaptive/test_stern_gerlach.py \
        --load proton_candidate_A_R40_IC.npz \
        --omega 0.05 --gradient 1e-5 --n-steps 30000 --report 1000 \
        --cross-scale 0.005 --lam-scale 0.05 \
        --sustain-spin \
        --plot-out tmp/stern_gerlach.png


) 1> >(tee $CWD/.run.sh.out) 2> >(tee $CWD/.run.sh.err >&2)


