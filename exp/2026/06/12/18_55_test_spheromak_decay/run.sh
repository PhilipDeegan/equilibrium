#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

(
    export PYTHONPATH=$PWD

    # R=40: Hopf decay to mid-state, snapshot, then long frozen run to let
    # exterior phi radiation clear and check whether H asymptotes.
    date && python3 -Ou tst/substrate/test_spheromak_decay.py \
        --R 40 --rings both --cross-scale 0.001 --lam-mode dynamic \
        --n-steps 12000 --report 2000 \
        --save-snapshot proton_candidate_R40.npz \
        --plot-out spheromak_R40_snapshot.png $@

    date && python3 -Ou tst/substrate/test_spheromak_decay.py \
        --R 40 --load proton_candidate_R40.npz --cross-scale 0 --lam-mode dynamic \
        --n-steps 16000 --report 2000 \
        --plot-out spheromak_R40_frozen.png $@

) 1> >(tee $CWD/.run.sh.out) 2> >(tee $CWD/.run.sh.err >&2)


