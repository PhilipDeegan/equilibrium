#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

(
    export PYTHONPATH=$PWD

    # Exp 2+3: Hopf decay to mid-state, snapshot, then freeze (cs=0) and check stability.
    # Per-ring circ_A/circ_B columns show proton/electron handedness separation.
    date && python3 -Ou tst/substrate/test_spheromak_decay.py \
        --rings both --cross-scale 0.001 --lam-mode dynamic \
        --n-steps 12000 --report 1000 \
        --save-snapshot proton_candidate.npz \
        --plot-out spheromak_decay_snapshot.png $@

    date && python3 -Ou tst/substrate/test_spheromak_decay.py \
        --load proton_candidate.npz --cross-scale 0 --lam-mode dynamic \
        --n-steps 8000 --report 1000 \
        --plot-out spheromak_decay_frozen.png $@

) 1> >(tee $CWD/.run.sh.out) 2> >(tee $CWD/.run.sh.err >&2)


