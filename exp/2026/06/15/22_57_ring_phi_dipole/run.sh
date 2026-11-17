#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex


(
    export PYTHONPATH=$PWD

    # Lorentz phi-B sweep (weak B) + radial phi profile (dipole check)
    date && python3 tst/substrate/inspect_profile_ring.py \
        --start tmp/profile_ring_start.npz \
        --end   tmp/profile_ring_end.npz \
        --r-shell-cells 4.0 6.0 8.0 \
        --out inspect_profile_ring.png

) 1> >(tee $CWD/.run.sh.out) 2> >(tee $CWD/.run.sh.err >&2)

