#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex


(
    export PYTHONPATH=$PWD

    # IC calibration — phi fixed-point iteration (sink mode, 500 steps/iter)
    date && python3 tst/substrate/probe_value_delta_map.py \
        --R 20 --lam 1.0 --u-central 1.0 \
        --vars phi --phi-mode sink \
        --n-short 500 --max-iter 500 --tol 1e-8 \
        --n-threads 10 \
        --plot-dir tmp/delta_map $@


) 1> >(tee $CWD/.run.sh.out) 2> >(tee $CWD/.run.sh.err >&2)

