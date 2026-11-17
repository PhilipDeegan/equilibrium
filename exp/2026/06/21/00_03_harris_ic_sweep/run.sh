#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

(
    export PYTHONPATH=$PWD

    # Harris IC sweep — alpha={5,10,15} at lam_relax_rate=0.1 vs no-relax baseline
    # 32 configs: B0∈{0.05,0.10} × lam_bg∈{0.4,2.0} × lam_peak∈{0.0,1.0} × (baseline + 3 alphas)
    date && python3 tst/substrate/adaptive/harris_ic_sweep.py \
        --out-dir tmp/harris_sweep_alpha

) 1> >(tee $CWD/.run.sh.out) 2> >(tee $CWD/.run.sh.err >&2)
