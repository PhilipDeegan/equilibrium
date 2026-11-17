#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

(
    export PYTHONPATH=$PWD

    m0=2.49
    L=0
    R=80
    cs=1e-3
    B="decay_L${L}_R${R}_m0${m0}_cs${cs}"
    mkdir -p "$PWD/tmp/${B}_snapshots" "$PWD/tmp/${B}_plots"

    date && python3 -Ou tst/substrate/adaptive/sweep_neutron_decay_cross_scale.py \
        --R $R --n-steps 4000 --report 400 \
        --cross-scales $cs --n-threads 8 --m0 $m0 \
        --save-snapshot-dir "$PWD/tmp/${B}_snapshots" --restart-from "$PWD/tmp/${B}_snapshots" \
        --plot-dir "$PWD/tmp/${B}_plots" --plot-every 500 \
        --out "$PWD/tmp/spheromak_${B}.png" --max-level $L

) 1> >(tee $CWD/.run.sh.out) 2> >(tee $CWD/.run.sh.err >&2)
