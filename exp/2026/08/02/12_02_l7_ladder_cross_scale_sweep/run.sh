#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

(
    export PYTHONPATH=$PWD

    # cross_scale=1e-3 sanity check COMPLETE (see doc/mems/project_odd27_amr_junction_leak.md
    # Part 17, exp/2026/08/01/16_15_l7_ladder_cross_scale_sanity_check): the full
    # provisional L0-L7 ladder (0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13, earned under the
    # fully-isolated harness) survived -- H/H0 bounded ~[10.3, 57.6] over 150 steps,
    # essentially identical to the pure isolated-harness result. Only the smallest
    # candidate of a planned sweep, though -- not proof the ladder holds at larger
    # cross_scale. Next: sweep upward by decade (1e-2, 1e-1, 1e0) against the SAME
    # fixed ladder to find where (if anywhere) it actually breaks, before touching
    # L8 (which needs a validated L7 base and costs hours regardless of outcome).
    # --cross-scales takes multiple values directly -- the script sweeps them in one
    # process and, thanks to --blowup-factor (default 100), continues past any
    # candidate that diverges instead of burning the full --n-steps on it.
    OUT_DIR="$PWD/exp/2026/08/02/12_02_l7_ladder_cross_scale_sweep"
    mkdir -p "$OUT_DIR"

    date
    echo "=== cross_scale sweep: full L0-L7 ladder, cross_scale=1e-2,1e-1,1e0 (lam still frozen, damp_max still 0) ==="
    python3 -u run/substrate/adaptive/refined_neutron_decay.py \
        --R 20 --max-level 7 --region-r0 5.0 \
        --scale-override 1.3606e-3 --scale-ref-level 8 \
        --lam-target 1.0 --lam-scale 0.05 --lam-mode frozen \
        --cross-scales 1e-2 1e-1 1e0 --damp-max 0 \
        --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 \
        --n-steps 150 --report 15 --n-threads 4 \
        > "$OUT_DIR/cross_scale_sweep.log" 2>&1
    date
    echo "=== finished cross_scale sweep ==="
    tail -40 "$OUT_DIR/cross_scale_sweep.log"

) 1> >(tee $CWD/.run.sh.out) 2> >(tee $CWD/.run.sh.err >&2)
