#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

(
    export PYTHONPATH=$PWD

    # L7 boundedness sweep COMPLETE (see doc/mems/project_odd27_amr_junction_leak.md
    # Part 16/17): 3e-14/1e-13 stable, 3e-13/1e-12 diverge (1e-12 worse+faster than
    # 3e-13 -- monotonic past the cliff, no point testing higher). Provisional L0-L7
    # ladder: 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13.
    #
    # ALL of the above (L1-L7) was earned under the fully-isolated harness
    # (--cross-scales 0 --lam-mode frozen --damp-max 0) -- deliberately, to
    # separate the AMR-boundary H-growth from real physics, same isolate-one-
    # variable-at-a-time methodology as the rest of this investigation (see
    # Part 2's step_cross/step_lam/sponge isolation). Next step, in that same
    # spirit: turn ON only cross_scale (leave lam_mode frozen, damp_max 0) and
    # check whether the existing isolated-harness ladder survives -- cheaper
    # than assuming a full re-earn from L1 is needed, and tells us which
    # levels (if any) actually need re-tuning against real cross_scale physics.
    # --blowup-factor (new this session, default 100) aborts early on
    # divergence instead of burning hours the way 3e-13/1e-12 did above.
    OUT_DIR="$PWD/exp/2026/08/01/16_15_l7_ladder_cross_scale_sanity_check"
    mkdir -p "$OUT_DIR"

    date
    echo "=== sanity check: full L0-L7 ladder, cross_scale=1e-3 (lam still frozen, damp_max still 0) ==="
    python3 -u run/substrate/adaptive/refined_neutron_decay.py \
        --R 20 --max-level 7 --region-r0 5.0 \
        --scale-override 1.3606e-3 --scale-ref-level 8 \
        --lam-target 1.0 --lam-scale 0.05 --lam-mode frozen \
        --cross-scales 1e-3 --damp-max 0 \
        --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 \
        --n-steps 150 --report 15 --n-threads 4 \
        > "$OUT_DIR/cross_scale_1e-3.log" 2>&1
    date
    echo "=== finished cross_scale sanity check ==="
    tail -20 "$OUT_DIR/cross_scale_1e-3.log"

) 1> >(tee $CWD/.run.sh.out) 2> >(tee $CWD/.run.sh.err >&2)
