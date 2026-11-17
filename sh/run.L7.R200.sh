#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

(
    export PYTHONPATH=$PWD

    # R=200/max-level=7 alternative: doubling R (more level-0 cells across
    # the same fixed physical domain, from the Bohr-radius calibration)
    # halves the physical size of a level-0 cell, which lets one FEWER
    # refinement level reach the same >=8-cells-across-the-neutron-diameter
    # threshold (L7 gives ~11.9 cells at R=200 vs needing L8's ~17.9 cells
    # at R=100 -- see doc/mems/project_odd27_amr_junction_leak.md and the
    # 2026-07-22 session discussion). L7 subcycles 3^7=2187 times at the
    # deepest level per coarse step, vs L8's 3^8=6561 -- a real cost lever,
    # traded off against ~8x more level-0 cells overall (R=200 vs 100 is
    # ~8x the sphere volume) -- net runtime effect UNTESTED, this run is
    # that test.
    #
    # eta_h_per_level's level-7 coefficient (1e-13) is an INTERPOLATED
    # ESTIMATE, not directly bisected like level 6's confirmed 1e-11 or
    # level 8's confirmed 1e-15 (project_odd27_amr_junction_leak Part 10) --
    # geometric mean of those two (sqrt(1e-11*1e-15)=1e-13), which lines up
    # almost exactly with the theoretical 1/ratio^4=1/81-per-level scaling
    # derived from D_i~1/L^4 (Part 11). Treat this run's own stability as
    # the first real check of that estimate -- if it diverges, it needs the
    # same isolated per-level bisection level 6 and 8 already got, not
    # another guess.
    OUT_DIR="$PWD/tmp/refined_neutron_decay_R200_L7"
    mkdir -p "$OUT_DIR"

    date
    python3 -u run/substrate/adaptive/refined_neutron_decay.py \
        --R 200 --max-level 7 \
        --n-steps 1 --report 10 --n-threads 8 \
        --eta-h-per-level 0 0 0 0 0 0 0 1e-13 \
        --save-snapshot-dir "$OUT_DIR" \
        --out "$OUT_DIR/plot.png" \
        > "$OUT_DIR/run.log" 2>&1
    date
    echo "=== final summary (R=200/L7) ==="
    tail -20 "$OUT_DIR/run.log"

) 1> >(tee $CWD/.run.sh.out) 2> >(tee $CWD/.run.sh.err >&2)
