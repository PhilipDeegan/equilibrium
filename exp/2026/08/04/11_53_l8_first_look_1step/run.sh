#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/../../../../..

set -ex

(
    export PYTHONPATH=$PWD

    # First-ever L8 run (production target, per CLAUDE.md/doc/dst.md; the L0-L7
    # ladder work in exp/2026/08/03/15_15_l7_dynamic_lam_ladder_test and
    # exp/2026/08/04/11_02_l7_etah_isolation_retest was all in preparation for
    # this). Not a stability test -- 1 coarse step, purely to see actual L8
    # cost (n_leaves, est_mem, wall time/step) before spending any time earning
    # an eta_h value for the new L8 slot.
    #
    # No --scale-override here (unlike the L0-L7 ladder runs): that flag exists
    # only to let a shallower max_level borrow the real L8-calibrated neutron
    # size (scale_ref_level defaults to 8) for pure numerical-stability probes
    # at cheaper levels. At max_level=8 itself we ARE at the reference level,
    # so the script's real calibration path (NEUTRON_RADIUS_M / sim_unit_m)
    # applies directly -- no bypass needed or wanted.
    #
    # eta_h for L0-L7 kept at the earned ladder values; L8's own slot is
    # deliberately left unset (only 8 entries given, for L0-L7) so it defaults
    # to eta_h=0 at L8 -- unbiased for a cost/first-look probe, not a guess.
    #
    # --save-snapshot-dir persists the full leaf state (phi/u/lam/pos/level/
    # ancestry) after the step so we can inspect it (cell distribution near
    # the core, actual resolution achieved, etc.) without re-running the
    # (potentially expensive) L8 mesh build.
    OUT_DIR="$PWD/exp/2026/08/04/11_53_l8_first_look_1step"
    mkdir -p "$OUT_DIR"

    date
    echo "=== L8 first look: cross_scale=1e-3, 1 coarse step, real calibration (no override), L8 eta_h defaults to 0 (unset) ==="
    python3 -u run/substrate/adaptive/refined_neutron_decay.py \
        --R 20 --max-level 8 --region-r0 5.0 \
        --lam-target 1.0 --lam-scale 0.05 --lam-mode dynamic \
        --cross-scales 1e-3 --damp-max 0 \
        --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 \
        --n-steps 1 --report 1 --n-threads 8 \
        --save-snapshot-dir "$OUT_DIR/snapshots" \
        > "$OUT_DIR/l8_1step.log" 2>&1
    date
    echo "=== finished L8 first look ==="
    tail -40 "$OUT_DIR/l8_1step.log"

) 1> >(tee "$CWD"/.run.sh.out) 2> >(tee "$CWD"/.run.sh.err >&2)
