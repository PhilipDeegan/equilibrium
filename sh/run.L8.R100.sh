#!/usr/bin/env bash

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..

set -ex

(
    export PYTHONPATH=$PWD

    # Real (Bohr/neutron-scale calibrated, no --scale-override) L8 refined
    # decay run -- R=100/max-level=8 defaults from refined_neutron_decay.py's
    # own real-world calibration (module docstring). 100 coarse timesteps,
    # with the confirmed-durable L8 mitigation from
    # doc/mems/project_odd27_amr_junction_leak.md Part 10/11: isolated
    # level-8-ONLY per-level hyper-resistivity (eta_h=1e-15 at level 8 only,
    # all other levels zero -- ran to full 60-step completion with no
    # divergence at the cheaper R=20 matched-footprint harness; NOT yet
    # re-verified at this real R=100/L8 scale, so treat this run itself as
    # that verification). Saves the final field state to --save-snapshot-dir
    # so it can be restarted/inspected later via --restart-from.
    #
    # RUNTIME IS UNKNOWN AND LIKELY LONG: a 2-coarse-step timing probe at
    # this exact R=100/L8 config did not complete within 9m50s (killed).
    # Per earlier memory (project_odd27_amr_junction_leak), R=100/L8 alone
    # was already known to take >8min for a single coarse step at ~5.58M
    # leaves -- each coarse step subcycles 3^8=6561 times at the deepest
    # level, so 100 coarse steps could plausibly take many hours. Runs in
    # the background (this script backgrounds itself below); check
    # tmp/refined_neutron_decay_L8/run.log for progress.
    OUT_DIR="$PWD/tmp/refined_neutron_decay_L8"
    mkdir -p "$OUT_DIR"

    date
    python3 -u run/substrate/adaptive/refined_neutron_decay.py \
        --n-steps 1 --report 10 --n-threads 8 \
        --eta-h-per-level 0 0 0 0 0 0 0 0 1e-15 \
        --save-snapshot-dir "$OUT_DIR" --restart-from "$OUT_DIR" \
        --out "$OUT_DIR/plot.png" \
        > "$OUT_DIR/run.log" 2>&1
    date
    echo "=== final summary ==="
    tail -20 "$OUT_DIR/run.log"


) 1> >(tee $CWD/.run.sh.out) 2> >(tee $CWD/.run.sh.err >&2)
