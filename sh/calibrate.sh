#!/usr/bin/env bash
# Calibration sweep: find (u_central, alpha) giving a stable chi terrain.
#
# ── Scale changes vs full orbit_sandbox ──────────────────────────────────
#   Parameter     Full      Calib   Factor   Transfer rule
#   R             30        15      /2       geometry only — scale back ×2
#   R_central      3         1.5    /2       geometry only — scale back ×2
#   max-level      2         1      -1 lvl   resolution only — restore for prod
#   n-steps    30 000     5 000     /6       time only — restore for prod
#   chi_central    0.5       0.5    —        UNCHANGED — transfers directly
#   u_central      *         *      —        UNCHANGED — transfers directly
#   alpha          *         *      —        UNCHANGED — transfers directly
#   v_rotation     *         *      —        UNCHANGED — transfers directly
# ─────────────────────────────────────────────────────────────────────────
# Only alpha and u_central are swept here.  The calibrated values plug
# straight into run.sh without any scaling.

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..
set -e
export PYTHONPATH=$PWD

BASE="--no-planet --lam-background 0 --max-level 1 --n-steps 5000 --report 5000"
BASE="$BASE --R 15 --R-central 1.5 --threads 4 --bc-target auto"
RC=1.5   # must match --R-central above

mkdir -p calib

printf "%-8s %-8s   %s\n" "u_central" "alpha" "Sun_core  inner     mid       outer     verdict"
printf "%-8s %-8s   %s\n" "---------" "-----" "--------  ------    ---       -----     -------"

for U in 0.1 0.5 1.0 2.0 5.0; do
    for A in 0.001 0.01 0.05 0.1 0.5; do
        PFX="calib/u${U}_a${A}"
        python3 -Ou run/substrate/orbit_sandbox.py $BASE \
            --u-central "$U" --alpha "$A" --save-prefix "$PFX" \
            > /dev/null 2>&1

        LINE=$(python3 -Ou vis/substrate/orbit_compare.py \
            --R-central "$RC" --short \
            "${PFX}_init.h5" "${PFX}_final.h5" 2>/dev/null)

        # verdict: STABLE if outer mean Δchi < 0.05, PARTIAL if < 0.2, SATURATED otherwise
        OUTER=$(echo "$LINE" | grep -oP 'outer=\K[+-][0-9.]+' || echo "1.0")
        if awk "BEGIN{exit !($OUTER < 0.05)}"; then
            V="STABLE"
        elif awk "BEGIN{exit !($OUTER < 0.2)}"; then
            V="partial"
        else
            V="SATURATED"
        fi

        printf "%-8s %-8s   %s  [%s]\n" "$U" "$A" "$LINE" "$V"
    done
done

echo ""
echo "done — checkpoint files in calib/"
