#!/usr/bin/env bash
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..
set -e
export PYTHONPATH=$PWD

BASE="--R 30 --r-sun 3 --chi-sun 0.7 --chi-ism 0.01
      --planets 5:0.05,10:0.02,15:0.01,20:0.005
      --alpha 0.005 --n-steps 50000 --report 10000 --threads 6"

run() {
    label=$1; shift
    echo ""
    echo "========================================"
    echo "TEST: $label"
    echo "========================================"
    python3 -Ou run/substrate/solar_sandbox.py $BASE "$@" --save-prefix /tmp/brk_${label} 2>&1 \
        | grep -E "step|early|error|Error|nan|NaN|solar|planet|v_circ|N " | head -20
}

# 1. Baseline (should pass)
run "baseline" --v-circ 0.05

# 2. No ISM — pure vacuum outside sun
run "no_ism" --chi-ism 0.0 --v-circ 0.05

# 3. Wrong phi — flat instead of Keplerian
run "flat_phi" --v-circ 0.05 --flat-phi

# 4. Way too fast — v_circ near wave speed
run "too_fast" --v-circ 0.5

# 5. Near-saturated sun — chi_sun close to 1
run "hi_chi_sun" --chi-sun 0.95 --v-circ 0.05

# 6. Planet chi comparable to sun — wrong mass ratio
run "heavy_planets" --planets "5:0.4,10:0.3,15:0.2,20:0.1" --v-circ 0.05

# 7. Tiny domain — boundary effects
run "tiny_domain" --R 15 --planets "3:0.05,5:0.02,8:0.01" --r-sun 2 --v-circ 0.05

# 8. Very low alpha — essentially frozen chi
run "zero_alpha" --alpha 0.0 --v-circ 0.05

# 9. High alpha — past stability boundary
run "hi_alpha" --alpha 0.5 --v-circ 0.05

# 10. Two solar systems — offset chi blob at (20,0,0) on top of existing planet
run "two_systems" --planets "5:0.05,10:0.02,15:0.5,20:0.5" --v-circ 0.05

echo ""
echo "=== break test done ==="
