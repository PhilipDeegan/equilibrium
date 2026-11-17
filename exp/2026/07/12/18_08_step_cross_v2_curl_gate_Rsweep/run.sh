#!/usr/bin/env bash
set -ex
export PYTHONPATH=$PWD

# R=30 run (completed)
python3 -Ou tst/substrate/adaptive/probe_cross_v2_gates.py \
    --R 30 --sph-frac 0.17 --gates v1,opposed,curl \
    --data-out "$PWD/gate_sweep_R30.npz" \
    --plot-out "$PWD/gate_sweep_R30.png" \
    2>&1 | tee output_R30.log

# R=80 run (killed — too slow, N=2,784,735 cells; only reached electron/v1 step 3000)
# python3 -Ou tst/substrate/adaptive/probe_cross_v2_gates.py \
#     --R 80 --sph-frac 0.06 --gates v1,opposed,curl \
#     --data-out "$PWD/gate_sweep_R80.npz" \
#     --plot-out "$PWD/gate_sweep_R80.png" \
#     2>&1 | tee output_R80.log
