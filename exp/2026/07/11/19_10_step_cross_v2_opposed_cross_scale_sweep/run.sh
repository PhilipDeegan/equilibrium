#!/usr/bin/env bash
set -ex
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=$PWD
for cs in 0.05 0.1 0.3; do
  python3 -u tst/substrate/adaptive/probe_cross_v2_gates.py \
    --gates opposed --refs 2.14e-2 --cross-scale "$cs" --n-steps 4000 \
    --data-out /tmp/opposed_cs_${cs}.npz --no-plot
done
