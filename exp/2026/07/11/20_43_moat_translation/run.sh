#!/usr/bin/env bash
set -ex
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=$PWD
for s in 0.01 0.03 0.1 0.3; do
  python3 -u tst/substrate/probe_moat_translation.py \
    --R 20 --n-base 1000 --n-kick 6000 --n-coast 4000 --report 1000 \
    --cross-scale "$s" --force-scale "$s" \
    --data-out /tmp/moat_s${s}.npz --no-plot
done
python3 -u tst/substrate/probe_moat_translation.py \
  --R 20 --n-base 1000 --n-kick 6000 --n-coast 4000 --report 1000 \
  --cross-scale 0.3 --force-scale 0.3 --kick-axis x \
  --data-out /tmp/moat_strong_x.npz --no-plot
