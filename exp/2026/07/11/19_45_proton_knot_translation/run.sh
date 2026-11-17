#!/usr/bin/env bash
set -ex
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=$PWD
python3 -u tst/substrate/probe_proton_translation.py \
  --R 20 --n-base 1000 --n-kick 6000 --n-coast 4000 --report 500 \
  --cross-scale 0.01 --force-scale 1e-2 \
  --data-out /tmp/proton_translate_baseline.npz --no-plot
python3 -u tst/substrate/probe_proton_translation.py \
  --R 20 --n-base 1000 --n-kick 6000 --n-coast 4000 --report 500 \
  --cross-scale 0.3 --force-scale 0.3 \
  --data-out /tmp/proton_translate_strong.npz --no-plot
