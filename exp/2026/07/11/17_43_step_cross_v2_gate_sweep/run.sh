#!/usr/bin/env bash
set -ex
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=$PWD
python3 -u tst/substrate/adaptive/probe_cross_v2_gates.py \
  --data-out /tmp/cross_v2_gates.npz \
  --plot-out /tmp/cross_v2_gates.png
