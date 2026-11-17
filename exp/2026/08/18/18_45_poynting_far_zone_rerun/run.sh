#!/usr/bin/env bash
set -ex
cd "$(git rev-parse --show-toplevel)"
PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_poynting_dipole_compare.py \
  --R 100 --hops 4 --m0 1e-4 --cross-scale 0.01 --n-steps 8000 --threads 8 \
  --r-min-fit 40 --r-max-fit 80 \
  --plot-out exp/2026/08/18/18_45_poynting_far_zone_rerun/poynting_dipole_compare_R100_farzone.png \
  > exp/2026/08/18/18_45_poynting_far_zone_rerun/run_R100_farzone.log 2>&1
