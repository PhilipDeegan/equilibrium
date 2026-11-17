#!/usr/bin/env bash
set -euo pipefail
cd /home/p/git/eq
export PYTHONPATH=$PWD

for eta5 in 3.3e-9 6.6e-9; do
  date
  python3 -u exp/2026/08/09/13_42_eta_h_l5_fine_bisect/sweep_eta_h_l5_resolved.py \
    --eta-h-l5 "$eta5" --n-steps 3 --n-threads 4 --tag "eta5_${eta5}" \
    2>&1 | tee "exp/2026/08/09/13_42_eta_h_l5_fine_bisect/eta5_${eta5}.log"
done
