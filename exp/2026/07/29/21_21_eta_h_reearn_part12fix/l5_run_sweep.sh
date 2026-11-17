#!/bin/bash
set -e
cd /home/p/git/eq
export PYTHONPATH=$PWD

for c in 1e-11 3e-11 1e-10 3e-10 7e-10 1e-9; do
  echo "$(date) starting eta5=$c"
  python3 -u tst/substrate/adaptive/refined_neutron_decay.py \
    --R 20 --max-level 5 --region-r0 5.0 \
    --scale-override 1.3606e-3 --scale-ref-level 8 \
    --lam-target 1.0 --lam-scale 0.05 --lam-mode frozen \
    --cross-scales 0 --damp-max 0 \
    --eta-h-per-level 0 0 1e-5 1e-7 3e-9 "$c" \
    --n-steps 150 --report 15 --n-threads 8 \
    > tmp/eta_sweep/l5_h_boundedness_reearn/eta_"$c".log 2>&1
  echo "$(date) finished eta5=$c"
done
echo "$(date) L5 150-step sweep complete"
