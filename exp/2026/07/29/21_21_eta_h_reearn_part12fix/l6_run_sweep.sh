#!/bin/bash
set -e
cd /home/p/git/eq
export PYTHONPATH=$PWD

echo "$(date) starting eta6=0 (baseline)"
python3 -u tst/substrate/adaptive/refined_neutron_decay.py \
  --R 20 --max-level 6 --region-r0 5.0 \
  --scale-override 1.3606e-3 --scale-ref-level 8 \
  --lam-target 1.0 --lam-scale 0.05 --lam-mode frozen \
  --cross-scales 0 --damp-max 0 \
  --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 0 \
  --n-steps 150 --report 15 --n-threads 8 \
  > tmp/eta_sweep/l6_h_boundedness_reearn/eta_0.log 2>&1
echo "$(date) finished eta6=0"

for c in 1e-13 3e-13 1e-12 3e-12 1e-11; do
  echo "$(date) starting eta6=$c"
  python3 -u tst/substrate/adaptive/refined_neutron_decay.py \
    --R 20 --max-level 6 --region-r0 5.0 \
    --scale-override 1.3606e-3 --scale-ref-level 8 \
    --lam-target 1.0 --lam-scale 0.05 --lam-mode frozen \
    --cross-scales 0 --damp-max 0 \
    --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 "$c" \
    --n-steps 150 --report 15 --n-threads 8 \
    > tmp/eta_sweep/l6_h_boundedness_reearn/eta_"$c".log 2>&1
  echo "$(date) finished eta6=$c"
done
echo "$(date) L6 150-step sweep complete"
