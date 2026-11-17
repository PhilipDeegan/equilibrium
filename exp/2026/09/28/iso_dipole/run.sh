#!/bin/bash
# iso-weight dipole benchmark vs the 09-27 discrete-weight 'dec' runs (source work 0.425 / 0.415, ref 0.4114)
cd "$(dirname "$0")"
for h in 0.5 0.25; do
  n=$(python3 -c "print(f'{20/$h:g}')")
  date
  PYTHONPATH=/home/deegan/git/eq python3 -u dipole.py --R 20 --h $h --cpp --threads 8 --op dec --weights iso --plots 20 --snaps 0 \
    --out dipole_iso_dec_R${n}_h${h}.npz > dipole_iso_dec_R${n}_h${h}.log 2>&1
done
date
