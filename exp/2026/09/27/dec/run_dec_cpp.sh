#!/bin/bash
cd "$(dirname "$0")"
for h in 0.5 0.25 0.125; do
  n=$(python3 -c "print(f'{20/$h:g}')")
  date
  PYTHONPATH=/home/p/git/eq python3 -u dipole_dec.py --R 20 --h $h --cpp --threads 8 --op dec --plots 20 --snaps 2 \
    --out dipole_dec_cpp_R${n}_h${h}.npz > dipole_dec_cpp_R${n}_h${h}.log 2>&1
done
date
