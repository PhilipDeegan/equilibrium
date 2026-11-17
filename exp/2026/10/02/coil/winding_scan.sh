#!/bin/bash
cd /home/p/git/eq/exp/2026/10/02/coil
N=$1; th=${2:-4}; r0=$((20+5*N)); R=$((r0+65))
PYTHONPATH=/home/p/git/eq python3 -u spring_cpp.py --N $N --a 20 --r0 $r0 --R $R --m 0.8 --xi 3.9 --walk_xi --T 13000 --n_plot 13 --recentre 500 --threads $th --tag scan > springcpp_scan_N${N}.log 2>&1
