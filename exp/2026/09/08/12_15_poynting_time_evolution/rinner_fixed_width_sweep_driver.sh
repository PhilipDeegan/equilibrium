#!/bin/bash
set -e
cd /home/p/git/eq
OUT=exp/2026/09/08/12_15_poynting_time_evolution
echo "=== starting rinner_fixed_width_sf0.3 $(date) ===" | tee -a $OUT/rinner_fixed_width_sweep_driver.log
python3 -u $OUT/probe_sponge_quality_v3.py --R 78.928571 --mode octagon --hops 8 --threads 8 \
  --sponge-fraction 0.3 --variants split,pml --damp-values 0.003 --gamma-values 0.5,1,2 \
  --out-prefix $OUT/sponge_v3_rinnerfix_sf0.3 2>&1 | tee -a $OUT/sponge_v3_rinnerfix_sf0.3.log
echo "=== starting rinner_fixed_width_sf0.4 $(date) ===" | tee -a $OUT/rinner_fixed_width_sweep_driver.log
python3 -u $OUT/probe_sponge_quality_v3.py --R 92.083333 --mode octagon --hops 8 --threads 8 \
  --sponge-fraction 0.4 --variants split,pml --damp-values 0.003 --gamma-values 0.5,1,2 \
  --out-prefix $OUT/sponge_v3_rinnerfix_sf0.4 2>&1 | tee -a $OUT/sponge_v3_rinnerfix_sf0.4.log
echo "=== finished rinner_fixed_width_sweep $(date) ===" | tee -a $OUT/rinner_fixed_width_sweep_driver.log
