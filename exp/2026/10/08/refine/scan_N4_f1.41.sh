#!/bin/bash
# 10-08: uniform refinement rung f = sqrt2 of the 10-07 N 4 runs (Philip: N should converge under refinement).
# xi 3.9 -> 5.515, r0 30 / 40 -> 42.43 / 56.57, a 10 -> 14.14, P 220 -> 312, tail 20,40 -> 28.3,56.6, slosh_dt 38 -> 53.7,
# T 20k -> 40k (time ~ xi^2). Both at once, 4 threads each.
cd "$(dirname "$0")"
ROOT=$(cd ../../../../.. && pwd)
run() {
  PYTHONPATH=$ROOT python3 -u spring_box.py --N 4 --a 14.14 --r0 $1 --P 312 --m 0.8 --xi 5.515 --walk_xi --T 40000 \
    --n_plot 10 --recentre 500 --threads 4 --parity --probe_every 400 --project_plus --profile gp_profile_xi5.515.npz \
    --tail 28.3,56.6 --slosh_every 40 --slosh_e_every 400 --slosh_dt 53.7 --tag _f141 > N4_r$1_f1.41.log 2>&1
}
date
run 42.43 &
run 56.57 &
wait
date
