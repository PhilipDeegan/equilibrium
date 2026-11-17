#!/usr/bin/env bash
# 10-19 recv R 9: lone ring vs box size (images = receivers), and two side-by-side rings A (R 9) + B (R 5.4); 4 x 1 thread
cd /home/p/git/eq
export PYTHONPATH=/home/p/git/eq
job() {
  n=$1; shift
  python3 -u exp/2026/10/18/zoo/zoo.py --R 9 --T 30 "$@" > exp/2026/10/19/recv/r9/$n.txt 2>&1
}
export -f job
printf '%s\n' \
  "lone_P144 --P 144" \
  "lone_P120 --P 120" \
  "side_P96 --P 96 --struct side --R2 5.4" \
  "sideA_P96 --P 96 --struct side_a --R2 5.4" \
  "sideB_P96 --P 96 --struct side_b --R2 5.4" \
  "lone_P96 --P 96" \
  "lone_P72 --P 72" | xargs -P 4 -I{} bash -c 'job {}'
