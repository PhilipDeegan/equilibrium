#!/usr/bin/env bash
export PYTHONPATH=/home/p/git/eq
B=/bk/dst/2026/10/16/geom
mkdir -p $B
run() { R=$1; P=$2; T=$3; sm=$4; g=$5; w=$6; f=$7; shift 7
  echo "$(date) start R$R $g wind $w"
  python3 -u geom.py --P $P --R0 $R --T $T --every 1 --smooth $sm --off 0.13 0.07 0.11 --kappa 1 --threads 8 --cpp --midpoint 2 \
    --geom $g --wind $w --wsw $(python3 -c "print(3*$f)") --rshell $(python3 -c "print(6*$f)") --wshell $(python3 -c "print(1.5*$f)") \
    --seed $B/seed_P${P}_R${R}_smooth${sm}.npz "$@" > R${R}_${g}_w${w}.log 2>&1
}
for g in plain twist tokamak hopf; do sm=1; [ $g = hopf ] && sm=0; run 9 72 20 $sm $g 1 0.45; done
for w in 0.5 2; do run 9 72 20 1 twist $w 0.45; run 9 72 20 1 tokamak $w 0.45; done
for g in plain twist tokamak hopf; do sm=4.938; [ $g = hopf ] && sm=0; run 20 140 30 $sm $g 1 1; done
echo "$(date) done"
