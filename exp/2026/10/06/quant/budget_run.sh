#!/bin/bash
# budget_run.sh N a r0 -> deficit budget + census at t 0..8000 of one scan run
N=$1; a=$2; r0=$3; S=$(python3 -c "print(220/3**0.5)")
D=/bk/dst/2026/10/06/quant/springbox_N${N}_S1_q1_a${a}_r${r0}_xi3.9_m0.8_P220_z0_wxi_rc_par
PYTHONPATH=/home/p/git/eq python3 -u deficit_budget.py --r0 $r0 --a $a --box $S $(for t in 00000 02000 04000 06000 08000; do echo --state $D/state_t$t.npz; done) > budget_N${N}_a${a}_r${r0}.log 2>&1
