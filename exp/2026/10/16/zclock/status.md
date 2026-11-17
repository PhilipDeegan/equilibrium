# 2026-10-16 zclock -- Z_N count with a neighbour-imbalance tick rule (valve test)

zclock.py: integer count n mod Nc (2^20) per cell; face offset o = wrapped difference; imbalance I = 2 sum_hex o +
sum_sq o; integer leapfrog n+ = 2n - n- + round(I / q), q 32 (CFL x 0.5). Exactly reversible. No floats in the state.

## Smooth limit
Plane wave, A 20000 counts: period ~80 ticks vs 78.4 from c^2 = 4 / q; energy kept to 1e-3.
A 200: energy wanders 20 %; A 60: energy grows x4 in 400 ticks. Rounding pumps grid-scale modes = hidden noise at
low amplitude (conflicts with "no noise, only context").

## Vortex ring, no current (P 72, R 8)
Completely frozen: defect set, radius, position unchanged over 1200 ticks. Solid-angle phase is harmonic -> I ~ 0 ->
nothing moves. Same as 10-12 Villain kappa 0: with a linear rule the integers are labels.

## Ring + box current k (winding k along x), slip = box-winding change on the chain through the ring
P 72 R 8:   k 1 frozen | k 2, 3 ring collapses, linked chain loses exactly 1 (clean slip) | k 4 half collapse |
            k 8, 12 lattice breakdown (defects everywhere, E x5). Opposite ring orientation: frozen at k 2, 4.
P 140 R 16: k 2, 3, 4 frozen | k 6 clean slip | k 8 partial.
Threshold k (same R / side, same structure-scale current) moves from 1-2 to 4-6 under x2 refinement: the barrier is
set by the cell (per-face wrap at the one-triangle core), not by the structure. Continuum limit: no slips at all.
Energy jumps at a slip (+0.7 to +1.8 %): the wrap makes the energy discontinuous.

## Verdict
Linear neighbour imbalance gives exact integer counts (the universal part) and a valve, but the valve is the floor:
threshold grows with refinement -> perfect conductor in the limit, no Ohm. Defects only move by the wrap; nothing at
structure scale sets the barrier. The tick rule needs a structure-scale dependence (e.g. on depletion / core state),
not just the linear imbalance. Also: a 1-scalar count gives 1 wave polarisation; Maxwell needs 2 -> E + M from one
rule needs more quantities per cell.
