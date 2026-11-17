# 10-08 refine: uniform refinement rung f = sqrt2 of the N 4 coils (Philip: N should converge under refinement; AMR walk
# not built yet -- CppWalk is level 0 only)

Scaling (all lengths x sqrt2, ticks x 2): xi 3.9 -> 5.515 (gp_profile_xi5.515.npz from 07/core/gp_vortex.py), r0 30 / 40
-> 42.43 / 56.57, a 10 -> 14.14, P 220 -> 312, tail 28.3,56.6, slosh_dt 53.7, T 40k (= 20k at f 1). scan_N4_f1.41.sh,
4 threads each, both at once. Compare with 07/slosh n4_long_10-07.log (f 1): r0 30 -> tight 4-turn ring (rad 10.4 -> 4.0-4.5,
Et settled ~539, winding -4 kept); r0 40 holds wide (rad ~8.5). At f sqrt2 the matching numbers are rad x sqrt2.
Question: same outcomes in xi units (converged at this rung) or not (resolution-dependent -> next rung / AMR walk).

## Result (compare_f.py -> compare_f_10-08.log; both 40k done 15:02, ~1 h 55 each at 4 threads)
f sqrt2 retraces f 1 in xi units and scaled time (t / f^2) to ~1-2 % at every 2k step to the 20k-equivalent:
  below floor (r0 7.9 xi): rad / xi 2.67 -> 1.16 vs 2.66 -> 1.18; rc / xi 7.87 -> 7.44 vs 7.87 -> 7.42; k4 helix
  modulation 2.50 -> 0.83 vs 2.50 -> 0.86; Et / f 632 -> 543 vs 634 -> 548 (last row 532 = a one-point bin at t 40k).
  above floor (r0 10.4 xi): rad / xi 2.66 -> 2.00 vs 2.66 -> 1.99; Et / f 816 -> 797 vs 818 -> 798; holds wide.
  Windings -4 throughout in both (f 1's -4/0 are offset probe circles leaving the thinned tube, as 10-07).
Reading: the walk coil is already converged at xi 3.9 cells; xi is its only length, so uniform refinement just
approaches its own scale-free (GP-like) limit, and that limit keeps N 4 (tight below the floor, wide above). Refining
the walk does not shrink the persistent-N set; an AMR walk would refine an already-converged solution. Selection of N
(Philip's expectation) needs a second length the walk lacks (substrate / L_MIN, lam structure, neighbour density).
