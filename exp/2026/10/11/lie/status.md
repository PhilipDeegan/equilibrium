# 10-11 lie: phi self-transport as an exchange-matrix commutator (no vectors)

lie.py. a_ij = phi_ik / W_k (circulation along the hop), v_ij = W_k phi_ik (flux = dH/da, H = step()'s sum phi^2 / 2),
both antisymmetric cell-to-cell matrices on the faces. da/dt = kappa P_S [M, v], M = a extended to 2-hop pairs by the mean
of a over the 2-hop paths (50 displacements, 1/2/4 paths). The 1-hop parts of [a, v] cancel exactly: the term is driven
only by path dependence of the 2-hop sums = loop circulation. No cell vectors, no cross product, no positions (nb only).

diag_10-11.log (P 120, ring R0 12 wt 3 a0 1): antisymmetric to 5e-16; sum phi . term = 7e-15 (energy-neutral exactly).
Contains a large gradient part (|div-free part| / |term| 0.225; carry's A x B per face: 0.231). Div-free parts correlate
+0.909 with carry's A x B (scale 7.58).
ring_10-11.log (kappa +-0.132 = 1 / 7.58, midpoint x3 centred, t 10): z +-0.0328 / time, exactly sign-reversed, steady;
shape kept (rho 13.03 -> 13.06); leapfrog invariant 1310.41698-1310.41751 (4e-7). carry (A x B, kappa 1, a0 1): 0.031.
The gradient part feeds u (max |u| ~0.07, sound) -- carry dropped it by hand; here it is part of the relation.
Reading: the 10-08 transport survives removing the vector import: a commutator of the face quantity with itself (via
2-hop path sums) reproduces it, energy-exact by construction.
Open: kappa is still picked (here matched to carry); Euler-Poincare would fix it to 1 in consistent units -- check
whether the lattice units give the 7.58 or not. Ring-size scaling (R0 18), refinement, then the tick-phase combination
(does the commutator move a 2 pi defect?).

## Ring size and refinement (10-11 afternoon)
ring_R18_10-11.log: R0 18 (wt 3, f 1): 0.0218 / time vs 0.0328 at R0 12, ratio 0.665 (carry 0.68; thin-ring 0.74).
lie.py now prints a t 0 instantaneous speed (one term evaluation, z centroid shift per unit time) and light progress
lines (running speed vs --expect, E_lf drift, s / tick, ETA). Small-box check: t 0 estimate 0.0502 vs running 0.0503.
rung_t0_10-11.log (all lengths x f, a0 x f -- a0 is the streamfunction amplitude, flow = a0 / wt; P 120 f):
  f 1 0.03435   f sqrt2 0.03574   f 2 0.03646   f 2 sqrt2 0.03693   (differences 139 / 72 / 47 e-5, ratios 1.93 / 1.53)
  converging, ~2nd order on the first steps, slower on the last; limit ~0.0375-0.039. Not a stencil artefact.
  f 1 long run 0.0328 vs t 0 0.03435 (long run ~4.5 % slower: u builds up). ring_fsqrt2_10-11.log (long, a0 sqrt2,
  ends ~18:35) checks the t 0 / long ratio at f sqrt2 (expect ~0.0341). The a0-fixed f sqrt2 long run (0.0248 at a0 1,
  -> 0.0351 by linearity) agrees.
  Stopped at t 2.84 (no longer needed): z 0.099 -> 0.0349 / time vs 0.0351 predicted by linearity.
