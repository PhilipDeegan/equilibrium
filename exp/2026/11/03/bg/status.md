# 2026-11-03 does a disturbance feel a settled background? (Philip: photons ride on a magnetically settled background)

bg.py: continuous sphere R 30 (N 147k), h 1. Background phi = div-free sample of A = B/2 z x r (uniform circulation about
z; phi ~ A). Small Gaussian u pulse (amp 1e-3, s 2) at the origin. Response = run(bg + pulse) - run(bg) vs the bare
pulse on no background. T 12 (pulse at r 12 ~t 7, before the rim). Probes r 12 on +x, +y, -x, +z.
--settle: u_bg = kappa psi, psi = potential of the gradient part of the background's own Cartan transport (its
centrifugal push balanced by u pressure): drift 8.7 % -> 2.3 % (B 0.05), 21 % -> 4.8 % (B 0.1). Results equal settled /
unsettled to ~1e-3 -> the effect is pulse-background coupling, not background drift.

  rule     B      ||resp-bare|| u / phi   circulation share   peak u, perp (x,y) / along (z) vs bare
  step     0.05   5e-12 / 2e-11           3e-22               identical (control: linear, superposes)
  cartan   0.05   3.6e-2 / 0.106          9.4e-3              +2.0 % / -1.2 %
  cartan   0.1    0.150 / 0.250           3.2e-2              +8.1 % / -4.8 %
  cartan  -0.1    same as +0.1 for u (deviation differs 6 %); phi deviation has odd and even parts
Reading: in step() a background has no effect (exact). With Cartan transport a settled circulation makes the scalar
pulse anisotropic: stronger across the background axis, weaker along it, ~B^2 and blind to the sense of B (pressure-
like, two-fold, the symmetry polarisation needs); and part of the pulse is converted into circulation, first order in
B and flipping with it (handed). Arrival times unchanged (<0.3 %). Equatorial probes identical (axisymmetry).
Not yet: whether the converted part propagates (and with what orientation), lattice orientation of the background
(z only), kappa dependence, refinement.
