# 10-08 tick: step() with phi derived from a per-cell periodic quantity (tick phase)

tick.py: theta per cell (mod 2 pi), phi_ik = W_k wrap(theta_j - theta_i), u -= dt sum W phi, theta -= dt (u + kappa q),
q = sum W^2 wrap^2 / (2 sum W^2) (a cell's clock shifted by how far out of step its junctions are; optional).
1. Smooth limit (--test pulse, P 60, Gaussian u bump 1e-3): == C++ step() to round-off (u 1e-10 relative, phi 2e-14).
2. Quantization (--test ring, P 120, ring R0 15 seeded as theta = Omega / 2 over 27 images): 696 defect triangles
   (elementary loops with a whole-number winding, |n| = 1), conserved exactly to t 30; no slips (ring_k0_10-08.log).
   But the ring is FROZEN: rho 14.78, z -0.002 throughout -- lattice-pinned. With no amplitude, a defect can only move by
   a slip (an offset through +-pi); its line tension (~1/R) never forces one.
3. Clock-shift relation (ring_k*_10-08.log): kappa +-1: one-off shift 0.04-0.05 in z, then frozen again, 696 triangles.
   kappa 5: moves (z +0.87 over t 30) and shrinks (rho 14.85 -> 13.96, 696 -> 672 triangles) but E triples (7.5k -> 21k):
   q pumps energy (not conservative) -> not trustworthy as transport.
Reading: a periodic per-cell quantity gives exact whole-number circulation for free, and step()'s smooth physics
unchanged; but a pure phase has no core, so defects are pinned. Mobility needs the junction at a defect to soften: the
cells there are maximally out of step (offsets ~ pi). DST-native candidate: the junction coupling (sigma at the face)
weakens with depletion, and depletion rises where neighbouring cells are far out of step -> a soft, depleted core that
can move. Open; not built.
