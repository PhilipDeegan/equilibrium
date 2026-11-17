# 2026-10-16 relay -- transverse photon sector on the BCC complex

rdisp.py: DEC Maxwell, E = edge integrals on the 7 cell-to-cell hops per cell, B on the 12 Delaunay triangles per
cell, stars continuous.dec_star('dec') (r 2, s2 3/4). One coupling K (triangle loop sum of hops) used both ways:
dE/dt = K^T B, dB/dt = -K E -> E and M from the same incidence rule.

## Field form (Bloch, exact)
7 branches per cell: 2 transverse, 1 longitudinal static (omega 0 exactly: loop sum of a gradient = 0), 4 gapped flat
at omega 3 sqrt3 = 5.196 (cell-internal, no propagation).
Transverse omega / q, directions [100] [111] [110] [123]:
  q 0.05 1.0000 all | q 0.3 0.9992 all | q 0.6 0.9967 all | q 1.0 0.9908 / 0.9908 / 0.9906 / 0.9907 |
  q 1.5 0.9795 / 0.9792 / 0.9785 / 0.9787-0.9791.
Isotropic to 1e-4 at q 1, two polarisations degenerate (birefringence < 4e-4 at q 1.5). Maxwell-grade.
Compare the one-cell junctions (eject / tnode): longitudinal wave or no transverse wave, ~6 % anisotropy.

## Relay form (one exact rotation per edge-triangle incidence, Strang over the 36 types)
theta 0.05: photon 0.999-1.002 + a spurious slow pair (0.017-0.024); theta 0.3: spurious 0.68-0.86 below the photon;
theta >= 0.8 and theta pi/2 (full hand-off): no long-wavelength photon branch at all. A literal full-packet relay
per incidence destroys Maxwell (cf. 1D magic time step: pure shift is exact only in 1D).

## Reading
"Cells don't keep a perturbation" holds for the transverse (loop) part: a hop perturbation with circulation is
converted to its triangles and back -> must travel. The longitudinal (gradient) part is held, static, and only where
div E is nonzero (exactly conserved: vacuum cannot make it) -> Coulomb is the held branch. The 4 gapped modes are
fixed-frequency cell-internal cycling (flat, omega 5.196) -- at L 1 possibly real cell activity; flagged only.
Open: an accept-eject (TLM-style) form that reproduces this dispersion; leapfrog (E half tick, B half tick) is the
working update.
