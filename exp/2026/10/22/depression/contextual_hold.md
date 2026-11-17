# 2026-10-22 contextual holding (paper). Philip: the substrate is one connected whole, never disconnected; the
# PATTERN's topology on it is contextual; bound structures are road blocks (compressed cells); contraction is slaved to
# local energy / particle presence, maybe outside the energy domain.

## Where today's rule imposes topology
Cartan: phi_t = -i_v d(alpha), v = dH/d(alpha). Any equation of the form -i_v d(alpha) - d(stuff) keeps every loop's
circulation along v-carried loops (Kelvin), for ANY v: rescaling v (csig, vcap, store) moves the loops differently but
keeps their circulation. So no rate / medium change on v can let the pattern rearrange -- that is why csig / vcap /
store only steered or reshaped, and why the spheromak could only unwind at the cell scale (into the 4 non-vector face
DOF, 50 % at t 450).

## The relation: what carries the pattern is read in context
Replace the vorticity in the transport by a context-blended one:
  phi_t = -i_v (kappa d(alpha) + (1 - kappa) <d(alpha)>)
  <d(alpha)> = the triangle loop sums averaged over neighbouring triangles (the pattern as read over several cells)
  kappa in [0, 1] per triangle, from the contraction of its cells, contraction slaved to local energy (no energy of
  its own), kappa = 1 where the pattern is held.
Discrete form: in cartan_transport each triangle's om becomes om_eff = kappa om + (1 - kappa) om_avg. Every triangle
still contributes om_eff (b2 - b3), om_eff (b3 - b1), -om_eff (b1 - b2), antisymmetric in b = dH/da, so
sum dH/dphi . out = 0 exactly -- the same proof that covers csig and vcap (any scalar per triangle). So:
- energy exact, reversible, no dissipation, no added rate;
- kappa = 1 everywhere == today's rule exactly (Kelvin where held);
- kappa < 1: the pattern is carried by its surroundings' reading of it, not by itself -> loop circulation can change
  smoothly over the averaging width (structure scale, not one cell) = the pattern rearranges without anything being cut;
- irrotational regions (om = 0, om_avg = 0) are untouched: the vacuum and plain waves are unaffected;
- the output -i_v(...) is not div-free in general -> feeds u through step() -> released twist can leave as waves
  (offload), not only pile up in place;
- transport by an averaged field is also what keeps a cascade from running to the cell scale.

## Picks (to be derived later)
- averaging stencil: the 12 triangles sharing an edge / cell with the triangle, uniform weights (one cell of reach).
- kappa(e): smooth monotone in the local energy of the triangle's cells, e.g. kappa = e^2 / (e^2 + e_h^2), e_h the
  holding scale. No clamp.

## Open (Philip)
1. Sign against L: L = h / <ell> is "how much averaging over discrete cells". Compressed cells -> L up -> MORE
   averaging there, which would loosen the pattern exactly where the road block should hold it. Either holding comes
   from coherence (many tightly coupled discrete cells read the same pattern, so their average IS the pattern), or the
   averaging here is a different thing from L. Which?
2. Is kappa from contraction (slaved to energy) enough, or does holding need the occupancy of the junctions too?

## Tests
T0 unit: kappa = 1 == step_cartan bit-identical; energy orthogonality per triangle 1e-13 for random kappa; om = 0 ->
   output 0.
T1 antiparallel straight pair, uniform kappa 0.5 vs 1: smooth rearrangement at structure scale, converged a vs 2a,
   where the energy goes (u waves).
T2 spheromak a 9 / 20 with kappa(e): held core vs relaxing edge; compare the step() + Cartan baseline (unwinds into
   face DOF at ~3.4 turnovers): does it decay smoothly instead, keep helicity longer than energy (D2), offload (D1)?
T3 opacity (zoo --pulse) once T2 shows holding.
