# Discrete boundary design — 2026-09-29

`sweep.py`, numpy step() (current weights, c = 2.309), interior Ri = 9.6, log `sweep.log`,
rows `sweep_rows.npy`.

## Waves: rate sponge, faces consistent
After each step: u *= exp(-g dt), phi *= exp(-g_face dt), g_face = mean of the two cells,
g = gmax t^p over width w. Gaussian u pulse (width 1.5); metric = mean interior energy fraction over
t = 6.1..14.4 (after the pulse leaves). Floor of this metric (reference domain, nothing returns):
1.0e-4 (dispersive tail still leaving early in the window).

| w (cells) | best gmax | p | reflection | production DampedLattice (damp 0.02) |
|---|---|---|---|---|
| 2.4 | 10 c/w | 2 | 5.6e-3 | 2.4e-2 |
| 4.8 | 10 c/w | 2 | 2.8e-3 | 3.2e-2 |
| 9.6 | 10 c/w | 3 | 8.3e-4 (p=2: 1.0e-3) | 7.2e-3 |
Design rule from the sweep: gmax ~ 10 c / w; reflection falls with width; 4-11x better than
production at the same width, dt-consistent and antisymmetric by construction.

## Statics: conflict with any phi-damping sponge
In step(), phi_t = -W grad u: a static u gradient makes phi grow linearly for ever, so a steady
potential needs phi to keep growing. Any sponge that damps phi caps it where the gradient reaches,
div phi != 0 there, and u piles up.
- Dirichlet shell only (u = 0 on the outermost shell, no sponge), fixed u = 1 source, 40k steps:
  log slope -1.085, fit to a(1/r - 1/b) rel rms resid 0.10, still changing (2.5e-2): nothing
  damps the transient, it rings between source and wall.
- Dirichlet + best sponge: log slope -0.02, u fills up (resid 0.43, still rising).
The discrete model has no true static field with a gradient: u can be steady only with phi
growing. The continuous AE model does (A, u static; E = -grad u - A_t). The per-cell sponge's
invisible sink (sponge_assess) is what let the point-source tests look static.
