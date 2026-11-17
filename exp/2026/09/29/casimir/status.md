# Casimir on the Kelvin lattice — 2026-09-29

## Finding
The 1/d^3 Casimir term on the lattice equals the scalar continuum value -pi^2 c / (1440 d^3)
(Dirichlet plates), with c = the lattice wave speed, to 1e-4 — for both W = sqrt3:1 (c = 2a, ratio
0.9999) and the old 3:1 (c = sqrt10 a, ratio 1.0000). It is universal: only the long-wavelength
(Kz -> 0) end sets it, so it carries no lattice-scale information at leading order.

## Method (`mode_sum.py`)
BCC stacked along z: layer spacing h = a/2, each layer a square lattice (spacing a). Per transverse
K, a layer chain: hex coupling to j+-1 (4 W_HEX^2 cos(hKx) cos(hKy)), square to j+-2 (W_SQ^2),
on-site D0. Plates = two fixed u = 0 layers each side (square links hop one layer, cf. the
double-slit wall). Exact eigenvalues, E(N) = 1/2 sum omega per area, d = N h.
Fit E = alpha N + beta + gamma/N^3 + g4/N^4 + g5/N^5 (g4: effective plate position):

| weights | N range | powers | gamma / theory |
|---|---|---|---|
| sqrt3:1 | 8-40 | 3,5 | 0.9754 |
| sqrt3:1 | 8-40 | 3,4,5 | 0.9998 |
| sqrt3:1 | 16-64 | 3,4,5 | 0.9999 |
| 3:1 | 16-64 | 3,4,5 | 1.0000 |

Transverse grid nxy 128 = 256 identically (Dirichlet gaps every mode -> exponential convergence).
No 1/d term exists in the exact sum.

## What was wrong in `equilibrium/substrate/casimir.py`
- Euler-Maclaurin derivatives taken only at the zone edge Kz = pi/a; the Casimir term comes from the
  Kz = 0 end (non-analytic |Kz|^3 part of the transverse average). term1 (1/d) and term3 are
  zone-edge quantities; their 1/d, 1/d^3 "scaling" tests are tautological (the formulas are those
  powers by construction).
- Kz range [0, pi/a] with d = N a: the z layer spacing is h = a/2 (range [0, pi/h]).
- Reference E_continuum used c = 1 (docstring said sqrt10). ratio3 0.72 -> 0.26 was edge-derivative
  arithmetic, not a BZ mode-density effect. doc/dst.md's "0.72x" paragraph inherits this.
- `_dispersion` itself is correct (casimir_polder.py uses only that).

## Consequence for the epsilon-floor calibration
Casimir at leading order cannot calibrate the lattice: the 1/d^3 coefficient is c-only. Lattice
information sits in g4/g5 (d ~ a corrections), beta (surface) and alpha (bulk vacuum energy
density). Real small-d deviations from ideal Casimir are dominated by the plate material (finite
conductivity), so a lattice-scale bound from them would be weak.
