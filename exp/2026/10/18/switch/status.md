# 2026-10-18 switch -- imposed core whose content switches electric <-> magnetic (Philip), moving at c

sw.py (from 10-16 nr.py): hop / triangle DEC, P 140, dt 0.25, T 40, sigma 3, om 0.5, v 1, ramp 5. p(t) = wP cos(om t) x
on hops; m(t) = wM cos(om t - psi) mdir as triangle magnetization (star2 M.A_t; triangle area vectors checked against
a uniform-B loop sum). psi 90 / 270 = switching (energy alternates E <-> M); psi 0 / 180 = in phase. Leak = growth of
energy behind the slab per unit distance (now read as a redshift-rate candidate, not a failure: Philip 10-18).

| core | dE_behind / distance | E_slab | ratio | on-axis wake behind (|a.x| at -16..-32) |
| p only | 6.33 | 47.9 | 0.13 | ~0.1-0.24 |
| m only (y) | 6.26 | 31.8 | 0.20 | -- |
| m || y, psi 90 (switching) | 12.36 | 75.5 | 0.16 | ~0.1-0.31 |
| m || y, psi 270 (switching) | 12.81 | 83.7 | 0.15 | |
| m || y, psi 0 (in phase, Huygens) | 12.22 | 128.5 | 0.095 | ~0.002-0.07 (10x smaller) |
| m || y, psi 180 | 13.36 | 71.6 | 0.19 | |
| m || x, psi 0 / 180 | 12.79 | 100.0 | 0.13 | |
| m || x, psi 90 / 270 | 12.58 | 79.6 | 0.16 | |

- wP = wM = 1 is already matched (equal leak alone).
- Switching cores leak exactly the sum of their parts: the E-M cross term averages out in quadrature (linear
  superposition). m || p: the two parts radiate orthogonal polarizations, so the phase does nothing to the leak.
- In phase with m perpendicular to p (Huygens / Kerker) kills the on-axis backward wake (10x) but the total leak drops
  only 3 %: almost all leak goes sideways (theta ~ 90 deg, falling behind a c-mover), where Kerker does nothing.
- The switching wake on axis alternates with period ~24 cells = Doppler-stretched 2 (2 pi / om) = 25 (backward).
- Reading: E <-> M quadrature is the standing-wave / cavity phase relation (travelling fields are in phase). A
  switching core is the signature of a confined field, but the linear update has no walls: in it the switching core is
  just two independent radiators. What would supply the cavity (the structure itself, a context, a nonlinearity) is
  open.
