# Hydrogen as two populations: proton and electron on the Kelvin lattice

Question: with the proton as a population too (not a fixed centre), do two populations that each see only the
other's density give hydrogen? Or is the electron's correlation with the proton (it follows the proton) essential?

## References (`hydrogen_pair.py --no-swarm`)

Same lattice and units as `../hydrogen_swarm` (m_e = 1, V = -kappa/r, kappa = 0.25: a0 = 4 cells, n = 64,
Coulomb by FFT with the exact 1/r kernel between all BCC cells). Field F = 0.01 au; alpha in m_e atomic units.

- **exact (relative)**: the two-body problem separates. The relative coordinate is hydrogen with the reduced mass
  mu = M/(M+1): E = -mu/2, alpha = 4.5/mu^3. Solved on the lattice as a check.
- **mean-field**: each population is a cloud in the other's average potential (two coupled lattice wave
  equations, iterated). This is what two swarms with self-pressure and no links reach as N grows, since the
  self-pressure is exact for one body.

`ref_M1836_M10.npz` (8 min):

| M | E exact | E relative (lattice) | E mean-field | alpha exact | alpha relative | alpha mean-field | rms width e / p (a0) |
|---|---|---|---|---|---|---|---|
| 1836 | -0.49973 | -0.50122 | -0.45364 (+9.5%) | 4.507 | 4.475 | 15.24 | 1.05 / 0.066 |
| 10 | -0.45455 | -0.45561 | -0.25411 (+44%) | 5.990 | 5.978 | 22.60 | 1.52 / 0.77 |

Mean-field fails even at the real mass. In it, the proton sits in the electron cloud's potential well and pays a
zero-point energy there. The well's curvature is 4 pi rho_e(0)/3 = 4/3 au, so the proton's zero-point energy is
(3/2) sqrt(4/(3M)) ~ 0.04 au at M = 1836, about 8% of the binding. Its polarisability is over 3x too large: the
proton slides inside the electron's cloud. In real hydrogen the electron follows the proton, and the proton's motion
costs only the reduced-mass correction (0.05%).

So two independent populations cannot be hydrogen. The populations must be correlated: the electron's constituents
respond to where the proton is, not to its average. This is Philip's "particles are not independent", here as a
measured 9.5% energy error and a 3.4x polarisability error.
