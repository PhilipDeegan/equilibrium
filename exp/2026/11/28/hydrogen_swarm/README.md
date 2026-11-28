# Hydrogen 1s as a swarm of constituents, next to Copenhagen, on the Kelvin lattice

Question: if the electron is made of N constituents with a local rule, does their static equilibrium around a proton
reproduce hydrogen's ground state? Does it respond to an electric field the way hydrogen does, leaning by the
measured amount and "fighting back" to a new equilibrium?

## Models

Both live on the same BCC (Kelvin) lattice: two sublattices of n^3 cells, cell volume 1/2, the 14-face Laplacian from
`exp/2026/11/27/bohm_free` (hex 2/3, square 1/3, which gives the continuum Laplacian). Units: m = hbar = 1, proton
charge Z (sets the scale: a0 = 1/Z cells, energy unit Z^2, field unit Z^3).

- **Proton**: fixed, a -Z/r potential centred on a corner cell. The centre cell uses its Voronoi average of 1/r
  (Monte Carlo). The field is added as + F z, the electron's potential energy for a field along +z.
- **Copenhagen**: the lattice ground state of H = -(1/2) lap + V (LOBPCG), with |psi|^2 as the density.
- **Swarm**: N constituents of mass 1/N at continuous positions. Each deposits onto both sublattices with trilinear
  weights, giving a density rho per cell. Energy:
  `E = sum_cells vol [ -(1/2) phi lap phi + V rho ]`, with `phi = sqrt(rho + eps)`.
  The first term is the von Weizsacker self-pressure: it penalises sharp density changes. There is no Coulomb
  between constituents. Each constituent is pushed by -grad(Q + V), interpolated from the cells it deposits to, with
  Q = -(1/2) lap phi / phi (the Madelung pressure). It relaxes overdamped, the step following the rms force, capped
  per constituent, and annealed over the second half. Equilibrium: Q + V is constant wherever the swarm is.
  eps = one constituent's density (1/N). With finite N the far tail has empty cells, and without a floor the
  pressure there is singular.
- **Field step**: start from the F = 0 equilibrium, switch on F, relax again. alpha = -(centroid shift) / F, in
  atomic units.

For one electron the continuum version of this swarm rule has exactly the Copenhagen ground state as its
equilibrium. So the swarm vs Copenhagen differences measure what finite N and local deposition do, and the
Copenhagen vs exact differences measure the lattice.

## Expectations (exact hydrogen, fixed nucleus)

E = -1/2, density e^{-2r}/pi, alpha = 9/2 = 4.5 a0^3 (dipole = -alpha F, the cloud shifts against the field direction).

## Runs

`hydrogen_swarm.py --out run.npz` prints convergence and PASS/FAIL; `analyse.py run.npz` writes run.png (density
slices, the radial profile against e^{-2r}/pi, and the field-on minus field-off maps: the lopsidedness).

| run | a0 (cells) | N | Copenhagen E / alpha | swarm E / alpha | wall |
|---|---|---|---|---|---|
| tiny (scratch) | 2 | 20k | -0.5106 / 4.16 | -0.5329 / 3.29 (not annealed) | 14 s |
| tiny (scratch) | 2 | 80k | -0.5106 / 4.16 | -0.5217 / 3.84 | 63 s |
| `run_a4` | 4 | 640k | -0.5015 / 4.47 | -0.5086 / 4.10 | 20 min |

Swarm energy bias ~ 1/sqrt(N) (4.4% at 20k, 2.0% at 80k), from the floor smoothing the sparse tail. The swarm's
energy-difference alpha is not usable: switching on the field disturbs the energy by as much as -(1/2) alpha F^2.
Only the centroid alpha is quoted. It is measured symmetrically: from the F = 0 equilibrium, relax once with +F and
once with -F for the same number of steps, then alpha = -(z+ - z-) / 2F. Leftover relaxation of the core, common to
both, cancels. A one-sided F - 0 difference showed a spurious central excess and gave alpha = 3.80.

`run_a4` (`--n 64 --Z 0.25 --N 640000 --dmax 0.04 --beta 0.9 --steps 1000 --steps-field 1500`):
- Copenhagen on the lattice: E -0.5015 (0.3% off exact), alpha 4.468 (0.7%).
- Swarm: E -0.5086 (1.4% below Copenhagen, the floor bias). Its density matches Copenhagen down to the
  one-constituent floor (1e-4 a0^-3, r ~ 4 a0); beyond that the sparse tail sits above it. That tail dominates the
  L1 of 0.066. alpha 4.10 (8% below Copenhagen). The (+F - -F)/2 density change matches Copenhagen's field response
  almost point for point (run_a4.png, bottom right). The centroid was still creeping at the end of the annealed field
  relaxations, so 4.10 is probably a lower bound.
- The rule reaches hydrogen's equilibrium and its measured lean in a field. The open errors are numerical
  (finite N, convergence), not a different equilibrium.

## Not yet

- Dynamics: this is static relaxation. A slow field ramp in real time, with Bohm particles on the Copenhagen wave
  alongside, is the next step.
- A proton population (phase 2): reduced mass with a light proton, and no net acceleration of the neutral pair in a
  uniform field.
- n = 2 (Stark mixing, where the lattice splits 2s/2p), and field ionisation (F ~ 1/16).
