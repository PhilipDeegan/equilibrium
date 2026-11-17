# Winding pair under self-sourced lam — 2026-10-01 (next item 1, spinor discussion)

**INVALID (Philip 10-01): far too under-resolved.** Tube width w = 1.5 cells puts the tube cross-section and the
poloidal (q = +-1) twist that defines helicity at grid scale, where lattice dispersion / anisotropy dominate. The
numbers below are kept as a record of the setup only; they are not evidence for or against pairing. Needs
w >= 3 cells (r0 ~ 12, R ~ 28, N ~ 120k) and a second resolution to show convergence.

Question (mem project_mass_as_decoupling_pressure): do two c-transit windings bind each other into a lasting
periodic pair through the lam each raises, where a single winding cannot? Philip 10-01: nothing in E = mc^2 can be a
single winding -> ring / helix are controls only. A pair counts only if it beats the equal-energy single-winding
controls while still circulating.

Setup (`run.py`, `analyze.py`, `retention.png`): travelling waves u = env cos(n theta + q psi - omega t) on a torus
(r0 = 5, tube width w = 1.5 -- about 1.5 cells, under-resolved), n = 3, seeded as travelling waves of the actual
rule (phi from the seeded state's own face sigma, damped fixed point to 1e-10). Same u-energy for every case.
Rule as exp/2026/09/30: lam = alpha 2 e, implicit midpoint, DampedLattice sponge 0.25, R = 14, 20 crossings
(t = 242; one proper crossing = 12.1). Cases: ring (n,0), helix (n,+1), pair_opp (n,+1)+(n,-1) (opposite helicity,
co-travelling), pair_cnt (n,+1)+(-n,+1) (opposite helicity, counter-travelling), pair_same (n,+1)+(-n,-1) (same
helicity, counter-travelling = standing knot). alpha 0, 2, 8, 32 (initial tube sigma ~ 1, 0.56, 0.34, 0.17).
A first launch at alpha 36 / 100 with sigma = 1 seeding was stopped (frozen-clock start, mis-seeded phi); its partial
`*_a36.log`, `*_a100.log` are kept.

## Tube energy / E0 at matched proper time (tau = integral of E-weighted tube sigma dt)

| alpha | case | tau 5 | 12 | 25 | 50 | 85 | end split u/grad/sol |
|---|---|---|---|---|---|---|---|
| 2 | ring | 0.357 | 0.039 | 0.021 | 0.019 | 0.019 | 0.00/0.00/1.00 |
| 2 | helix | 0.355 | 0.051 | 0.017 | 0.012 | 0.012 | 0.00/0.00/1.00 |
| 2 | pair_opp | 0.470 | 0.115 | 0.024 | 0.021 | 0.021 | 0.00/0.00/1.00 |
| 2 | pair_cnt | 0.496 | 0.135 | 0.038 | 0.037 | 0.037 | 0.00/0.00/1.00 |
| 2 | pair_same | 0.284 | 0.034 | 0.008 | 0.002 | 0.002 | 0.08/0.02/0.90 |
| 8 | ring | 0.734 | 0.551 | 0.338 | 0.217 | 0.202 | 0.00/0.02/0.98 |
| 8 | helix | 0.769 | 0.589 | 0.363 | 0.209 | 0.193 | 0.00/0.01/0.99 |
| 8 | pair_opp | 0.709 | 0.522 | 0.297 | 0.205 | 0.182 | 0.00/0.01/0.99 |
| 8 | pair_cnt | 0.675 | 0.474 | 0.296 | 0.192 | 0.177 | 0.00/0.01/0.99 |
| 8 | pair_same | 0.599 | 0.310 | 0.112 | 0.067 | 0.067 | 0.00/0.00/1.00 |
| 32 | ring | 0.869 | 0.794 | 0.720 | 0.702 (end tau 30) | | 0.23/0.25/0.52 |
| 32 | helix | 0.889 | 0.823 | 0.760 | 0.745 (end tau 29) | | 0.30/0.22/0.48 |
| 32 | pair_opp | 0.851 | 0.779 | 0.700 | 0.680 (end tau 31) | | 0.26/0.22/0.52 |
| 32 | pair_cnt | 0.882 | 0.796 | 0.718 | 0.691 (end tau 32) | | 0.25/0.22/0.53 |
| 32 | pair_same | 0.867 | 0.768 | 0.663 | 0.586 (end tau 40) | | 0.21/0.22/0.57 |

alpha 0: everything leaves the tube within ~1-2 crossings (end <= 0.3%).

## Reading (INVALID, see top -- kept for the record)
- No pairing advantage. At alpha 8 and 32 the single windings retain as much as or more than either
  opposite-helicity pair at every matched proper time; the standing knot (pair_same) is worst everywhere. alpha 2:
  the pairs hold slightly more early (tau 5-12), but everything ends as 1-4% frozen circulation.
- alpha 2 and 8 survivors are 98-100% divergence-free phi = frozen circulation (1e: not survival); the mode
  amplitudes still circulating are ~1% of their seeds.
- alpha 32 keeps 0.6-0.75 and still circulates (u ~ 25%, mode frequencies ~ sigma x seed), but only ~2.5 proper
  crossings elapse: the frozen-clock regime of 09-30, and it is case-blind (ring = helix = pairs).
- Self-sourced lam holds energy in proportion to how much it slows the clock, whatever the structure: it does not
  couple two windings into a pair. Per the 10-01 plan this points away from "two windings in the existing rule" and
  toward two components in the rule itself (option (a): two c-movers per cell along the 8 hex diagonals, mass = a local
  coupling between them, lossy allowed), unless a different lam coupling is tried first.
- Not done: resolution (w >= 3 cells, larger R), other n / r0, longer proper time at alpha 32 (C++ midpoint needed),
  separation-restoring stiffness under a push on one winding (moot without binding).
