# Cooling a thermal ambient state through a sponge — 2026-09-29

Question (Philip): can the ambient-state machinery find the closest thing to absolute zero?
Setup (`cool.py`): discrete step(), lam = 0, R = 12 (N = 9377, 61288 faces), thermal IC with equal
energy per canonical coordinate (u_i, phi_face ~ N(0, T)), DampedLattice sponge 0.2 / damp 0.02,
40k steps (t = 533, ~100 crossings). Energy split each 400 steps: u; phi's gradient part (global
least-squares projection onto W (psi_k - psi_i)); the rest ("sol", weighted-divergence-free).
Log `cool_R12.log`, plot `cool_R12.png`, final state + series `cool_R12.npz`.

## Result
NOTE: `cool_R12` ran on a stale 09-25 `equilibrium.cpp` build (step_cpp FACE_W = 3, i.e. the old 9:1
weights, with the new stable_dt). Rebuilt 09-29 12:16; rerun as `cool_R12_w3` (current sqrt3:1).
The frozen share (faces - cells + 1) does not depend on the weights; rates do.
`cool_R12_w3` (current weights, same t = 533 but c 2.31 vs 3.65 -> fewer crossings): u 4696 -> 3.5
(7.5e-4), still falling; interior sol 12893 -> 12826 (-0.5%); grad plateau 138 (interior 37);
antisym 3.55. Same picture; numbers below are the stale-build run.

- Frozen share of DOF: faces - cells + 1 = 51912 / 61288+9377 = 0.735. The IC puts 73% of the
  energy there.
- u (propagating sector): 4696 -> 1.5 (3e-4) and still falling as a power law ~ t^-1.6, no floor:
  the slow (near-zone-edge) modes trickle out. Third-law-like: zero approached, not reached.
- sol, interior faces: 12893 -> 12747 (-1.1% over the whole run, most of it in the first 4000
  steps). The divergence-free phi in the interior does not cool. Sponge-region sol is removed.
- grad plateaus at 211 (interior 106): with the global split this is the harmonic part (gradient
  of a potential harmonic inside, i.e. divergence-free there) -- also frozen, fed at the sponge.
- End state: flat u + circulating phi carrying 74% of the interior's initial energy.

Reading: the closest thing to absolute zero reachable by cooling from outside is the propagating
sector at zero plus frozen circulation whose amount is set by the history (initial T), not by the
lattice -- frozen disorder, like a glass's residual entropy. It does not fix a universal epsilon;
that needs something that couples the circulation back to u (lam nonlinearity) or discreteness.
Linear-rule fact (stability != reality).

## Side finding: the sponge breaks phi antisymmetry
`DampedLattice.apply_sponge` damps phi per cell row, so the two stored copies of a face
(phi_ik, phi_ki) are damped by different factors where the two cells have different sponge
depth: max |phi_ik + phi_ki| = 3.66 after the run (0 at IC). step()'s dphi is antisymmetric, so
the symmetric part created there is frozen for good. Damping per face (e.g. the mean of the two
cells' factors) would keep it antisymmetric.
