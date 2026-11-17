# Sponge assessment: per-cell (production) vs per-face damping — 2026-09-29

`sponge_ab.py`, discrete step(), numpy DampedLattice (current build), R = 12, sponge 0.2 (2.4 cells),
damp_max 0.02. "row" = production apply_sponge (each cell scales its own copy of each face);
"face" = both copies of a face scaled by the mean of the two cells' factors. u damping identical.
Logs: `sponge_ab_40k.log`; plots `A_point_source.png`, `B_pulse.png` (from the 40k run).

## A. Fixed-u point source (u = 1 at the centre cell)
| | 4k steps | 40k steps |
|---|---|---|
| row: window slope r in [2, 8.6] | -1.422 | -1.424 (late change 3.6e-4) |
| row: flux out of source / apparent charge at sponge edge | +1208 / -1194 | +12076 / -11642 |
| face: window slope | -0.075 | -0.0002 (u fills the domain, still creeping 8.8e-3) |
| face: flux out of source / charge at sponge edge | +975 / -951 | +2309 / -2309 |

A static u gradient makes phi grow linearly (phi_t = -W grad u), so a steady u profile needs the
source's flux to disappear somewhere. With the per-cell sponge it disappears into the mismatch
between the two copies at the sponge edge: the cells there see their own copies balance, so u
never registers the loss. That invisible sink is what holds the 1/r^1.42 profile. With the
per-face sponge the flux has to be removed through u damping, which at these settings can't hold
u down: u fills in to the source value and the profile goes flat.
=> The point-source "Coulomb" tests (test_point_source_poisson_uniform / _continuous, C++
quadratic_sponge, also per cell) measure a profile set by this artefact. item 2c (plateau -1.63 ->
-1.93) can't be interpreted until the static far-field BC is designed (e.g. u = 0 held at the
boundary).

## B. Outgoing gaussian u pulse (width 1.5), interior energy (r < 9.6)
Front leaves at t ~ 5; reflection returns and plateaus: row ~2.5e-2, face ~1e-1 of initial;
at t = 15.6: row 1.5e-2, face 4.7e-2. Reference (domain 3.5x larger, nothing back yet): 1.8e-10.
Both sponges reflect heavily at this width (2.4 cells ~ pulse width). The per-cell version
absorbs better partly through the same invisible sink. Fixing antisymmetry alone makes the
sponge reflect more; the sponge needs redesigning (width / profile vs wavelength, Nataf ABC/PML
theory), not just the copy fix.

## C. Continuous model (`continuous_ab.py`, C++ step_ae, DEC, iso weights, R = 8, h = 0.5, sponge 0.2
= 3.2 cells, damp 0.02; log `continuous_ab.log`, plot `continuous_ab.png`)
Interior sum(u^2 + phi^2 + e^2) / initial, window t = 6..10 (after the packet leaves, before the
reference domain's own reflection returns at t > 10):
| packet | reference | prod (step_sponge, E undamped) | row incl. E | face incl. E |
|---|---|---|---|---|
| T (rotational A) | ~1.5e-3 (static remnant) | ~0.5 | ~0.07 | ~0.07 |
| S (gaussian u) | 1e-12 .. 1e-18 | ~2e-2 | ~2e-2 | ~0.1 |
- Production continuous sponge leaves E undamped: the transverse packet comes back at ~50% of
  its initial amplitude, ~7x worse than damping E too. (The 09-25..28 dipole / induction
  benchmarks ran without a sponge; the continuous point-source tests use step(), not step_ae.)
- Per-cell vs per-face: no difference in T; per-face 5x worse in S (as in the discrete model).
- Every variant reflects 2e-2 .. 1e-1 at this width: none is an absorber yet.
