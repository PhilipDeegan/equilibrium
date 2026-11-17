# Entries for 2026-10-13 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-13 — Cartan-form transport -i_v(da) replaces the 11/lie commutator: vector-free per-triangle rule, energy-exact, creates no circulation in irrotational flow, kappa = 1 (Euler) matches lie's ring speed; whole-turn smooth rings keep shape to t ~4 (R 9 +1.6 %, R 20 +0.22 % vs lie +108 % / +58 %) but leak later (+34 % sum|Gamma| by t 20, item 7); in library + tests; pair test classical

1. **Rule** (exp/2026/10/13/cartan/status.md, fit.py): per Delaunay triangle, edge e += omega_t (a_next - a_prev) along
   the loop, x 3/4 / W_e^2 (a = phi / W). Fitted antisymmetric flux coupling hex-hex : hex-square = 1/2 makes the sum
   over a face's triangles exactly i_u omega for uniform curl / flow (residual 2e-16). No vectors (loop sums and the
   triangle's other edges). Gradient part d(i_v a) left to step()'s u sector.
2. **Checks** (diag.py): antisymmetric exactly, sum phi . T / sum |phi T| ~1e-18; far / core circulation creation
   7.8e-4 / 4.4e-4 at R 9 / 20 (lie 0.337 / 0.552) -- the residual is the seed's own far circulation.
3. **kappa = 1**: net-zero ring speed matches lie (0.132) at kappa 1.005 -> Euler's coefficient in units where a is the
   velocity 1-form; the 7.58 was the commutator's normalisation. Closes the kappa-derivation item.
4. **Whole-turn leak test** (vill.py + --flow cartan): turn 1.000000, E 1e-6 / 1e-7, sum|Gamma| flat, speed 0.17 / 0.077
   (R 9 / 20), core rms +1.6 % / +0.22 % over t 3.56 (kappa 0 frozen exactly). R 40 not run (trend clear; seed 40 min).
5. **Library**: continuous.cartan_transport(lat, phi) + tst/substrate/test_continuous.py TestCartanTransport (3 tests,
   3 s): antisymmetry + energy orthogonality, zero on gradients and away from a localized vortex, exact uniform limit at
   h 0.5 / 0.25.
6. **Consequence**: Kelvin holds by construction -> an off-quantum loop keeps its circulation; "seed off-quantum, look
   for shedding" (10-12 plan) is a null test under step() + Cartan alone. Quantization needs a further relation.


# Entries for 2026-10-13 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-10-13 — read `now` 10-12 + `exp/2026/10/12/villain/status.md` (+ `exp/2026/10/12/rate/status.md`)

State: lattice pinning is not the obstacle; (b) + remainder == step() + commutator (integers = labels); the whole-turn
ring widens because the 11/lie commutator creates circulation in irrotational flow, error rising with refinement
(far / core 0.34 / 0.55 / 0.72 at R 9 / 20 / 40). Fix the transport before any more geometry work.

1. **Cartan-form transport** (~2-2.5 h incl. R 20): continuum L_v a = i_v da + d(i_v a) -> where da = 0 the change is a
   pure gradient. Build da/dt = -kappa i_v(da) + exact on the lattice (triangle loop sums da, flux v = W phi), i_v
   antisymmetric in the flux so energy stays exact (sum v . i_v da = da(v, v) = 0). Vector-free options from 10-11:
   discrete wedge (Hirani / FEEC), Elcott loops. Design first (~30 min), numpy next to lie() in vill.py.
2. **Checks at P 72** (minutes): circulation created in curl-free flow ~1e-15 (farfield.py ratio -> ~0); energy exact;
   antisymmetry; smooth net-zero ring speed vs 11/lie (0.0328 at R0 12 wt 3, kappa matched).
3. **Whole-turn leak test**: vill.py --smooth (R / 9)^2 --off 0.13 0.07 0.11 at R 20 (P 140, ~15 min), then R 40 (P 278:
   seed 40 min + run ~85 min -- cache the seed to /bk/dst, start early; no overnight). Measure core width with a
   core-only weight (|Gamma| above the far-field level), not raw |Gamma| (raw picks up small spread circulation).
4. **Then geometry**: which shapes cycle locally under the fixed transport (Philip: wrong geometry leaks) -- tokamak
   core + windings, Hopf-wound ring, twisted ring. Quantization must EMERGE (Philip: 2 pi = pressure equalisation):
   seed off-quantum circulations, look for shedding to a preferred value. Do not build an amplitude zero to force it.
5. **Z_N clock** (no pi in a discrete model): integer state per cell, offset = count difference mod N, loop closure =
   integer. Update rule is Philip's call: what drives a cell's state change (rate from cell size, uniform at electron
   scale; neighbour imbalance?). Propose, then leak-test closed loops vs N.
6. Parked: rate gradient / gravity (electron scale, until refined near discrete).
7. Carried: kappa derivation; tst/ tests (tick smooth limit == step(); commutator / Cartan antisymmetry + energy
   orthogonality; farfield fast fail; barrier closure check); C++ port after transport settles; walk beta-clock OK;
   /bk/dst deletions; MEMORY.md trim; lit search (amplitude-free lattice defect mobility; structure-preserving discrete
   Lie derivative).

---
