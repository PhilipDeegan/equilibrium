# Entries for 2026-10-12 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-12 — (a) soft core vs (b) Villain -> lattice pinning is not the obstacle; (b) + remainder == step() + commutator with integer labels; the one-triangle ring leaks, smooth rings cycle; whole-turn smooth rings widen because the commutator CREATES circulation in irrotational flow, and the error grows under refinement (R 9 / 20 / 40) -> rebuild transport in Cartan form. Philip: 2 pi = pressure equalisation, no pi / no time in a discrete model, rate from cell size, no gravity at electron scale

1. **Villain barrier** (exp/2026/10/12/villain/status.md, barrier.py): static energy landscape of a Villain ring over
   one lattice period, box 4R, R 10-40 (R 40: P 278, 5.4M cells): spread ~36-60 flat in R (no coherent pinning, kinked
   hop no extra cost), / E 0.8e-2 -> ~0.25e-2, ~0.1-0.2 of dE/dR x h -> breathes, does not pin. Sampling 12 -> 48 shifts:
   up to ~30 % low at coarse sampling.
2. **(b) + remainder is step() + commutator**: n <-> r transfer by 2 pi is phi-neutral, so integers are labels only.
   tick.py wrap == Villain with n = round. Tick-ring IC (vill.py, P 72 R0 9): kappa 0 frozen + whole to 1e-15; kappa
   0.132 smears quanta (off-integer 0.47 by t 0.5, 408 -> 15 defects by t 2), E runaway t ~3; dt x 0.5 identical.
3. **Geometry vs leak** (Philip: wrong geometry leaks instead of cycling): same rules, one-triangle core splits fast;
   11/lie smooth rings wt 1 / 1.5 / 3 hold (core rms +3.5 / +1.5 / +0.1 %, E 1e-5..1e-6). BUT those smooth rings link
   ~zero NET circulation (flow loops inside the tube) -- matched-circulation attempt invalid.
4. **Whole-turn smooth rings** (vill.py --smooth tau: curl-curl diffusion of the tick IC, linked turn kept 1.000):
   ring centred on a lattice symmetry leaves one face at exactly pi (wrap ambiguous) -> blow-up; offset (0.13, 0.07,
   0.11) fixes it (also affects today's / 10-08 tick-IC runs). Offset runs: E exact 1e-7, turn kept, speed 0.165 (R 9)
   / 0.085 (R 20), but |Gamma|-weighted core rms grows ~0.9 cells / time at both sizes (not scale-shaped).
5. **Cause = commutator** (farfield.py, one evaluation, fast fail): kappa 0 frozen; the term creates circulation where
   the ring has none (ring |Gamma| ~1e-16, term 1e-5..1e-6 / time wherever the through-hole flow is). Continuum: zero
   (advection of irrotational flow is a gradient). far / core ratio 0.337 / 0.552 / 0.720 at R 9 / 20 / 40 (shape
   scaled, tau = (R / 9)^2) -> rises with refinement: STRUCTURAL, not resolution. Net-zero rings held only because
   they have no far flow. R 40 seed build alone 40 min (not cached).
6. **Rate gradient** (exp/2026/10/12/rate/status.md): frozen ring ignores it; smooth ring drift came from sigma x
   commutator (breaks energy orthogonality, artefact); step()-only no clear deflection. Parked (Philip: no gravity at
   electron scale until refined near discrete).
7. **Philip**: 2 pi is pressure equalisation, not special -- quantization should emerge, not be forced; no pi in a
   discrete model (Z_N clock, integer counts, not U(1)); there is no time, only rate of change (dt = ordering; don't
   call a cell's state change a "tick"); rate set by local discrete cell size (gravity, dilation); cell.L kept for
   when needed. Practice: useful + under 10 min -> just run it.
7. **CORRECTION (afternoon)**: the whole-turn ring holds only to t ~4; T 20 shows a steady late leak (sum|Gamma| +34 %,
   rms 2.0 -> 2.9, speed 0.17 -> 0.10), identical for midpoint 2 / 3 / 6 and at P 72 / 120. Wake from a non-equilibrium
   core, or discretisation residual. R 20 to t 70 (C++): at matched travel 0.25 R sum|Gamma| +5.4 % vs R 9 +23 %, rms
   +14 % vs +33 % -> shrinks under refinement (~4x / 2.4x at f 2.22): mostly discretisation residual, converging.
11. **C++ port**: alat.cartan_transport / step_cartan (0.17 s / tick at 686k, 8 threads), 5 tests in
   test_continuous_cpp (== numpy 1e-13, step_cartan == step() + numpy carry 1e-12, physics identities).
8. **Core-width scan** (turning limit, Philip: triangle core = race car): holding sets in at core rms ~1.5-2 cells; leak
   per distance travelled ~independent of circulation (geometric, fixed in cells).
9. **Pair test** (exp/2026/10/13/pair/status.md): whole-turn pairs interact strongly (same sign co-move +30 %, opposite
   close and widen), net-zero ~10-15 %; all classical vortex-ring dynamics, no binding or threshold.
10. **Philip 10-13**: off-quantum shedding shouldn't happen (bad test); quantization reciprocal = activation energy
   before offload (discrete valve?); smoothness = pressure equalising (balloon); but no perfect equalising -- imbalance /
   energy gap is what bonds; pinch -> two balloons needs both halves supported; isolated electron delocalizes to fill
   uncompeted space (why QM says infinity).


# Entries for 2026-10-12 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-10-12 — read `now`'s 2026-10-11 entry, `exp/2026/10/11/lie/status.md`, `exp/2026/10/11/tick/status.md`

Direction unchanged (Philip 10-08): step() + relations between cell / face quantities. Transport now vector-free
(commutator, converged); quantization = tick phase (exact); missing = a DOF that lets defects take the transport.

1. **Philip's call**: core DOF for tick defects -- (a) soft / depleted core with depletion driven by the commutator's part
   theta cannot hold (proposed), (b) Villain integers, (c) wrapped phase + smooth remainder. Don't build before the answer.
2. **Build the chosen one** in numpy on a small box (P 60-80; estimate ETA first, nothing overnight; t 0 + progress prints):
   seeded ring -> does it unpin and move, circulation whole-number, energy exact? Compare speed with the smooth-ring
   commutator at matched flow.
3. **kappa**: derive instead of picking -- Euler-Poincare has no free coefficient in consistent units; find whether the
   lattice units produce the 7.58 (matched to carry) or something else.
4. **Tests** (tst/): tick smooth limit == step() to round-off; commutator antisymmetry + energy orthogonality (sum phi .
   term ~ 1e-15); t 0 speed rung (cheap, minutes).
5. **Port to C++** only after 2 settles; numpy commutator ~4 s / tick at 432k cells (single thread). f 2 refinement then.
6. Carried: walk beta-clock removal (Philip's OK pending, asked 10-08); /bk/dst deletions (OK given: 10-03 coil, 10-04
   pair/pbox, 10-05 R200, halo/scf, 10-06 N 8 R200 t4000, + 10-08 couple/refine states); MEMORY.md over the size limit
   (trim index lines); lit search for prior work on amplitude-free lattice defect mobility (before claiming novelty).
   Walk-scoped low priority: lone-box P 220 superflow discrepancy, self-term offset 0.0022, N 4 tight state at r0 40.

---
