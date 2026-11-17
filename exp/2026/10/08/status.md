# Entries for 2026-10-08 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-08 — walk results explained, then the walk recognised as a reified approximation; back to step(): its missing piece is a RELATION (self-transport of phi), not a quantity -- added, energy-exact, a smooth ring moves itself; a per-cell periodic tick phase gives exact whole-number circulation for free but its defects are lattice-pinned (no soft core). Morning: pair speed-up = superflow advection (no-fit, confirmed by a sign-reversing test); refinement rung shows the walk is converged/scale-free (cannot select N)

1. **Carriers** (exp/2026/10/08/couple/status.md, tails.py): - branch content ~1e-9 (not a carrier); deficit floor
   does not order the drift (rho_mid 0.96-1.00 in lone and pair alike); the flow does.
2. **Far flow** (farflow.py): uniform FORWARD u_z far from all tubes, box mean ~ 2 pi pi rc^2 / V per coil (87-90 %):
   lines outside the rings have zero winding, lines through the holes 2 pi.
3. **Prediction** (biot_exact.py, exact periodic Fourier sum, seed helix, core 3.9): partner dv +0.00097 / +41 / +21
   (x 1e-5) at P 240 / 300 / 360; measured pair speeds minus predicted totals 0.00222 / 0.00218 / 0.00216 (constant =
   self-term core model). q -1 vs z-sep: +0.00062 beside -> -0.00041 at the body centre, matches the 10-07 bins.
   Lone box (images): P 220 predicted +0.0002 over P 340, measured ~0 -> open.
4. **Decisive test DONE** (couple/scan_side.sh, drift_side_10-08.log): same-sense partner beside at P 300 runs
   0.01249 +- 0.00005 = -0.00064 (-4.9 %) vs lone; predicted -0.00062. Nearer partner, opposite effect: geometry of
   the flow, not distance. Superflow advection confirmed.

5. **Refinement rung f sqrt2** (exp/2026/10/08/refine, compare_f_10-08.log): N 4 at r0 7.9 xi and 10.4 xi with xi 3.9 ->
   5.515 (all lengths x sqrt2, ticks x 2) retraces f 1 to ~1-2 % in xi units and scaled time (rad / xi 2.67 -> 1.16 vs
   1.18 tight; 2.66 -> 2.00 vs 1.99 wide; Et / f within ~1 %); windings -4 throughout. Already converged: xi is the
   walk's only length, so refinement approaches its scale-free limit, which keeps N 4. N selection needs a second length.

6. **Walk vs step() quantified** (exp/2026/10/08/vs/status.md): Philip: we reified an approximation (walk = wave field on
   uniform cells, GP limit). Matched Gaussian bumps: same outgoing sound at long wave (L2 diff 0.08 at w 4 xi, eps 0.01);
   the walk disperses below ~4 xi (0.44-0.86 at 2 xi; unrelated at 1 xi) and adds amplitude-dependent transport (0.08 ->
   0.26 at eps 0.6). step() freezes any divergence-free A exactly (1e-15 / 200 ticks); the walk has no smooth vorticity,
   only quantized, mobile phase lines. step()'s missing piece = nonlinear self-transport of the flow. Density-weighted flux
   cannot help (curl of step()'s gradient update is exactly 0). Adding phi += dt kappa W (A x B).d (carry.py) makes a smooth
   ring move itself: 0.031 / time at a0 1, linear in a0, slower for larger R (ratio 0.68 vs 0.74 estimate), sign with kappa,
   shape kept; made exactly energy-conserving (wsum flow + implicit midpoint time-centred on the staggering: invariant to
   2e-6 over t 20, motion unchanged). No quantization (phi real-valued).
7. **Tick phase** (exp/2026/10/08/tick/status.md; Philip: no dimensions, KK = an operation: count modulo a period):
   phi = wrapped offset of a per-cell periodic theta. Smooth limit == step() to round-off; a seeded ring gives exactly
   whole-number, conserved circulation (696 defect loops, no slips) but is lattice-PINNED (frozen to t 30). Clock shift by
   squared offsets: kappa +-1 still pinned; kappa 5 moves only by pumping energy (x3). Mobility needs a soft core:
   candidate = junction coupling weakening with depletion where cells are far out of step. Not built.
8. **Side threads / Philip's statements today** (memory has each):
   - decay gate: a density-between-turns trigger cannot fire (turns are toroidally separated; tightening moves them apart,
     exp/2026/10/08/gate, tor_ridge_10-08.log). Philip: a true electron is quantized but deformable (size leeway); the
     continuum is more permissive than the discrete; expects N to converge under refinement (AMR).
   - cell.L: "how we discretize the continuous model"; = how much averaging over discrete cells the finest coarse cell does;
     NOT a physical knob; L0 ~1e18 is just a scale factor. L -> 1 = the discrete rule.
   - beta clock in the walk (walk.py beta=, C++ set_walk_clock, TestWalkClock): built + tested, but treats L as a knob ->
     to remove (Philip's OK not yet given). Finding kept: on-site L(rho) = a change of g; |beta| > g / (om0 + g) ~ 0.9 %
     makes the vacuum focusing.
   - "no such thing as spacetime" / "physics hides operations behind labels" (curled-up dimension = gibberish; the
     operation is counting modulo a period). Don't import frameworks (curved-space walks, KK) as pictures.
   - "we've reified an approximation": walk = wave field on uniform cells (GP limit); all walk results are walk-scoped.


# Entries for 2026-10-08 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-10-08 — read `now`'s 2026-10-07 evening entry, `exp/2026/10/07/slosh/status.md` (last four sections), `exp/2026/10/07/core/status.md`

Status of the 10-07 evening queue below: 1 DONE (N 4 r0 30 -> tight 4-turn ring by ~25k, settled to 40k; r0 40 holds);
4 DONE (same-metric core: relaxed 3 % thinner than GP; relaxed seed: ringing ~ mismatch, breathing coil-local);
5 DONE and extended (fitted drifts: q -1 net none, q +1 real, exponential in spacing, length ~31); 2, 3 open.

1. **What carries the same-sense coupling** (no runs): on the saved finals (/bk/dst/2026/10/07/slosh/pairq+1_*_gp8
   P 240 / 300 / 360 state_t08000, lone P 220 _gp state on fedora only -> rerun lone P 220 8k or use the _rp final)
   compare between the coils: deficit (1 - rho) along the body-diagonal line, phase gradient / flow, + / - branch
   content. Which one decays with length ~31? Then a theory check: does the walk at m 0.8 have an evanescent length
   ~31 (gap in the - branch? Bogoliubov at the coil's own frequency?), cf. xi / sqrt2 = 2.8.
2. **Spacing law, more points**: q +1 at P 420 (d 210, ~45 min) and P 210 (d 105; tubes must not overlap: assert
   2 (r0 + a + 2 xi) < lateral d sqrt(2/3)) -> exponential vs power; plus a different geometry (partner offset along
   x only, coaxial-lateral) to see whether it is distance or the body-centre arrangement.
3. **Opposite sense at fixed separation**: the q -1 pass gives +4.5 % near, ~0 far, both senses of approach. A static
   comparison needs a counter-propagating pair held at fixed z-sep -- impossible with free coils; instead repeat the
   pass at other lateral distances (P 240 / 360 q -1) and compare excess vs 3D distance with the q +1 law.
4. **N 4 tight state**: is it also preferred at r0 40 (seed a ~4-5, 20k)? One N 4 family (wide + tight member) vs
   floor-only? And does N 3 at r0 30 tighten too given 40k (it was "nearly settled" at 10k)?
5. Carried over: 10-07 evening 2 (Et with the (g/2)(1 - rho) term in shell_profile.py / quasi_energy.py on the 10-06
   finals; P 480 t8000 / t16000 kept, split E_+ / E_- first) and 3 (resume 10-06 P 340 to 16-24k: does the Et spacing
   trend survive -- its state_t08000 is kept). The ~2k residual oscillation: needs 16k+ windows.
6. /bk/dst: Philip to OK the remaining deletions (now 10-07 evening item 7). Keep pruning intermediates after each batch.

---
