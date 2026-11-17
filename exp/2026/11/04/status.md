## SESSION 2026-11-04 — pair regime map: separation +-24 / +-30 and opposite senses at +-18 (exp/2026/11/04/pair/status.md); Philip: no fields, only substrate; 4-core cap

1. Philip: there are no fields, only the substrate; all particles (atoms too) are bound states on it -> the 11-03 hold
   on the pair scan ("field-only merging, dendrites may need atoms") rested on a false split. Merging is a question of
   what the bound states / rule do (scale, count, context). Scan run.
2. pair.py (exp/2026/11/04/pair/) = 11-03 + --senses (per-ring sense, flips that ring's azimuth). Same body (R 8 a 3.2
   q 4 depth 0.8 --src local dmax 0.9 dt 0.05 T 300), each pair with its own solos at the same n (box effect ~2x).
   run.sh (8 single-thread jobs, pinned to 4 cores from ~13:00 -> 2.3-3.3 h each); compare.py PAIR SOLO_M SOLO_P CX.
   Saves /bk/dst/2026/11/04/pair/*.npz (full state + x-axis d / u^2 at t 100 / 200 / 300).
3. Results at t 300 (middle resid = d_pair - solo sum at |x| <= 4; toward / sideways per body; solos drift <= 0.2 /
   ~+0.3 same sign for both senses = box / lattice baseline; resid ~-0.15 in the rings at t 100 = kappa -7 % confound):
     same +-18 (11-03)  +0.43 saturated, u^2 x490   toward +2.85   sideways +2.65 / -2.65 (pair turns)
     same +-24 (n 128)  +0.05, u^2 x0.8             toward +0.50   sideways +0.64 / -0.64
     same +-30 (n 144)  +0.02, u^2 x52              toward +1.48   sideways +0.67 / -0.67
     opp  +-18 (n 112)  +0.15, u^2 x66              toward +2.68   sideways +2.08 / +2.08 (co-drift, no turn)
   No bridge at +-24 / +-30 within T 300, but no settled pair either: both approach, accelerating over t 200-300,
   and +-30 approaches MORE than +-24 (not monotonic in separation) -> two readings: onset delayed (T 300 too short)
   or box size (n 144 vs 128) matters. Opposite senses: same approach, 1/3 the middle deepening, the pair translates
   instead of turning: sideways motion follows the senses (10-25 gradient-drift / Magnus-like idea). All runs lose
   5-7 % energy at the rim by t 300 (solos too); d max falls t 200 -> 300 everywhere.
4. Runtime: 4 cores (cores 0-3 + HT siblings) for the session (Philip); 8 jobs on 4 cores ran ~2.5-3x slower than
   1 core each, own ETAs were unreliable early on.

## QUEUE 2026-11-04 — read `now` (STATE + 11-04 session) first. Older queues verbatim in exp/2026/MM/DD/status.md (after that day's now entries). 11-04 pre-tidy copies: /bk/dst/2026/11/04/{now,next}.pre_tidy. CPU: 4 cores unless Philip says otherwise.

Resume here (11-04 -> 11-05), all tied to doc/dst_versus.md A1-A3 and Philip's emission picture:
1. **Pair: delayed bridge or box effect?** (exp/2026/11/04/pair/, pair.py, run.sh, compare.py). +-24 / +-30 don't bridge
   by t 300 but approach, accelerating, and +-30 more than +-24. Decide between readings:
   a. +-24 at n 144 (pair + 2 solos): same box as +-30 -> is the non-monotonic approach a box effect?
   b. +-24 / +-30 to T 600: does the approach turn into a bridge? pair.py has no restart -> add one from the saved
      npz (PA, PB, VA, VB, uA, uB, ph, sqA, sqB, kappa) or rerun from t 0.
   Cost on 4 cores: unknown and slow (8 jobs took 2.3-3.3 h to T 300; T 600 roughly doubles). Run at most 4 jobs at
   once, estimate ETA from the first t 50 before committing, nothing overnight. Priority a (cheaper, 3 jobs).
   Also: opposite-sense sideways co-drift speed vs separation (gradient-drift check), solo --kfix control.
   Need a pair that SETTLES for step 2 (small kick on the emitter: does what leaves depend on the receiver).
2. **The collecting body** (main blocker for A1 detector, A2, A3 receivers, neutrino absorption): a body that releases
   content in steps set by itself. Constraints: sits near threshold, preloaded by the settled background (prompt
   photoemission); quantised units collective (10-19); capacity exists only as smooth set_cell_capacity. Under
   Philip's global picture the step is relational (emitter + absorber pair), so the pair (1) comes first.
3. **Settled background follow-ups** (exp/2026/11/03/bg/): does the converted (handed) circulation propagate and with
   what orientation; background axis vs lattice (z only); kappa; refinement. Emission-pattern test (needs a held,
   shedding body): does shedding align with the background axis, null along it (Zeeman)?
4. **Two readings of the photon** (W: spreads, unit set at absorption; P: held quantum on the settled background,
   guided by the background's disturbance). Test for P: held packet + its disturbance through two slits, do landings
   form fringes? Needs a transversely held packet (what holds it = background pressure balance?).
5. A1 relay / joint CHSH, A2 one-click, A3 receivers, neutrino absorption coupling, passive-target k-exponent scan
   (expected Rayleigh): parked until 1-2 exist. A1 bars: canal / ambient speed >~1e4 (sigma law caps at 1 + lam_bg);
   superdeterminism flag if settings belong to the global equilibrium (Philip's view open).

Done 11-04 (detail in now): pair regime map T 300: +-24 / +-30 no bridge but approach; opposite senses translate.

Live (in order) — all on the self-held depression body (exp/2026/10/25/jigsaw/ring2.py, status.md v5-v10):
0. **Mass scaling / universality of fall** DONE first pass (status.md v10): no trend over content x4 or counts 3-5 =
   consistent with universal fall at ~+-50 %. To sharpen: bigger box (n 96-112, box effect ~2x now), longer gentler
   push, average both senses; fixed-coupling energy scan needs a source / clip that doesn't run away.
1. **The depression outside the body**: P / d profile vs r (1/r tail?), its depth vs body energy, and whether an
   expansion-sourced body (the displaced units, the electron candidate) gives the opposite-sign tail. Two bodies: 11-03
   pair run -- same sign at +-18 approach and bridge (see Resume 1); opposite sign still open.
2. **Depth from the rule, not picked**: the hold needs core speed x <~0.1-0.25 (cells ~1/3). What sets it: saturation
   at the floor (dmax), the source / coupling strength with kappa fixed (depth then follows the body: A scan with
   --kfix), the pinch that formed it? Map the threshold vs coupling and body size.
3. **Count selection** (jigsaw): 3 / 4 / 5 all hold; only the mismatched 4.5 never settles. Needs the fit condition
   (Philip): windings at c fitted to the perimeter, units around the perimeter, linking of two windings? Signposts:
   proton 4 own periods per perimeter transit (r_p = 4 hbar / m_p c, predicts r_p = 0.84124 fm), electron 1/2 (720 deg).
   Candidate h test: a near-miss seed's shed per unit own frequency.
4. **Coarse ladder on the carrier**: the body at L = 1 vs the continuous model at L = 3 / 9 (face medium from P), now
   that something holds; needs the carrier ported to the continuous C++ (step + set_face_medium) or a numpy coarse box.
5. **Library**: if 0-2 hold up, move the carrier P + local-tick source into equilibrium/substrate (numpy) + C++ with tests.

Ideas (10-25, not yet tests):
- Electron = the displaced units (expansion) of the proton's contraction: equal counts -> exactly equal and opposite
  charge; mass ratio ~ extent ratio; neutron = both co-located (net zero); decay = the expansion leaving. Open: why
  the displaced units gather into one discrete electron; neutron excess 0.78 MeV not yet quantitative.
- Sideways drift that mirrors with circulation sense under a gradient = gradient-drift / Magnus-like: check scaling
  with gradient and body circulation; compare to a gyrating charge's grad-B drift.
- Strong / EM / weak / gravity as regimes of ONE depression profile: target ratios (strong ~1, EM 1/137, weak ~1e-6,
  gravity ~1e-39 at ~1 fm) and ranges (1 fm, infinite, 1e-18 m, infinite).
- Dithered unit defects (discrete): only rule-level; at real scale transparent. Junction rules for unequal neighbours
  and unit-move rate laws are the remaining discrete-level relations.
- Context-dependent h (offload step vs local rate) bounded by clock tests (alpha, m_p / m_e drift < ~1e-7-1e-6 per
  unit Delta U / c^2).

Questions for Philip:
- Pair: box check (+-24 at n 144) first, or straight to T 600? MHD two-body reference (spectral.py) still worth it?
- Bell: are detector settings part of the global mutual equilibrium (superdeterminism route) or free?
- The fit condition for the jigsaw (what are the pieces; what fits with what).
- Why the displaced units (electron) gather into one body instead of spreading.
- Flux vs energy as the depletion source: flux per local tick holds; background-tick flux starves itself.
- L sign / how a structure modifies lam_bg (09-30 / 10-01 / 10-22); walk beta-clock removal (commented, 10-25).

Parked / optional:
- Embedded spheromak: n / n-bar pair inside the 10-24 ambient (deprioritised; the ambient does not hold anything).
- Hybrid reference rung: spheromak in PHARE (~/git/phare/master); prior EMHD spheromak drifts along -z (09-26/27).
- Size law of a cell-scale hand-off (a 6 / 9 / 13), once an outflow route exists.
- Memory per cell (~0.6-0.8 KB) and continuous AMR (level 0 only today). (10-22)
- Finite propagation range (Philip 09-06): energy inside an expanding shell on the 14-neighbour lattice. Cheap.
- H r = const at fixed circulation (exact, 09-02) never tested dynamically.
- Precession scan of (p, q) torus coils (10-20); Rauch-Werner 4 pi as an observation (~9 h at R 20; think first).
- Downstream photon tests (polarizer, Eddington 2x vs 4x, Casimir d^-4) -- coarse only (no discrete photon).
- Derive sigma = ell and ell(depletion) from the rule (lensing GR/T 2.02, 09-25).
- Macro-particle scaling (09-27 idea); S_c refinement (10-21); minor whole-turn ring items (10-14 / 10-15 / 10-20);
  4/3 curl calibration vs the 09-28 weights.

Hygiene:
- C++ speed (11-03): remaining step_cartan cost is neighbour gathers in cartan_transport + slab copies in
  step_cartan (~8 %); 8 threads look memory-bound. Profile at 8 threads first; only if runtime blocks.
- Code hazards in doc/dst_caveats.md "Latent hazards"; psi_pair C2_CALIBRATED fitted on legacy weights.
- Casimir-Polder redo or withdraw (flagged in dst.md).
- BC leftovers after AMR (09-30).
- /bk/dst cleanup (exp/2026/10/18/bk_cleanup.sh ~73 GB, walk states ~25 GB, ckpts /bk/dst/2026/10/{14,18,19,20}/,
  R160 'dec' partials). MEMORY.md trim (~37 KB; stale: L8 "survives many causes" -> resolution, not depth).
- Lit: discrete Lie derivative (Elcott, Mullen, Pavlov), anapole / luminal nonradiating sources, Purcell, geons
  (Wheeler: wave held by its own depression -- today's body is geon-like), boson stars / Schrodinger-Newton.
