# Entries for 2026-10-02 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-02 — coils (+ evening: winding scan N 3-7, Hopf seed): loops / dense windings fail, a single-strand spring holds; C++ level-0 walk; the walk's background is grid-scale unstable below m ~ 1/sqrt2 (cure: m 0.8); N 4 spring persists (resolution-converged, rigidly turning, breathing); N 8 holds on a doubled torus until a 2-fold elongation sets in

1. **Coil** (`exp/2026/10/02/coil/status.md`; N 8, a 10, r0 30, xi 3.9, R72, 1200 ticks): no rigid rotation.
   Same sense: windings hold to t 120, inner legs (spacing 4 xi, half the outer) reconnect by t 180-240, outer arcs
   tilt and drift outward into the sponge, no core cells from t 600. `--alt`: neighbours collide head-on, merge into 4
   large loops by t 300, gone by t 360. Suspect: inner-side crowding (aspect 3); single loop vs mutual not separated
   (controls N 1, N 4 proposed).
2. **Spring coil** (Philip: loops were stellarator-like; a coil is a spring, turns draped at an angle; `spring.py`,
   status.md): one continuous strand on the torus helix (N 8 / N 4 turns), solid-angle phase seed (generic cone apex
   needed). Both self-propel along +z as a whole (net toroidal circulation = a ring) and lose the poloidal turns:
   N 8 pinches off 8 small rings and leaves a flat ring by t 600; N 4 stays one strand, flattens into a flat ring
   with a rotating 4-lobed wave. Probes fixed at z = 0 (turn count visual only); no state saved.
3. **Dense coils** (Philip: windings like a close-wound copper coil, a mesh of lines around the torus; 24 loops / 24-turn
   spring, xi 2.6, spacing 2-4 xi): both blow apart by t 300 (loops expand into the sponge, spring leaves a flat ring),
   then cell-to-cell variation over the whole domain. Walk-level: seeds force a pair-angle gradient N / rho rad/cell
   along the tube (dense 0.6-1.2, 8 loops 0.2-0.4, spring N 4 0.1-0.2: the only survivor as one strand); correlation,
   no mechanism. Philip: no "supersonic" / "sound" / "superfluid" -- envelope-layer words, reification.
4. **Discussion (Philip)**: steady-state target -> structure ~90% of domain (mem feedback_steady_state_domain_fill);
   vacuum electron state = Saha-type limit at infinite neighbour distance, plasma later; old saha noise was only a
   leakage/reintroduction test (mem project_ambient_vacuum_saha_limit); any friction must come out of the rule (what
   leaves a structure and whether it returns: periodic / reflecting cell), never be added. Walk explained in full (rule, GP limit, xi / c_s / kappa, caveats). Next: C++ level-0 walk port
   (A_h and nonlinearity as parameters, background ghost + sponge; periodic later).
5. **C++ level-0 walk** (Philip: port first, copper is solid because of the windings we're trying to create -- the
   coil is a shape guide, not a mechanism): `AdaptiveLattice::step_walk` + set_walk / set_walk_state / set_walk_sponge
   (continuous/adaptive_lattice.hpp, src/eq/lattice.cpp), `equilibrium/substrate/walk.py` (numpy reference Walk,
   CppWalk, cpp_sphere, healing_g, sponge_factor), `tst/substrate/test_walk.py` 2/2 (== numpy to 1e-12, 1 and 2
   threads; background stationary). 10-01 ring reproduced (0.079 vs 0.076 cells/tick); ~55M cell-ticks/s at 4 threads.
6. **Background stability** (`exp/2026/10/02/gradient/status.md`): the uniform background is unstable at grid scale
   (~0.44 g per tick; xi 2.6: 2.2%, xi 3.9: 1%), confirmed on a periodic box (no edge); not a splitting artefact;
   neighbour-averaged density reduces it. Every walk run is time-limited by it. Philip's call on the cure.
7. **N 4 spring with co-moving probes**: keeps 4 turns + toroidal circuit to t 1600 while travelling +z (0.027/tick),
   ends in the sponge -- earlier "flattening" was the fixed-probe artefact. First coil that holds.
8. **N 4 spring, C++ R110 from z -45** (`spring_cpp.py`, == numpy): intact to t ~1900 (4 turns, minor radius
   steady ~8.5, travelling +z 0.026/tick), then dissolves when the background itself breaks up (t ~2000-2200).
   R150 resolution check deferred: background growth per structure time is resolution independent.
9. **Background cure = mass** (`gradient/status.md` "Cure search"): neighbour-averaged density only reduces the growth
   (~7x); the background is stable iff arcsin m > pi/4 (+ O(g)), m_c -> 1/sqrt2 (clock < 8 ticks); m 0.5 was unstable.
   64^3 periodic box at m 0.8: flat 3000 ticks (xi 2.6 and 3.9). Walk small-k coefficient is n/(6m) (true core
   width sqrt(n) x the old xi label; healing_g(walk=True)).
10. **N 4 spring at m 0.8** (R110, true xi 3.9, 7000 ticks): intact throughout, minor radius steady ~8, travelling
   +z 0.0113/tick, background smooth -> on a stable background the coil persists. Four-way ring test parked: its
   small-m premise (relative-winding reach c / arcsin m ~ 6 cells at m 0.1) is on the unstable side.
11. **Resolution check R150** (true xi 5.2, 18.4M cells): converged with R110 (minor radius, torus radius, speed to
   ~1-3% in structure units). "Density leak" (Philip): deficit breathes, no trend over 7000 ticks; the xy panel was
   a fixed z = 0 slice. spring_cpp.py: co-moving slice, deficits per record, --recentre (lattice shifts) for long runs;
   20k-tick co-moving run started (periodicity of the shape, slow drift under the breathing).
12. **Long co-moving run** (20000 ticks): windings exact throughout; turns rigidly about z (best rotation = helix
   phase / 4 to ~1 deg; Philip: density constant while rotating); breathing damped (period 4268, decay 18.5k ticks,
   centre drifting ~7% over the run); late 2-fold precessing shape oscillation in beats (A_2 up to 0.19). No
   steady state yet, nothing broken.
13. **N 8 on the original torus** (m 0.8, R110): unwinds 8 -> 4 -> 0 turns by t ~1700 into a plain ring (Philip: "just
   turns into a torus"); toroidal circuit kept. Stopped at t ~7000.
14. **N 8 on a doubled torus** (r0 60, a 20, R125, inner turn spacing 31 cells as N 4; Philip: "awesome", "could lead
   somewhere"): 8 turns to t ~16000 (counted along the followed tube centre line, `turns_followed.py`; the circle probe
   misreads once the torus elongates), then 6 (17-18k), 4 (19-20k) as a 2-fold elongation grows (1.05 -> 1.53),
   turns lost in pairs at the long-axis ends; long axis near the y lattice axis (+-90 deg, t 10k-18k). Travels slowly
   (0.001 -> 0.007 cells/tick, accelerating), helix 0.0023 rad/tick. Spacing (not the 8:1 ratio) caused the
   original-torus loss.
15. **Central dip** (Philip: "spontaneous circle in the middle", t 2500, `centre_blob.py`): on-axis density hole to
   rho 0.013, ~10 cells wide, no winding round it (not a vortex), from the first contraction's inward focusing; weak
   repeat at t 6000-7000, gone by 8000. Same time / frame shift in the 22.5-deg rotated seed -> a property of the
   structure.
16. **Rotated seed (22.5 deg) -> elongation is LATTICE-LOCKED** (`*_th22.5_long`, `turns_followed_N8big_th22.5.log`):
   long axis -57 (t 5000) -> -82 / -86 / +85 deg (t 10k-13k), elongation 1.11-1.21 -- same axis (y) and rate as the
   unrotated seed (-83 / +87, 1.10-1.18); seed-locked would be ~ +-67.5. The walk's D2-only symmetry (x, y
   inequivalent) drives the 2-fold mode that strips N 8's turns. Full run: 8 turns to t 13000, 6 from t 14000 (2k
   earlier than unrotated), 4 at t 20000; elongation 1.25 (15k) -> 1.55 (18k) -> 1.32-1.35 (19-20k: partly recovers);
   long axis +62..+88 deg. Philip: maybe 8 is just not a correct winding number (-> next item 3). Recurrence (`recurrence.py`, Philip: t 6500
   "looks almost like the IC"): best-rotation rms vs t 0: 0.150 (t 500), 0.20-0.21 (t 2500-4000), 0.139 (6000), 0.134
   (7000, closest), 0.162 (7500); structure rms 0.18 -- a partial return about one breathing period (~5000-6500
   ticks) after the first relaxation, torus radius back to 62 (vs 60.4); compare against a relaxed reference next.
17. **Philip, framing (mems)**: walk is not a fluid -- no superfluid / supersonic / sound language (emergent-layer
   reification); no view from outside the system (Feynman checkerboard; physical claims = relations a structure or
   the background could register); u and phi were always preliminary; borrowing from Dirac / QM is fine with two
   requirements: no renormalization problems, never complex numbers in the rule (fine in the math / notation).
   Spinor-analogue question (curious only, not pursued): rule has the 2 pi sign; a (2, 1) spring and phase-per-
   rotation vs background would be the structure-level tests.
18. **Winding scan at fixed turn spacing** (Philip: "is 8 a correct winding number?"; `coil/winding_scan.sh`,
   status.md; N 3, 5, 6, 7, r0 20 + 5N, a 20, m 0.8, 13000 ticks): every N keeps all its turns to t 13000, no
   odd/even distinction, speed falls smoothly with N (0.0076 / 0.0048 / 0.0040 / 0.0033 / 0.0028 for N 3/5/6/7/8,
   t 1-7k). The y-locked 2-fold elongation at 13k grows with N (1.13 / 1.14 / 1.20 / 1.2 for N 5/6/7/8): N 7 carries
   the same mode as N 8 at the same size. CAVEAT: 13k is before N 8's own loss window (13-16k); no N 8 run survived
   (original torus 8 -> 0 by t 1700; doubled torus 8 -> 6 -> 4 from t 13-16k, both seeds). So "8 not singled out" is
   only "N 3-7 look like N 8 did at 13k"; survival past 16k is untested (spring_cpp.py has no resume).
   Why N 4 holds and N 8 not (partial): (a) turn spacing threshold (16 cells fails, 31 holds, ~4-8 xi); (b) N 8's
   2-fold elongation grows (1.05 -> 1.5) and turns go in pairs at the long-axis ends (squeezed spacing there?
   untested), N 4's stays bounded (A_2 <= 0.19); (c) confound: the surviving N 4 is on r0 30 a 10, N 8 on r0 60 a 20
   -- torus size and N not separated (N 4 at a 20 is the missing control). Lattice-causes vs lattice-orients the
   2-fold mode is open (resolution check).
19. **Tight winding (Philip)**: 8 is far too few; tight coiling may be ~60 windings -- "like a Hopf fibration" (the
   Hopf picture illustrates the TYPE of windings: nested, every line linking every other; the core is the toroidal
   loop, windings wrap around it -- not each turn a density-zero line as in the springs). Cost of N 60 as a
   core-per-turn spring at 31-cell spacing: r0 ~300, ~350M cells, ~22 GB per state, ~12 h per 13k ticks.
20. **Hopf seed** (`exp/2026/10/02/hopf/hopf.py`, status.md; R80, 2.8M cells, ~1 min per 2000 ticks): the k = 0 rest
   space is 2-fold degenerate -- diag(U, U) v is an exact rest state for every SU(2) U (residual 1e-16) -> a Hopf
   texture = which rest state each cell is in, density uniform at t 0. Degree-1 map, (alpha^q, beta^p) / norm, f(r)
   = pi / (1 + (r/a)^4), a 15: alpha = 0 on the ring r = a (the toroidal loop core), beta fills it with phase turning
   p times along the loop (p = windings around the core).
   p 1: density gathers on axis (peak 2.74, t 100), the texture travels -z and SHRINKS (ring 15 -> 6, t 300-650),
   a near-zero forms (rho 0.005, t 550), the flip vanishes by t 850 (unwound), rest radiates; nothing survives.
   p 8 / p 30: no shrink, but the windings along the loop are shed fast (flip gone by t ~250 / ~100) and the core
   empties -> a plain ring (rho min 0.002, radius ~10-12, -z at ~0.025/tick) persists to t 2000.
   Reading: twist along the loop stops the shrink, nothing holds the windings on the loop. Confound: beta^p also
   narrows the core fill (~ sin^p f), so p and fill width are mixed -> clean seed next.
21. Nothing committed (Philip commits). New library code: equilibrium/substrate/walk.py, C++ step_walk (header +
   bindings, built), tst/substrate/test_walk.py (2/2). Full suite not rerun. State saves: /bk/dst/2026/10/02/coil/
   (scan states every 1000 ticks; Hopf runs saved no state).

---


# Entries for 2026-10-02 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-10-02 (evening update) — start here; read `now`'s 2026-10-01 evening session

Uncommitted: exp/2026/10/01/{poloidal_ae, poloidal_pulse, weyl_bcc}/, winding_pair_r20 battery files, now, next,
doc/mems. .so rebuilt 10-01 19:08 (mod, mod_lattice). Working rule now: real-form two-component BCC walk + local
density turn (`exp/2026/10/01/weyl_bcc/`); step()-family structure tests are closed (step() cannot transport
circulation; fixed depletion traps nothing).

1. **8-strand poloidal coil** (`weyl_bcc/coil.py`, defaults: N 8 loops, a 10, r0 30, xi 3.9, m 0.5, R72, 2M cells,
   1200 ticks ~ 15-20 min numpy): do same-sense loops rotate the coil as a whole at ~ v_ring / r0 (0.0027 rad/tick)
   and keep their windings / radii, or reconnect / unwind? Control `--alt` (alternating senses). Then helical pitch
   (strands advancing toroidally per poloidal turn) -- Philip's call: closed loops vs helical, still open.
2. **Handedness check**: ring travelling along [111] (tilt the seed): the walk's first-order parity-odd error is zero
   along axes, where all rings so far travelled; compare q +-1 speeds along [111] vs [-1-1-1] at s 1 / 1.5 / 2.
3. **C++ port** of the real-form walk + density turn (8 hops, real 4x4, pair turn; threaded like step()): needed for
   16 strands (~R140, 14M cells); numpy is 1.3 s/tick at 2.8M.
4. **Longer ring runs** (periodic or larger domain, many crossings); ring pairs q +-1 (leapfrog / annihilate); what a
   perturbed ring radiates.
5. **Physics calls (Philip)**: what the background is (uniform a0 v = low-lam vacuum? relation to lam / depletion);
   whether m and g can be derived or must be calibrated (signposts: alpha, m_p/m_e, g-2); whether the density turn is
   the right nonlinearity (vs mass depending on density = decoupling pressure); unitarity not required (lossy ok).
7. **Walk vs step() at long wavelength**: does a coarse-grained walk (densities / currents of the pairs, or a
   background perturbation's sound sector) reproduce step()'s far-field behaviour (gravity via grad u, lensing,
   Coulomb profile)? If yes, step() is the walk's coarse limit and its far-field results carry over; if not, the
   two models conflict and that needs resolving. (Philip 10-01: the current step may be inadequate, but there are
   things to test.)
6. Older 10-01 / 10-02 items below (battery remainder, lagged load, Stern-Gerlach forcing, lam gradient, bath
   scoring, symmetric-state instability) are superseded for structures by the walk; keep for the step() rule only.

---

## QUEUED 2026-10-02 — start here; read `now`'s 2026-10-01 session

Uncommitted (Philip commits): inc/eq/substrate.hpp + src/mod.cpp (selflam midpoint kernel), equilibrium/substrate/
self_lam.py, tst/substrate/test_self_lam.py, exp/2026/10/01/ (winding_pair, winding_pair_r20), now, next, doc/mems;
untracked build.log / build_mod.log. Full suite 472/472 (tpp/phlop at 74a1050). Snapshots: ~/tmp/dst/2026/10/01/.

1. **Fixed centred depletion battery** (in-situ only, current action = speed only; Philip: run a few tonight, more
   10-02). Harness `exp/2026/10/01/winding_pair_r20/run.py --lam_shape S --lam0 L --case C` (verified 10-01 on
   pair_opp / shell 4). Question: which static, centred depletion shape traps a resolved winding (pair vs
   single-winding control), at a steady level and still circulating?
   - shapes: core (dense centre, Gaussian width 3), ball (fills r < r0 + 3w, tanh edge 2), shell (through the tube
     centre, width w), sheath (toroidal, d = 2w, width 0.75w), tube (fills the tube, width w = static analogue of
     the self-sourced rule)
   - lam0 1, 4, 16 (sigma_min 0.5, 0.2, 0.06); cases helix (control), pair_opp (pair_cnt / pair_same later, for
     shapes that hold) -> 30 runs
   - length: 10 crossings (t ~ 485, ~36k steps at dt = stable_dt, `--steps 36400`), snapshots every 2 crossings
     (`--snap_every 7280`), ~25 min per run at 8 threads (one at a time), ~1 GB each -> ~12-13 h, ~30 GB.
     Cheap first pass option: 3 crossings (`--steps 10920 --snap_every 3640`) for all 30 (~4 h), extend holders.
   - command (one run): `PYTHONPATH=$PWD OMP_NUM_THREADS=1 python3 -u exp/2026/10/01/winding_pair_r20/run.py
     --lam_shape shell --lam0 4 --case pair_opp --steps 36400 --every 40 --snap_every 7280 --tag _bat >
     exp/2026/10/01/winding_pair_r20/bat_pair_opp_shell4.log 2>&1` (run with the Bash background flag, not &).
   - scoring: E_struct (r < r0 + 4w) and E_tube vs t and proper time; winding amplitudes |a_(n,+-1)| and their late
     frequency (travelling vs frozen); grid-scale fraction; pair vs control at equal energy. Trapping = steady level
     while circulating, not slow leakage. Needs a plot_battery.py (extend plot_scan.py; rec col 8 = E_struct).
   - caveats: symplectic Euler H oscillation <= 8e-4 (midpoint ~7x cost if exact H needed); R20 = screening only
     (a positive result is a reason to go higher R, never a calibration).
2. **Stronger in-situ actions** (Philip's second facet, after 1): same shapes with depletion acting as impedance
   (sigma on one leg -> reflection, a wall not a lens), a throughput cap, or a steeper sigma(lam) -- Philip picks.
3. **Depletion as a lagged load**: availability recovers at a finite rate toward the local processing load
   (symmetric, period-averaged; tests whether the instantaneous response caused the collapse). Small kernel change.
4. **Option (a)**: two-component rule via exact Bloch symbols (mass gap, c-movers on the 8 hex diagonals, 2pi sign);
   quantised windings need a compact quantity (phi non-compact today). Check the BCC Weyl automaton refs first.
5. Older items still open: 10-01 block items 1a (in situ vs remote, wire L later), 1b (Stern-Gerlach as forcing,
   once something persists), 2-7 below.

---
