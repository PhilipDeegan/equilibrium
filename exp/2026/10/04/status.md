# Entries for 2026-10-04 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-04 — tight windings never pinch but cap at ~8 xi pitch (rule physics, not resolution: 2x agrees); twist = toroidal flow, leaks into the cubic lattice; C++ seed 57x; two-component (v1 / v2) structures: lone component rings and like pairs disperse, charge-2 full splits, opposite-component pair co-moves

1. **C++ solid-angle seed** (`exp/2026/10/04/pbox/status.md`): numpy seed was memory-bound (P 450 ~30 min); fork pool
   (bit-identical, only 2.3x), then `inc/eq/core/math/solid_angle.hpp` + `lattice.solid_angle` + `walk.solid_angle`
   (threaded, sums over box images), tests TestSolidAngle (vs numpy 1e-11, circle on axis exact): 57x, P 450 seed
   ~1.5 min. spring_box.py `--state_every`. /bk cleaned by Philip (387 GB free).
2. **Box P 450, N 8 a 20 parity, 20k**: all 8 outer legs pinch at 5k (P 346 copies joined 4 across the faces;
   pinch time ~5k in sphere / P 346 / P 450 = intrinsic); 8 closed rings (|w| 1, r 12-13) in two sets of 4, travel
   opposite to the main ring, out to the faces and back; 16k "stellarator" look = rings vertical on a circle at the
   main ring's radius but ~115 away in z (halfway to the copy), NOT threaded (ring_link.py); 20k fuse pairwise -> 4.
3. **Octagon = helix** (`exp/2026/10/04/octagon/`, ring_shape.py): R150 octagon r_8 / h_8 90 deg apart = N 8 spring
   tightened to a ~4.5 (~1.2 xi), not unwound (centre-circle probe unreliable through the core).
4. **Tight-helix scan** (r0 60, a 4.5, P 300): N 8 / 12 hold, N 16 shrinks, N 24 collapses by ~2k -> pitch floor
   ~25-31 cells ~ 7-8 xi = same as loose springs; N 60 that way ~350M cells. Plain-ring control: collapsed N 24 = plain
   ring (+ 0.5-cell non-helical 24-fold ripple, 3% slower). **2x resolution** (xi 7.8, r0 120, a 9, P 560, 44M cells,
   12k): same collapse on the xi-scaled clock (r_24 2.9 planar, h_24 0.6, helix phase lost by 6k) -> rule physics.
   24 trailing spiral arms ("Parker spiral") = waves radiated by the rotating lobes; periodic box = cavity (radiation
   returns); axis-facing parts deform first = copy proximity (4-fold in box runs likely the copies, not only the
   cubic term). Leftover = breathing wavy plain ring, persists.
5. **Toroidal flow** (toroidal_tube.py): the twist is a toroidal flow within ~2 xi of the core; N 8 keeps 89% of L_z
   over 10k, N 24 loses ~all by 5k; whole-box L_z NOT conserved (cubic lattice) -> leak may set the floor. v2 never
   fills the core (core_twist.py). **No tight run ever pinched** (one piece always); twist sheds as box waves (bg sd
   grows with N). Philip: "it seems like pinching is not possible" (mem).
6. **Item C (residual cubic)**, symbol only (`exp/2026/10/04/aniso/status.md`): split-step axial tick = dilated
   diagonal walk; one-axis ticks split the spin doublet at k^2 -> no simple axial compensation (D'Ariano-Perinotti:
   isotropic 2-component Weyl walks only on BCC). Parked.
7. **Two-component exclusion test** (`exp/2026/10/04/pair/`, pair_box.py, table.txt; Philip: photons superpose, what
   makes a two-component dipole hard): mixed vacuum (v1 + v2)/sqrt2, rings FULL or in ONE component. v1 + v2
   coincident == a full ring. E2 (v1 / v2, 8 apart): leapfrog then a compact co-moving pair to 8k. D2 / D (v1 + v1):
   fuse, v2-flooded core swells, box-wide v1/v2 bands by ~3-5k. B (lone v1 ring): disperses by 5k, same in P 450
   (box uniform term ruled out -> intrinsic; zero spin stiffness?). C (full + full coincident): splits into two unit
   rings that orbit ~10 apart (like windings can't share a place). Pauli discussion: not in the model; linearity =
   the about-the-average limit (Philip: "basically energy conservation", mem); walk turn is bounded (wraps at ~pi/g).
8. Paper read: Gao et al. Sci. Adv. 12 eaeg5930 (Fe5GeTe2 interaction-driven flat band + charge order; Kondo-like
   log T; press oversold). Process lapses: launched pair runs while the 2x run was alive; SIGSTOP guard made the loop
   start another -> 3 jobs (mem: check ps before launch).


# Entries for 2026-10-04 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-10-04 (evening update) — start here; read `exp/2026/10/03/aniso/status.md`, `pbox/status.md`, `coil/status.md`

Memory: run ONE big job at a time; background jobs die at 2 h (states every 1000 ticks -> resume).
A. **Box spacing scan** (does the 4-fold part come from the copies; does the 8-fold ripple survive): spring_box.py N 8
   --parity at P 450 (side 260, ~23M cells, seed ~40 min) and P 398 (side 230): 4- vs 8-fold amplitude of the end ring,
   ring turnaround point vs spacing, merge time. `cd exp/2026/10/03/pbox && PYTHONPATH=/home/p/git/eq python3 -u
   spring_box.py --N 8 --a 20 --r0 60 --P 450 --m 0.8 --xi 3.9 --walk_xi --T 20000 --n_plot 20 --recentre 500
   --threads 8 --parity`; analysis box_pieces.py (periodic pieces; winding sign arbitrary), shape harmonics as in
   shape_P346.log.
B. **Octagon ring as a structure**: is the flat ring + 8-fold ripple an attractor (seed it directly: plain ring radius
   ~60 with an 8-fold corrugation, sphere R150 --parity, and in the box); ripple rotation rate (save every ~100 ticks:
   direction aliased at 1000); does the ripple ever decay.
C. **Residual cubic anisotropy** (parity walk still pinches on x / y and squares up): design on the symbol first --
   mix the 6 axial (square-face) second neighbours, e.g. alternate a diagonal tick with an axial tick weighted to cancel
   the cubic k^2 term (aniso/compensate*.py pattern); must stay real, lossless.
D. **Merge in detail** (P 346): resume from state_t16000 with states every 100 ticks over 17-19.5k: how the 4 rings
   enter the hole and reconnect into the main ring; neutral corner blob at 13k (resume 12k, every 100).
E. **R125 --parity 30k -> 40k** (rotating square recurrence; 90 deg period ~7k): resume
   /bk/dst/2026/10/03/coil/springcpp_N8_S1_q1_a20_r60_xi3.9_m0.8_R125_z0_wxi_rc_par/state_t30000.npz (--T 40000 --n_plot 40).
F. **Neck approach law** (QM reconnection: delta ~ sqrt(kappa (t0 - t))): neck every ~50 ticks before a pinch (R175 from
   state_t5000 or R125 par from 18000), computed in-run (states are 1-2 GB).
G. Pair-production-like nucleation (Philip 10-03 idea, electron should not pinch): rings from a uniform background
   with a localized strong phase gradient, threshold? opposite windings? -- only if time.
Older items below (3, 3b-3d, 4) still open; 3c/3d now have the box.

---

## QUEUED 2026-10-04 — start here; read `exp/2026/10/03/coil/status.md` and `exp/2026/10/03/hopf/status.md`

10-03 done: Hopf seed dropped (Philip: only an illustration that windings are fibres); `spring_cpp.py` (10-03 copy) has
--resume / --snap_root, plots per run in nested dirs (name split on "_"), whole-domain plot with sponge circle;
`exp/2026/10/03/nest_plots.py`. Results: N 7 loses turns like N 8 (16k+); N 4 on the a 20 tube holds to 20k; big-tube
runs (N 4 a20, N 7, N 8, R167) breathe INTO the sponge -> their turn loss / y-parking are sponge-affected; N 4 small
torus (never near the sponge) unwinds 4 -> 2 at ~23k, then a periodic orbit (~5k breathing, rigid turning, torus
radius 29-31) to 40k = only rotating, non-expanding structure so far. Philip's target: rotates but doesn't expand;
big springs too "loose" (outer legs expand to wherever the sponge starts: R167 125, R175 150).
Memory: ~20-24 GB per N 8 xi 5.2 run; Chrome varies -> run ONE at a time. Background jobs die at 2 h: resume.

1. **[DONE 10-03: killed by Philip at 18k; rings tracked 12-18k, pieces.py / leg_check.py, coil/status.md: free rings
   persist and move AWAY (gap 13 -> 42), one leg pinches per ~5k ticks, sponge eats the old ones; resumable from
   state_t18000] Finish R175** (N 8, xi 5.2, sponge 0.14, r_inner 150, 29M cells; t ~10-12k: 4 outer legs pinched off as CLOSED RINGS (standalone tori, radius ~18) at the sponge edge, coil/status.md; Philip: interesting, maybe the dipole "pinching" mechanism -- finish it and track the rings: persist / travel / absorbed): started 13:00 10-03 from t 0, ~200 ticks/min at 8
   threads, killed at the 2 h limit (~24k). Resume from the last state to 28000:
   `cd exp/2026/10/03/coil && PYTHONPATH=/home/p/git/eq python3 -u spring_cpp.py --N 8 --a 26.6667 --r0 80 --R 175
   --sponge 0.14 --m 0.8 --xi 5.2 --walk_xi --T 28000 --n_plot 28 --recentre 500 --threads 8 --tag res --resume
   /bk/dst/2026/10/03/coil/springcpp_N8_S1_q1_a26.6667_r80_xi5.2_m0.8_R175_z0_wxi_rcres/state_tNNNNN.npz`
   (pick the latest state; if it already reached 28000, skip). Then turns_followed.py + core r max vs r_inner per state.
2. **[DONE 10-03: R167 to 28k; ~10k-tick breathing recurs (16k ~ 6k) while the long axis turns; parks on y ~20k, 2-fold
   elongation 1.08 -> 1.26 (28k), neck 2nd breath 16.1 < 18.9 -> same lattice-locked failure as R125, ~2x later.
   coil/status.md] Then resume R167** (exact 4/3 scale of the 10-02 N 8 R125, sponge at 125; resolution check like-for-like) from
   t 11000 to 28000, same command with --R 167, no --sponge, --resume .../springcpp_N8_S1_q1_a26.6667_r80_xi5.2_m0.8_R167_z0_wxi_rcres/state_t11000.npz
   (flat-named plots from before the layout change: nest with `python3 ../nest_plots.py . springcpp_N8_S1_q1_a26.6667_r80_xi5.2_m0.8_R167`).
   Compare R167 vs 10-02 R125 (turn loss time vs scaled time, elongation) and R175 vs R167 (sponge effect).
2b. **[DONE 10-03, see QUEUED 10-04 evening above] Walk parity compensation** (exp/2026/10/03/aniso/status.md; walk.py parity=True,
   spring_cpp.py --parity): N 8 R125 --parity held 8 turns, elongation <= 1.05, wandering axis to 18k (plain: 1.21 /
   6 turns at 13-14k); 19-20k cores pair up, elongation 1.14 on y, neck 13.4, first sponge contact. Resume to 30k
   (coil/springcpp_N8_R125_par_resume20k.log), then queued N 8 --parity on R150 (sponge 112.5) to 20k
   (coil/springcpp_N8_R150_par.log): sponge vs residual cubic anisotropy. Analysis: turns_followed.py, neck.py.
   R125 par result: pinched 4 neutral blobs at 23k (8 -> 4 turns), then a rigidly rotating 4-turn square, bounded, 23-30k.
   TODO after R150: resume R125 par 30k -> 40k (recurrence vs 23k / 30k, 90-deg period ~7k).
3. **The 2-turn N 4 state** (Philip: "doesn't appear to deform any further"): run on past 40k (resume
   /bk/dst/2026/10/03/coil/springcpp_N4_S1_q1_a10_r30_xi3.9_m0.8_R110_z0_wxi_rc_long/state_t40000.npz); seed an N 2
   spring directly (same tube) -> same state = attractor?; describe it (saddle ring vs N 2 spring: core on the tube
   centre in the meridional cut, pair-angle winding 2).
3b. **Pinch -> minimal loop** (Philip 10-03: CME-like pinching may produce proton dipoles; the dipole = the smallest
   loop that persists and cannot be pinched smaller; may apply to electron guns): smallest persistent ring radius
   (xi units, m 0.8; Hopf p 1 ring shrank 15 -> 6 and unwound, plain rings ~10-12 persisted) by seeding rings of
   decreasing radius; does a larger ring pinch again, a minimal one not; a pinch-off well inside the domain.
3c. **Confinement by neighbours** (Philip 10-03: isolated electron can expand (QM -> infinity); surrounded by others it
   is confined, cf. shell lattice): (1) two identical structures side by side (N 4 2-turn state or rings), free: do
   they separate (same vs opposite sense)? (2) periodic box = lattice of copies: does expansion stop at a size set by
   the spacing? (needs item 5's periodic boundary; scan spacing). Reframes "too loose" as maybe right in isolation.
3d. **Periodic box for 3c** (Philip: the Harris-sheet work had a periodic setup): that one (adaptive.wire_periodic_bcc,
   step()) broke on sq faces (BCC box incommensurate, mem project_harris_periodic_bc). The walk only hops along the 8
   hex diagonals, so a periodic walk on an integer grid with x+y+z parity and np.roll is exact -- already done in
   exp/2026/10/02/gradient/periodic_check.py (64^3 at m 0.8 flat 3000 ticks). Build the spring / ring seed on that
   box (numpy first; C++ step_walk periodic wiring for big boxes), box side = neighbour spacing.
4. Next items unchanged otherwise: periodic / reflecting boundary in step_walk (now also needed because the outer legs
   are held by the sponge), lattice-elongation question (redo on a big domain), N 5 / N 6 past 13k with room.

---
