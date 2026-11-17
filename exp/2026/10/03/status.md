# Entries for 2026-10-03 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-03 — big N 8 springs fail by the lattice: the plain walk prefers axes (90 deg turn = its mirror); parity-image alternation restores cubic symmetry (delays, does not remove, the failure); pinching = like-sign fission or neutral blobs; persistent end states: rotating 4-turn square (sponge-held), octagon-rippled flat ring (R150, periodic box); periodic BCC box built, shed rings come back and merge

1. **R175 rings** (`exp/2026/10/03/coil/status.md`, pieces.py, leg_check.py; killed at 18k by Philip): legs pinch
   one pair per ~5k ticks; free rings (radius ~20, ~10k cells) persist, self-propel outward along their normals, gap
   to the strand 13 -> 42 (no reattachment in a sphere); the sponge eats the old ones. ring_charge.py: every ring has
   the parent's tube winding (like-sign fission, one turn each; turns conserved strand + rings).
2. **Neck** (neck.py: closest self-approach at geodesic >= 60): R175 34.6 -> 11.2 (6k) -> reconnected; R167 bottoms
   at 18.9 and reopens -> pinch threshold ~11-19 (R167 scale). Philip: knots tighten then loosen; tighter -> pinch.
3. **R167 resumed 11k -> 28k**: ~10k-tick breathing recurs (16k ~ 6k, rms 0.096 = +-1k neighbour level) while the long
   axis turns; parks on y ~20k, elongation 1.08 -> 1.26, 2nd neck 16.1 -> same failure as R125 ~2x later
   (resolution delays only). Philip: perfect periodicity is the aim (mem updated).
4. **Walk anisotropy** (`exp/2026/10/03/aniso/status.md`; Philip: "we might have to compensate"): spectrum(+1, R90 k)
   == spectrum(-1, k): a 90 deg turn maps the walk onto its mirror; relative anisotropy ~ k (8.8% at k 0.2, ~11% at
   the xi 3.9 core). Parity-image alternation (odd ticks: L and R exchange A_h <-> A_h^T) -> 90-deg asymmetry 0,
   residual cubic ~ k^2 (0.59% at 0.2), same kinetic term / rest state. walk.py `parity=True`, C++ step_walk +
   set_walk_parity, spring_cpp.py --parity, tests (periodic plane wave == symbol, C++ == numpy, background stationary).
5. **N 8 R125 --parity**: elongation <= 1.05, axis wanders, 8 turns to 18k (plain lost turns at 14k); 19-20k cores
   pair up + first sponge contact; 23k 4-fold pinch on the x / y axes into 4 NEUTRAL blobs (winding 0, ~400 cells);
   then a 4-turn square that ROTATES rigidly (-12.8 deg / 1k) and stays bounded 23-30k (sponge-held). Philip: an
   electron should not pinch at all, very maybe neutrinos form that way (mem). QM / spinor reconnection discussion:
   dissipationless reconnection at the core, sqrt(t0 - t) approach law, non-Abelian obstruction, TMR as overlap-
   matching analogy (not acted on).
6. **N 8 R150 --parity** (killed 15.5k): legs to the sponge by 4k, 4-fold ring pinch at 5k, turns unwind 8 -> 9k ->
   one flat ring (radius ~61, tube winding 1) with an 8-fold ripple ("octagon"), PERSISTENT 9-15k (amplitude 0.71
   constant, travels +z ~8.8 / 1k). -> without a confining agent the turns do not survive; the sponge holds R125.
7. **Periodic box** (`exp/2026/10/03/pbox/status.md`): build_bcc_box_periodic (even P, exact for the hex-only walk; the
   Harris incommensurability does not arise), cpp_box, test; spring_box.py: 27-image solid-angle seed minus the
   mean face jump (seam < 0.01 rad), exact periodic recentre. N 8 parity P 346 (side 200): turns 8 -> 4 -> 0, clover
   (9k) -> flat square ring + 4 rings at the faces (12k) that turn around, return, enter the ring's hole and MERGE
   (19k; Philip's "rings can reattach"); neutral blob at the box corner (13k); 20k one flat ring, 4-fold 0.55 +
   8-fold 0.38 (octagon with alternating sides; Philip: "ends up as an octagon again"). Box background rho sd ~0.11
   (seed transients stay).


# Entries for 2026-10-03 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-10-03 — start here; read `now`'s 2026-10-02 session (items 1-21)

Uncommitted: equilibrium/substrate/walk.py, inc/eq/core/data/lattice/continuous/adaptive_lattice.hpp (step_walk),
src/eq/lattice.cpp (walk bindings; .so rebuilt 10-02), tst/substrate/test_walk.py, exp/2026/10/02/{coil, gradient,
walk_cpp, hopf}/, now, next, doc/mems. Working rule: real-form BCC walk + density turn, C++ `CppWalk` / `spring_cpp.py`
(--recentre, --walk_xi, --theta0); m 0.8 (background stable iff arcsin m > pi/4); core widths with healing_g(walk=True).

0. **Hopf seed, clean (Philip 10-02: "we should try next")**: keep the core-fill profile fixed and change only the
   number of windings along the loop: beta = sin f e^{i p phi} (no sin^p f narrowing), alpha unchanged; p = 1, 8, 30,
   60 (a 15 -> p 60 is 0.64 rad/cell along the loop: use a 30, R ~140, ~15M cells, for p >= 30). Question: do the
   windings stay on the toroidal loop (filled core persists, n_z flip survives) or are they shed as with the
   beta^p seed (plain ring left)? Measure fill lifetime vs p and vs fill width; save states (hopf.py saves none yet).
   Then poloidal q > 1 (tight winding around the core) and the (p, q) torus-knot fibres the Hopf picture implies.
   Philip 10-02: N 8 (doubled torus) held 8 turns to t ~13-16k -> possibly salvageable; keep it in view (what
   stops the 2-fold mode: resolution, lattice isotropy, or a relaxed seed).
0b. **N 4 at a 20** (r0 40, R105, ~40 min at 4 threads): the missing control separating torus size from N in the
   N 4 vs N 8 comparison (now item 18); plus inner turn spacing at the long-axis ends when N 8 loses turns (saved
   states, analysis only).
1. **Lattice-driven elongation** (10-02 scan: grows with N, N 7 ~ N 8 at 13k -> this is the deciding item; also add a
   resume option to spring_cpp.py to take the scan states (/bk/dst, every 1000) on to 20k+) (10-02: rotated seed keeps the long axis on y -> lattice-locked): does its rate fall
   with resolution (N 8 doubled torus at true xi 5.2, or N 4 at R150 already saved: measure A_2 / elongation vs
   structure time) -> discretisation effect that converges away, or a real anisotropy of the rule (then: a more
   isotropic hop set / mirror-pair combination that stays real and lossless enough). Check the finished th22.5 run
   to t 20000 (`turns_followed.py ... 15000 ... 20000`).
2. **Steady state of the N 4 spring** (Philip's goal): breathing decays (tau 18.5k ticks) -- run 40k+ co-moving, or
   relax the seed first; check the 2-fold beats stay bounded; then define the steady state in the co-moving turning
   frame (speed 0.0116, turn 0.000675 rad/tick) and let it fill most of the domain (mem feedback_steady_state_domain_fill).
3. **[DONE 10-02, now item 18: no N rejected to 13k; elongation grows with N, N 7 = N 8 -> lattice, item 1]** Winding scan at fixed spacing -- is 8 an allowed winding? (Philip 10-02: "it might be 8 is just not a correct
   winding number"; N 8 big torus goes 8 -> 6 -> 4 as the y-elongation passes ~1.2-1.4, N 4 survives 20k ticks with
   bounded beats). N = 3, 4, 5, 6, 7, 8 at fixed turn spacing (~8 xi, r0 scaled with N): do a few N hold
   indefinitely and others decay to them (selection) -- odd N cannot be paired by the lattice's 180-deg symmetry --
   or does every large-N spring follow the lattice elongation (then item 1 decides)? Also N 8 at true xi 5.2.
   Measure speed and turn rate vs N too (N 8 travels ~3x slower than ring scaling, cf. Oberti & Ricca 2019).
4. **Recurrence / periodicity** (Philip: t 6500 "almost the IC"; a periodic system repeats for as many cycles as we
   run): `recurrence.py` against a relaxed reference over MANY cycles -- N 4 long run (20k ticks ~4.7 breathing
   periods, states every 500) first: returns tightening (-> steady turning state) vs constant (-> periodic orbit);
   N 8 returns will drift with the lattice-locked elongation.
5. **Periodic / reflecting boundary** in step_walk (cell context; friction from the rule's own exchange, mem
   project_ambient_vacuum_saha_limit); full test suite rerun after the C++ change.
6. Parked: four-way (L / R / relative) ring test (needs small m, now unstable); uniform-gradient k_max at m 0.8;
   (2, 1) spring / phase-per-rotation (spinor analogue, Philip: curious only); walk vs step() far field (item 7 below).

---
