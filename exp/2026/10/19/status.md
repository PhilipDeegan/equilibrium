# Entries for 2026-10-19 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-19 — zoo R 20 complete with rotation-corrected shape (coils precess rigidly; precession set by embedding, not knot type); glider momentum loss closed as moot; lattice geometry note (bookkeeping only); Philip: no single-cell events, no isolated cell, no spinors, no periodic systems, activation energy set by context; two-body dielectric test (wrong kind of body); throughput gradient in step() barely reaches a held ring; sigma inside the Cartan transport (csig) = steering only (refraction toward slower cells, energy exact); Philip: any emitter is a receiver, vacuum is not; periodic images are a lone ring's receivers; equal R 20 ring pairs: no transfer, the partner sets the in-flight energy (x0.8 same / x1.33 opposite sense); Philip: lowest-energy preference, offload because capacity is finite, cyclic offload with superposition (space holds up to 2x); carrier bound (vcap) sheds nothing; lam = rate side effect, capacitance structure-caused, distinct, reaches outward (Coulomb) -> next: finite holding capacity

1. **Zoo R 20 finished** (exp/2026/10/18/zoo/status.md 10-19 sections; logs exp/2026/10/19/zoo/r20/; end states
   /bk/dst/2026/10/19/zoo/end_*_R20.npz): added rms2r = best rigid motion (weighted ICP / Kabsch; recovers a known 26.3
   deg rotation to 0.1 deg). Ranking by rms2r: ring 1.7 < coil_1_3 2.8 < pair_co 4.2 < trefoil 5.1 < hopf_link 5.5 <
   concentric 10.2 < coil_3_2 11.8 < pair_counter 17.8 %. Coils precess steadily and rigidly (coil_1_3 0.24 deg / t,
   coil_3_2 ~1.0); trefoil (2 / 3, tube 0.45 R) is the same knot as coil_3_2 (3 / 2, tube 0.35 R) but turns 0.7 vs 33
   deg -> precession set by embedding. Counter-rotating sets shrink only 6-12x R 9 -> 20 (rings ~20x). Gotcha: a
   finished run leaves *_ckpt.npz -> rerun with the same --seed resumes (pass --ckpt or delete). 4-core cap (Philip).
2. **Glider momentum** (exp/2026/10/19/glider/status.md): mom() = D x B is the right conserved form; a 3 (smooth
   regime) still loses ~50 % Pz at P 100 / 140 / 200, and at P 140+ the packet collapses instead of dispersing: the
   scale-free rule always makes a grid-scale core that takes momentum. Moot under "no isolated cell". Closed.
3. **Geometry note** (exp/2026/10/19/geom/geometry.md + geom_plots.py, 5 figs): BCC cell (8 hex + 6 square faces, 7
   hops / 12 triangles / 6 tetrahedra per cell, Euler 0); 6 triangles round a <111> hop, 4 round a <100> hop = edges of
   the dual face; e = 1 on hops crossing a disk (R 4.2, 325 hops) leaves |b| = 1 on 108 rim triangles only (lattice
   Stokes, exact, integer). Philip: the single-cell ring is too sharp, never happens (background suppresses discrete
   local events) -> bookkeeping only; quantized units must be collective.
4. **Philip's statements** (memory has each): no isolated cell (neighbours ~1e-8 apart; measured zoo neighbour
   contrast 15-26 % -> our cells are coarse cells); no spinors (no vectors -> no spinors; belt-trick = SU(2) by another
   name; Rauch-Werner 4 pi is an observation to reproduce); 2 pi a product of pressures, no periodic systems (repetition
   is a context pattern) -> phase-slip / tick / Z_N routes out; activation energy is necessary (else ambient soup),
   context supplies it (planet-planet tension path; entanglement = discrete version); receiver felt via the tension
   path built by all scales; base substrate floors (Casimir: transmission with zero particles); two radiating bodies +
   the space between = one charge distribution, "charge" = average in-flight photon energy at a snapshot, floor,
   offload bounded by total energy vs Z0; without a receiver there is no gradient; a lone body may cycle energy
   locally (recurrent, like dipole field lines) but not far.
5. **Multi-winding ring, step() + Cartan** (exp/2026/10/19/zoo/status.md, R 9): speed 0.125 / 0.212 / 0.306 for n 1 / 2 /
   3 (Kelvin log law); circulation is a continuous label -> no whole-winding preference by construction. Walk / GP
   data: energy smooth in size (no hill), isolated rings held by conservation (impulse, linking), not a barrier.
6. **Two bodies, linear DEC** (exp/2026/10/19/pair/status.md, pair.py): dielectric spheres (Z0 / n) leak 1/e ~25 / 50 /
   ~250 at n 3 / 4 / 6; matched receiver accumulates 4-6x its geometric share, mismatched (n 5) takes then loses it;
   gap snapshot energy set by the emitter; emitter decay ~unchanged by B. Wrong kind of body (leaks with no receiver;
   the absorbing shell was a receiver). Closed-box runs stopped.
7. **Held ring in a throughput gradient** (exp/2026/10/19/grad/status.md; zoo.py --tilt / --dip, Eu + shell energies):
   R 20, tilt 0 / 0.01 / 0.03 / 0.1, T 30, dE/E 2e-8. Lone ring: Eu 0 -> ~16 (0.25 % of E), oscillates, shells flat
   to ~1 % -> bounded cyclic exchange (Cartan's exact part feeds u). Tilt adds Eu ~ g^2 (0.39 at 0.1 = 6e-5 E) and a
   linear sum|G| deformation; no threshold, no offload. sigma enters step() only; the transport ignores it -> the
   "receiver" never acts on what holds the structure.
8. **Throughput inside the transport** (exp/2026/10/19/csig/status.md; alat.set_cartan_sigma / zoo.py --csig; test
   TestCartanTransport.test_sigma_weighted): each triangle's Cartan term x mean edge sigma (-sigma i_v da) -> still
   antisymmetric per triangle, energy exact (dE/E unchanged). R 20 tilt 0.1: ring normal turns toward the slow side,
   1.40 deg vs 0.49 without (kinematic estimate 0.66); dip at the rim: +0.15 deg toward it, z speed -0.4 %. Eu, shells,
   sum|G| unchanged: pure kinematics (refraction / lensing sign), no exchange, no offload, no threshold. A rate-only
   receiver can steer a held structure, not make it offload -> offload needs an energy-carrying coupling.
9. **Philip: any emitter is also a receiver; raw vacuum is not one** (else photons decay into the background) ->
   receiver = another body of the same kind; sponges / absorbing shells are vacuum receivers (transient hygiene only).
   exp/2026/10/19/recv/status.md (R 9, zoo.py --struct side / side_a / side_b): lone-ring Eu vs box P 72-144: the
   first rise (0 -> 1.17 % -> 0.89 % by t 8) is box-independent (local dressing, once), later swings leave the large-box
   curve at times growing with the box -> the periodic images are the lone ring's receivers (10-19 R 20 "cycle" likely
   that). Pair A R 9 + B R 5.4 at P 96: shared (cross) energy shifts from B's half to A's, 5.5 by t 30, accelerating;
   Eu no trend; ambiguous (B overtakes A by 1.35 in z) -> needs R 20 + a cleaner observable.
10. **Equal rings R 20 side by side** (recv/status.md, P 192, 8 threads; zoo diagnostics now ~3x cheaper, identical):
   same sense: cross energy -5.5 % of E, split between halves constant to 1e-6 (no transfer); in-flight (Eu) energy
   up to 20 % BELOW two lone rings, pair slower (0.064 vs 0.075). Opposite sense: cross +5.4 %, Eu up to 33 % ABOVE.
   -> the partner sets how much energy sits in flight around / between the bodies (like / unlike dipole field energy),
   bounded, no offload. R 9 asymmetric shift = geometry (none at equal speed).
11. **Philip: lowest-energy preference** -- photons offload because a body's capacity is finite (a proton storing
   infinite energy -> no photons); thermalization IS the preference to equilibrate with the surroundings. step() +
   Cartan structures have no capacity limit (continuous circulation label, energy smooth in size) = the infinite-storage
   case -> explains no offload anywhere today. Next: a smooth capacity on what holds the structure.
12. **Philip: offloading can be cyclical** (A -> B -> A); photons superpose, so the space between holds up to 2x the
   energy (both directions in flight). Data: R 20 pairs in-flight Eu x0.8 (same sense) / x1.33 (opposite) of two lone
   rings = cross term of the overlap (0 .. 2x the sum); lone ring's late swings = cycling with its periodic images.
   Capacity scale check: R 20 ring core |a| max 0.275 vs C0 2 (14 % of c) -> a c-bound is far from n 1.
13. **Capacity form 1 = carrier bound** (exp/2026/10/19/vcap/status.md; alat.set_cartan_vcap / zoo.py --vcap; test
   test_vcap): R 9, vcap 0.6 -> 0.075 (core |a| 0.61): ring slows (v 0.111 -> 0.030), Eu DROPS (0.90 -> 0.09 % of E),
   outer shell grows less, nothing shed. A rate is not a store (as csig): capacity must bound what a cell holds
   (form 2 energy cost / form 3 lam), with the transport re-weighted to stay energy exact.
14. **Philip: lam is a side effect of a proton's presence** (the structure causes the lam gradient); similarly the
   structure allows for charge capacitance -- similar but different. -> capacity is structure-caused and distinct
   from lam (lam = rate side, steers; capacitance = holding side). "lam as holding capacity" (form 3) dropped.
   Philip: the Coulomb force reaches beyond the particle, so the capacitance does too (the space around holds the
   body's field). Reading: per-cell ceiling = substrate; structure's capacitance = its field spread over the cells
   around it; cells nearest saturate first (Schwinger-like) -> excess leaves. Uniform ceiling + geometry suffices.
