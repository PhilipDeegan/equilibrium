# Entries for 2026-10-24 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-24 — Philip: no E / B (find the geometric mechanism whose effects E and B describe); a reference ladder (continuum Maxwell / resistive MHD / QED situations rebuilt from DST primitives) before any decay mechanism; 10-22 "unwinding" numbers were E/B reconstructions (not windings) -> holonomies; native look at the spheromak states

1. Native slices (exp/2026/10/24/geo/slices.py, a9.png, a20.png): square lattice loop circulations (4 square hops,
   edge integrals phi / W) in the meridional (x-z) and equatorial (x-y) planes + face content. Seed reads right (two
   opposite toroidal lobes; poloidal core + return ring closing inside a). a 9 t 200: all grid scale. a 20 t 200: lobes
   intact, face content smooth, but grid-scale loop spots already sit in the ring r 10-20 (91 % of the loop deviation from
   the seed), outside the magnetic axis. t 450: whole interior salt-and-pepper, content still inside a, slight +z plume.
2. Onset is lattice-locked (azimuth.py): equatorial ring r 10-20 at t 200, angular power |c_m| / c_0 m 4 0.66, m 8 0.32
   (m 1-3 <= 0.13); m 4 maxima at 44 deg = projections of the hex (body-diagonal) faces, not the square axes. -> the
   10-22 "structural" reading needs qualification: same turnover onset at a 9 / a 20, but its orientation is set by the
   lattice. Philip: at the real cell size (~1e-44) everything happens on the lattice, so "lattice-locked" is redundant ->
   orientation is not the finding (here it is the coarse anchor mesh's own anisotropy anyway); the finding is that
   content runs down to the smallest loops available. At real scale that is a cascade over ~30 decades, which a
   persistent structure does not show -> the rule needs a stop at structure scale. (1,1,1) rotation test dropped. Philip: the substrate may not be Kelvin cells either -> the true lattice is undefined; mechanisms must be lattice-agnostic (loops, adjacency, content, hand-off), lattice-free references are the target.
3. Reference ladder BUILT (Philip go): equilibrium/reference/spectral.py (periodic FFT box: Taylor seed matching sph.py's
   smoothing, vacuum Maxwell exact, held current exact, diffusion exact, incompressible resistive MHD RK4-IF 2/3
   dealiased, 1D Vlasov-Poisson); tst/reference/test_spectral.py 5 pass (Beltrami static / exact resistive, ideal E and
   H to 1e-6, vacuum energy, held static, Landau gamma -0.1533 omega 1.4156). Runs exp/2026/10/24/ref/ (a 9 geometry,
   side 55.4, amp 0.3, saves /bk/dst/2026/10/24/ref):
   vacuum: content out of r < a by t ~20 (light crossing), half to the partner sector -> unheld = gone in ~a / c.
   diffuse eta 0.05: smooth decay + outward spread, lam a 0.89 -> 0.39, no fine content.
   MHD (v seeded 1e-4), T 1500 = 50 Alfven times: HELD in place, no breakdown, no tilt, flow ~1e-4 of EB, fine-scale
   share <= 2e-4 throughout. Decays self-similarly at the Taylor rate 2 eta lam^2 (eta 2e-3: EB 0.315 vs 0.30 predicted);
   helicity outlasts energy (eta 5e-4 / 2e-3 / 1e-2: EB 0.72 / 0.32 / 0.022, H 0.75 / 0.36 / 0.036); lam drifts down
   (relaxes to larger scale), content spreads outward slowly. vs DST step() + Cartan: breaks into grid-scale loops at
   ~3.4 turnovers (~100 time units) with no decay -> the target behaviour is held + slow self-similar decay.
   Philip: Landau damping is another required offload mechanism (collisionless, resonant particles, phase mixing) ->
   rung 6 = VlasovPoisson1D (memory project_offload_mechanisms_reference_ladder).
   compare.py / compare.png (turnovers a / amp, content = DST sum phi^2 vs reference EB): first ~3 turnovers DST and
   MHD readjust the same way (shape change ~0.1 = the non-Beltrami seed settling, real in the continuum too); then MHD
   flattens (0.07-0.19 by 12 turnovers) while DST runs away at ~3.4 (1.2-1.3 by 6.7); DST content flat (no decay) vs
   MHD Taylor-rate decay; share inside a held in both until DST's scramble spreads.
4. Hand-off of unmatched loop content (exp/2026/10/24/handoff/ho.py; numpy over C++ step_cartan): loop sums w = D1 phi
   on the 12 Delaunay triangles / cell, context M = same template averaged over cell + 14 neighbours, loop partner e:
   e_t = g ((I - M)^p D1) phi, phi_t = -g (...)^T e -> 1/4 sum phi^2 + 1/2 sum e^2 exact (1e-5 measured), gradients
   untouched, g calibrated to lattice Maxwell at p = 0. a 6 P 64 (turnover 20), T 150 = 7.5 turnovers:
   baseline cos 0.97 -> 0.08 (breaks ~3.3 turnovers, fine share 0.87); p 1 = drains the structure in < 1 turnover
   (vacuum-like: one-cell context calls 13 % of a 6-cell structure unmatched); p 3 held (cos >= 0.96 throughout),
   content 0.86, ~2 % / turnover (MHD eta 2e-3-like); p 4 delays breakdown to ~6 turnovers.
   BUT p 3 at a 9 P 96 (T 225): hand-off ~9x weaker (Ee at 2 turnovers 0.0033 vs 0.030; (h / a)^6 predicts 11x) and
   the structure breaks at ~5 turnovers (cos 0.89 at t 150, 0.03 at 225) -> rate is mesh-set = hyperviscosity, fails
   refinement; NOT the mechanism. And e mostly stays local (a 6: 55-70 % of e inside r < a) = local store, weak outflow.
   Philip: the mesh spacing is more fundamental than structure -> a cell-set rate is physics; a 6 / a 9 in these
   (L = 1) runs are different physical structures (a / ell), so the rate law vs a / ell is a prediction, not a failure.
   Kept requirement: the partner
   must carry content away (propagate), not ring in place.
5. Propagating partner (ho.py --photon / --sponge; v1 kept as ho_v1.py): loop partner e shared with a full lattice
   Maxwell pair (psi on faces <-> e on loops, gamma = g_c, unfiltered D1), sponge on the pair at r > 0.38 side with the
   absorbed energy tracked; energy (phi + pair + absorbed) exact to 5e-6. a 6 p 3, T 40: hand-off collapses (pair 0.24 %
   vs 3 % for the local store), 90 % of it inside r < a, nothing reaches the sponge, the cascade proceeds (fine share
   0.0004 -> 0.047, cos 0.953). Reading: the unmatched content varies at the slow Cartan rate, so the Maxwell pair just
   carries its near field (a steady current does not radiate; Larmor ~ omega^4). Outflow needs the fine content to be
   turned into fast local oscillation first (heat), which then emits = two stages (store -> emission), not one.
6. Receivers (Philip): cell-scale stores would be invented -> neutron / antineutron analogue instead: spheromak + mirror
   (toroidal winding flipped, helicity negated; assumption for "anti"). spectral.taylor_spheromak(center, hel),
   spheromak_pair. MHD refs (exp/2026/10/24/ref/pair*, a 9, touching sep 2a, eta = nu = 2e-3):
   unpushed: closed bodies do not see each other (T 600: dz 18.0 -> 17.8 counter, 18.0 co; decay as singles).
   push 0.1 (1/3 vA): compress to dz 10.5 and bounce, co and counter alike (counter EB 0.64 vs co 0.71 at T 400).
   push 0.3 (~vA): co = compress (dz 5.2), bounce, separate (dz 17.9, overlap 0.88, EB 0.77); counter = stronger
   interaction: fine share 0.11 at contact, EB 0.46, stays bound (dz 12.8), toroidal share drops -> annihilation-like
   loss + bound remnant vs co elastic bounce. (Experiments merge with walls / gun-made open flux; isolated Taylor
   spheres bounce.)
   DST step() + Cartan a 6 P 64 touching (dst_pair.py): co and counter IDENTICAL to 3 digits, dz 12.0 -> 12.1, each
   body breaks down on its own (|w|^2 x33, |dphi| 1.3) -> no interaction; and DST has no separate velocity to push with
   (motion comes from a body's own shape; Beltrami = static) -> how do two bodies approach in DST? open, for Philip.
   Philip: no dipole / E / B field -- only disturbances of the substrate; deviation from the mean is an observation.
   -> a real neutron is observed from outside (moment), so its disturbance must extend past the core; the closed
   spheromak has none (content beyond 1.5 a = 0 in every run) = unobservable, non-interacting. Candidate tail: the
   substrate depression around a structure (10-22: structure = depression, cells contract near it), absent from
   step() + Cartan; two bodies interact where their depressions overlap.
7. Embedded spheromak (Philip: every neutron sits in larger-scale disturbances; exp/2026/10/24/embed/dst_embed.py,
   mhd_embed.py): spectral.beltrami_ambient (box-scale force-free B0 (0, sin kx, cos kx), uniform magnitude, along +z
   at the body; steady under Cartan), rot_x / R for body orientation. DST a 6 P 96 T 150 (turnover 20), ambient
   frac 0.1 of body peak: the ambient does NOT stop the cascade -- loop content x10 at t ~80 in all cases
   (baseline 10.3, aligned 10.0, opposed 9.2, tilt45 17.8 = fastest), end x30-38; body content inside a 0.63-0.71
   (baseline 0.70); body carried along by the ambient (cen z +3.4 aligned, +2.7 tilt, +1.3 opposed, +1.1 none);
   ambient exchange ~ -0.5 % (small). Opposed slightly slower than aligned; tilt destabilises. (overlap partly
   reflects drift, use loops / inside.) frac 0.3 and the MHD side NOT run (Philip left; commands in `next`).
8. Earlier proposal (kept): reference ladder (vacuum spheromak under Maxwell disperses; held steady current no
   radiation; oscillating held current Larmor; resistive Taylor decay 2 eta lam^2 / relaxation to Taylor; QED
   spontaneous emission) via a numpy spectral solver; holonomy unwinding measures (toroidal / poloidal flux on lattice
   loops, winding ratio q per flux surface, linkage helicity).


# Entries for 2026-10-24 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-10-24 — read `now` 10-24 (items 1-6 + embed) first

**Frame (Philip 10-24):** no E / B / dipole fields -- only substrate disturbances; deviation from the mean is the
observation. Mesh spacing is more fundamental than structure (a cell-set rate is physics; a 6 / a 9 in L = 1 runs are
different physical structures). The true lattice is undefined (maybe not Kelvin) -> mechanisms lattice-agnostic.
Every neutron is embedded in larger-scale disturbances (Earth, Sun, galaxy) -> empty-box bodies are a limiting case.
MHD is the crudest reference (below hybrid PHARE and full PIC); references are what DST's own dynamics must reproduce.
Memories: feedback_geometric_mechanism_not_em, feedback_mesh_spacing_more_fundamental, project_neutrons_always_embedded,
project_offload_mechanisms_reference_ladder (Landau damping is a required offload channel).

State: step() + Cartan spheromak = content cascades to the smallest loops at ~3.4 turnovers (orientation set by the
coarse mesh, not the finding); no decay, no exit for held content. MHD reference: held + slow self-similar decay.
Hand-off of unmatched loop content into a local store holds a 6 at p 3 but weakens as (ell / a)^6 (a 9 breaks at
~5 turnovers) and the store mostly rings in place; into a travelling pair it barely transfers (slow source = near
field only). Closed spheromaks (and n / n-bar pairs) have no disturbance outside the core -> no interaction without
contact; DST has no independent push.

1. **Embedded spheromak results** (exp/2026/10/24/embed/): DST a 6 P 96 baseline + frac 0.1 x {aligned, opposed,
   tilt45} DONE (see `now` item 7: frac 0.1 does not stop the cascade, tilt45 worst); frac 0.3 x 3 NOT RUN -> rerun (4 parallel x 2 threads, ~20 min):
   `cd exp/2026/10/24/embed && PYTHONPATH=/home/p/git/eq OMP_NUM_THREADS=1 python3 -u dst_embed.py --frac 0.3 --orient {aligned|opposed|tilt45} --threads 2 --save /bk/dst/2026/10/24/embed/dst_0.3_{orient}.npz > dst_0.3_{orient}.txt 2>&1`
   MHD side NOT RUN: `PYTHONPATH=/home/p/git/eq python3 -u mhd_embed.py --frac {0,0.1,0.3} --orient {...} > mhd_{frac}_{orient}.txt`
   (~1-2 min each). Score: does the ambient hold the body past 3.4 turnovers, does content go into the ambient, does
   the body turn / drift. Then the n / n-bar pair inside the same ambient (do they approach).
2. **Tail / depression** (Philip: structure = substrate depression; closed spheromak unobservable from outside):
   slaved-geometry prototype (cell size from local content each tick, set_face_medium n = L / h) -> does a body get a
   tail that falls off with distance; do two tails make n / n-bar approach. L sign question still open with Philip.
3. **Hybrid reference**: PHARE built at ~/git/phare/master -- scope a spheromak (2D / 3D?) as the next rung above MHD.
4. **Size law** of any cell-scale hand-off: a 6 / 9 / 13 (rate vs a / ell), once the outflow route is settled.
5. Carried: 10-22 queue (memory per cell items, continuous AMR), 10-21 items; /bk/dst cleanup; MEMORY.md trim (36 KB).

---
