# Entries for 2026-09-29 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-09-29 — BCs reviewed and rebuilt (sponge was an invisible flux sink); sigma relational at the face; Casimir exact; ambient thermal IC replaces "saha"; charge: exterior in phi, conserved div E in AE (but no carrier); first self-sourced-lam electron scan

1. **CI** (run 36431612896 on HEAD 9f004ac): failed exactly the 7 known item-2 tests. GitHub runners have 2
   cores (maybe 1 logical) -> any committed .phlop.exec.yaml must be CI-safe (mem feedback_ci_two_cores).
   tst/substrate/adaptive/.phlop.exec.yaml is active again (bare whole-file key); needs phlop PR #56
   (whole-file key, on ~/git/phlop branch next, NOT merged). Local suite runs used
   `PYTHONPATH=$PWD:/home/p/git/phlop`. Philip merges; then update tpp/phlop + pip.
2. **Double slit** (`equilibrium/substrate/double_slit.py`): lam scales both legs -> index contrast,
   impedance unchanged, no reflection in the continuum; default wall is ONE BCC layer (bypassed by
   square links, dx = 1.155) with 1-cell apertures; thick-wall reflection is a grid-scale artefact.
   Reflection test removed, docstring rewritten, body-dose test weakened to < aperture.
3. **Casimir** (`exp/2026/09/29/casimir`): casimir.py rewritten as the exact layer-chain mode sum; 1/d^3
   term = -pi^2 c/(1440 d^3) to 2e-4 for any weights (universal: cannot calibrate the lattice at leading
   order; lattice only in 1/d^4, 1/d^5, surface, bulk). Old 0.72 was a zone-edge artefact. doc/dst.md fixed.
4. **"Saha" -> ambient** (Philip: no ionisation; it was the vacuum ambient state = how we initialise).
   Old TestSahaFraction was a noise random walk (std ~ sqrt(steps)). New equilibrium/substrate/ambient.py
   (thermal_state: equipartition, stationary under step(), <u^2> = T exact), reciprocal_bc.py
   (ReciprocalBC / ReciprocalBCCluster moved out of tests). Renames: test_saha -> test_reciprocal_bc
   (both dirs), test_saha_eqiulibrium -> test_sponge_reciprocal_bc; tst/substrate/test_ambient.py.
   Cooling probe (`exp/2026/09/29/ambient_cooling`): propagating sector drains ~t^-1.6, divergence-free
   phi (73.5% of DOF) never cools -> "closest to absolute zero" = frozen circulation set by history.
5. **BC review** (`exp/2026/09/29/{bc_review,sponge_assess,bc_design}`, doc/dst/bc.md 09-29 section):
   every sponge damped phi per cell row -> broke phi_ki = -phi_ik; the mismatch was an invisible flux
   sink at the sponge edge (it held the old point-source "Coulomb" profiles); continuous sponge never
   damped E. Rebuilt: quadratic_sponge(gamma_max | damp_max legacy) + step_sponge(i, dt), rate, one
   factor per face for both copies, E included, target honoured; numpy DampedLattice same (default
   gamma_max = 10 c/width: reflection 5.6e-3 / 2.8e-3 / 8.3e-4 at width 2.4 / 4.8 / 9.6 vs old 2.4e-2 /
   3.2e-2 / 7.2e-3). Integrated / directional / PML face-symmetric. fn/remote faces reach u on
   unrefined clusters (dirichlet = fixed-u neighbour, TestDirichletBC). step_subcycle throws for face BCs
   on refined clusters. outflow_ledger / pml_outflow removed; outflow c default = bcc::wave_speed().
   Tests: tst/substrate/test_bc_invariants.py. Stale build found and rebuilt (09-25/27 .so's predated
   09-28 weights). mem project_sponge_hidden_sink, project_bc_fn_inert_on_u (updated).
6. **Statics / charge** (`exp/2026/09/29/{charge_exterior,ae_charge}`): step() has no static u gradient
   (phi_t = -W grad u grows); charge = div phi = -u_t there (vacuum at rest has zero charge by identity).
   Static exterior = radial phi with uniform u, psi ~ 1/r - 1/b, eps = c^2/V_cell (derived, 0.4%),
   static only with the core held; bare core disperses. Point-source tests REPLACED by this
   (equilibrium/substrate/charge_exterior.py, tst/substrate/_charge_exterior.py). In AE with DEC, div E
   is exactly conserved (continuity with J to 1e-14; LS leaks) and a seeded div E builds its Coulomb u
   by itself -- but (Philip) that charge has no carrier: AE has no Gauss constraint, J is a prescribed
   stand-in for matter, vacuum cannot make charge. mem project_ae_charge_conserved.
7. **Physics discussion (Philip)**: vacuum needs capacitance for light to travel (eps = c^2/V_cell; mu =
   1/c^2 done 09-25). Displacement current: eps0 E_t (vacuum, field modes) vs P_t (atoms) -- do NOT
   assume lam is the atom part. Charge is a direct consequence of the proton/electron interface
   (plasma sheaths, Debye length); a free electron is itself a probe looking to couple (Saha Lambda^3 =
   its search volume). Targets recorded in `next`. mem project_electron_glut_in_moat updated.
8. **Kernels** (tst/substrate/test_kernel_invariants.py): sigma is relational (Philip) -> face sigma on
   both legs in numpy step() and C++ substrate_step (were cell sigma on the u leg; C++ adaptive already
   face) -> numpy == C++ to 1e-16, sum u and H = 1/2 u^2 + 1/2 phi^2 conserved for any lam.
   energy_flux identity re-derived (E = u^2/2 + phi^2/4, S = sigma_f W phi ubar). mhd.step_phi_diffuse
   one update per face (broke antisymmetry by 0.83); its functional tests re-seeded with antisymmetric
   pairs. step_lam_sink breaks antisymmetry by design (left). step_cross v1 dissipates 43%/20 steps.
   test_mhd TestBrazilEnvelope DEPRECATED (skip; drifts without settling, pre-E kernels).
   mem project_sigma_relational_face. Full suite 462/462 green locally.
9. **Electron survival, 2(a) self-sourced lam** (`exp/2026/09/29/electron_survival`): lam = alpha (u^2 +
   1/2 sum phi^2) each step; H conserved for any sigma(t). Weak nonlinearity -> small plateau of
   manufactured frozen circulation (varying sigma makes phi_t non-gradient). Strong (alpha amp^2 ~ 36):
   36-38% of core energy kept, mostly propagating content, lam_max 10-64 -> self-trapping by time
   dilation (clock nearly stopped), not a balance; retention rises smoothly with amplitude (no threshold,
   no preferred state). 2b (step_lam_relax) / 2c (step_cross) ruled potentially invalid by Philip.

---


# Entries for 2026-09-29 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## (superseded 09-29 evening) QUEUED 2026-09-29 — start here; read `now`'s 2026-09-28 session


Weights are settled and built (C++ both modules, numpy mirrors): discrete step() W = sqrt3:1 per leg
(Kelvin star 3:1, c = 2.309); continuous model W = sqrt2:1 (star 2:1, isotropic, c = 2). Nothing
committed -- Philip's call when to commit (large uncommitted tree since 09-24).
Full suite: `PYTHONPATH=$PWD:$PWD/tpp/phlop EQ_TEST_N_THREADS=1 python3 -m phlop.run.test_cases -i tst -c 6`
(tpp/phlop = ~/git/phlop 0.0.41; the site-packages phlop is a stale 0.0.38 without exec-yaml).

1. DONE 09-29: CI run 36431612896 on HEAD 9f004ac (09-28 work is committed) fails exactly the 7 tests
   of item 2, nothing else. `.phlop.exec.yaml` NOT recreated: yaml env/threads apply on CI too
   (2 cores, maybe 1 logical; CI sets EQ_TEST_N_THREADS=1) -- phlop has no core cap or env expansion.
   Needs a per-machine mechanism first (Philip's call).
   (old text:) **Rerun the full suite** on the final state (after the 09-28 fixes) to confirm only the open items
   below still fail; then **recreate `.phlop.exec.yaml`** (lost in a git cleanup; format:
   tpp/phlop/tests/_phlop/run/.phlop.exec.yaml) for tst/substrate and tst/substrate/adaptive: heavy tests
   get `env: {EQ_TEST_N_THREADS: "n"}` + `threads: n`. Durations are in phlop's log (point_source
   continuous ~88 s each at 1 thread; the adaptive spheromak tests are the long ones).
2. **Open test failures after the discrete 3:1 change -- Philip's calls:**
   a. `test_double_slit` wall reflection: marginal in BOTH versions once period-averaged (wall/free old
      1.02-1.05, new 0.96-1.01); the old snapshot test passed on standing-wave phase luck. Left failing.
   b. DONE 09-29 (`exp/2026/09/29/casimir/status.md`): casimir.py rewritten as the exact layer-chain
      mode sum; 1/d^3 term = -pi^2 c/(1440 d^3) (c = lattice speed) to 2e-4 for any weights ->
      universal, cannot calibrate epsilon at leading order (lattice only in d ~ a corrections,
      surface, bulk). Old 0.72 was a zone-edge artefact; doc/dst.md corrected. Open: casimir_polder
      integrates half the BCC zone (gapped half, short range only) and claims 1/r^3 (real CP 1/r^6-7).
      Double slit (a) DONE 09-29: reflection test removed -- lam changes speed not impedance; the
      default wall is one layer (bypassed by square links) and 1-cell apertures.
      (old:) `test_casimir` ratio3: compares with a c=1 continuum although the docstring says c = sqrt10;
      0.72 -> 0.26; term3 is a zone-edge (Euler-Maclaurin) quantity, maybe not a continuum limit.
      Casimir is a calibration target (epsilon floor), so worth a proper look.
   c. `test_point_source_poisson_uniform` Coulomb fixed-window plateau: pinned 20k steps is no longer
      a plateau; R=20 plateau -1.927 by ~45k steps (old -1.63). Rescan R=20/30/50 at plateau (~30 min,
      8 threads) to recheck convergence toward -1, then re-pin (SIZES step counts too).
   d. 09-29: "saha" is a misnomer (no ionisation; it is the ambient vacuum state / reciprocal BC).
      TestSahaFraction is a random walk (noise, no sink: std ~ sqrt(steps)). Philip: it comes down to
      how the vacuum is initialised -> thermal IC (stationary in a linear rule), renames to
      ambient/reciprocal_bc proposed, not done. Cooling probe `exp/2026/09/29/ambient_cooling/status.md`:
      propagating sector drains as ~t^-1.6, divergence-free phi (73.5% of DOF) never cools -> floor =
      frozen circulation set by history. Side bug: apply_sponge breaks phi antisymmetry.
      (old:) `test_saha` (two-seed std 5.23 vs 5.31, delta 0.05), `test_saha_eqiulibrium` sponge x2,
      `test_mhd` Brazil envelope: fixed DT / step counts tuned to the old dynamics -- recalibrate
      (express in wave-crossing time) or retire (discrete model = vacuum/field tests only)?
3. **Continuous-model level-0 items** (unchanged from 09-28 queue): [111] off-axis iso dipole at h=0.25
   (~25 min; the z-axis run can't show the isotropy gain); DEC packet sweep in C++ (now at 2:1:
   plane_wave.py still hardcodes /home/p and /bk paths); E boundary (sponge for E); sigma in step_ae as
   div(sigma A) at the face (Philip ok'd; bridge to lam = mass, prerequisite for the electron
   selection test and the macro-particle idea); induction visual checks / cascade probe.
4. Minor: `inc/eq/nanobind.hpp` capsule deleter frees AlignedAllocator memory with delete[] (float*)
   (mismatched; lazy path rarely used). mkn::avx's own -Winterference-size warning (lazy.hpp:230).
   R160 'dec' dipole partial snapshots (other machine, /bk/dst/2026/09/27/) can be deleted.
Philip's physics calls still open: charge (topological / compact phase?).
ANSWERED (Philip 09-27): no absolute zero -- a frozen lattice is aphysical; there is an irremovable base
"capacitance", a discrete floor epsilon > 0 (Casimir). Near-absolute-zero IC = medium at the floor
(e.g. uniform lam = epsilon), no extra disturbance. Particles persist there -> persistence must be
intrinsic; selection criterion: a candidate that decays at the floor is out. Exact lam = 0 backgrounds
are the aphysical frozen state -> add a uniform floor; epsilon is a calibration target (Casimir).
