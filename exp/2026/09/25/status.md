# Entries for 2026-09-25 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-09-25 — induction on AE resolved (magnetisation source, Neumann to 0.15%); C++ continuous model level 0 built (step, sponge, step_ae, numpy-identical, ~170x); 09-24 dipole "classical match" retracted, h=0.25 converges to the form-factor-corrected value; evening: face-anchor medium, 4x less memory, R144 induction converged, curl-grad leak found, ontology notes

1. **Point-source continuous test**: failed by design (continuous::step
   stub). Prerequisites discussed (update rule, Coulomb carrier, E ontology);
   now passes because level-0 continuous::step = discrete rule (plumbing
   check only; docstring updated).
2. **Induction** (`exp/2026/09/25/induction/status.md`): sampled loop J has a
   null(K) part -- static CG on it DIVERGES; `current_from_magnetisation`
   (J = C0SQ antisym(K^T m)) converges. Dynamics: Phi1 steady, Phi2 matches
   static target <0.3%. Reference must be the gaussian-tube continuum value
   (1.0508 x filament M), not the filament: h=0.25 gives 1.0492 (-0.15%),
   Phi1 +1.5% (from +8.9% at h=0.5). Test: TestStaticCurrentBalance.
3. **C++ continuous level 0** (Philip: build it, back-and-forth expected):
   see `next` item 5. E stored per leaf in continuous::AdaptiveLattice
   (shared cluster storage untouched). discrete vs continuous step agree to
   1 ulp (FMA contraction), not bit-for-bit.
4. **Dipole** (`exp/2026/09/25/dipole/status.md`): 09-24's reference omitted
   the gaussian form factor (F^2 = 0.411); h=0.5 flux exceeds source work by
   ~50%. h=0.25: clean dipole field (slope -1.994), far-field A amplitude =
   F to 0.2%, source work +8% (2nd-order convergence).
5. **Coulomb as time-rate disparity?** (discussion, Philip): a scalar
   clock-rate field gives like-attract (spin-0), and gravity's response is
   universal while E-field response is q/m-signed; the charge-signed version
   is the QM phase rate qV/hbar (Weyl 1918 -> gauge; Kaluza-Klein). Proposed
   (not run): deficit vs glut structure in a static sigma gradient.
6. **Depletion = cell size + tick rate** (Philip; `exp/2026/09/25/lensing/`):
   smaller cells + slower ticks with rate ~ size = GR weak field (full
   deflection); time-only = half (explains Eddington's pending 2x vs 4x);
   Nordstrom-like = none. Level-0 test: GR/T delay ratio 2.02-2.06, N = 0;
   absolute delays diffraction-limited (not eikonal). continuous::step now
   takes per-cell L (symmetric pass-2 coefficient); `set_leaf_L` binding.
   AMR deferred (Philip: level 0 only for now).
7. **Two flux sources** (`exp/2026/09/25/phase_interaction/status.md`): DC
   flux-fixed sources on step(): like repel, Coulomb 1/d^2 (0.92-0.96 of
   continuum). Oscillating: retarded classical sign alternation (repel for
   kd < ~2.8), not inverted Bjerknes. Phase alone is not charge: kd >> 1
   averages away -> Coulomb needs a nonzero mean flux divergence. Coupling
   (q.A vs grad u) unresolved without a free structure. Also fixed a
   release -Werror (debug-only counters, [[maybe_unused]]).

### 2026-09-25 evening (Philip: level 0 bulletproof first, AMR later)

8. **Decisions**: E stays a stored DOF until decided otherwise. Charge: the
   topological soliton is preferred (Wheeler = the same idea with the core
   removed). sigma placement: div(sigma A) at the face.
9. **Medium at the face anchors** (Philip: cell.L = coarse size / MEAN size
   of the discrete cells it covers, so L = h/<ell>, varying with depletion at
   fixed h; junction distortion averaged). continuous::step no longer reads
   per-cell L or averages 2/(L_i+L_nb): `set_face_medium(dsig, dn)` takes
   deviations of sigma and n = L/h, sampled from the continuum at the face
   anchor. Lensing unchanged (GR/T 2.017, N 0). Precision: deviation input
   keeps eps in the coefficient, but u/phi still accumulate in float64 (eps
   below ~1e-12 needs base + deviation evolution). mem
   project_sigma_ell_constitutive_not_cell_size.
10. **Memory 2242 -> ~590 B/cell** (C++): reconstruction operators stored
   once per face-presence pattern; no per-face scratch (dA/y/cc computed on
   the fly); discrete scratch freed by the continuous rebuild; step's pass 2
   updates phi in place. The experiment scripts had the same waste (full-
   lattice numpy diagnostics): now row-restricted (`reconstruct_A_at`,
   `curl_A_at`, `phi_rows`/`e_rows`, `drop_caches`), source passed once plus a
   per-step `scale`. Checked identical to the old logs at R36/R40.
   **Runs are named by R/h** (cells across the radius), not physical R.
11. **step_ae energy**: E-B energy bounded (~1% to t=4, no secular growth).
   FOUND: curl_curl_sym is not zero on discrete gradients (the docstring
   claimed exact zero; corrected). Grid-scale u leaks O(1) into B; smooth u
   leaks at 2nd order. Philip: the continuous model needs convergence, not
   exact discrete identities; grid-scale content is under-resolution (the
   leak ratio could be a refinement indicator). Root: u<->phi uses step()'s
   face difference while E/B use the LS reconstruction.
12. **Third resolution point**: induction R144 matches the continuum within
   the reference's own precision (Phi2/M 1.0505 vs 1.0508, Phi1 +0.005%).
   Dipole R160 killed at step 6400/12604 (Philip), queued for rerun. Caveat (Philip): no saved state
   has ever been plotted, so every verdict is scalar-only and could be a
   false positive.
13. **Ontology (Philip)**: no infinities (classical = continuum
   approximation; the substrate is the cutoff); nature is not sharp (no
   kinks even in reconnection; dipole = sharpest curve); no noise, only
   context; noise/chaos/turbulence are reifications; energy tends to diffuse
   and structures (Great Red Spot, protons, electrons) are energy that
   can't: the open question is WHAT blocks it. mems project_nature_not_sharp,
   project_structure_is_blocked_dissipation, feedback_reproduce_dont_reify.

---


# Entries for 2026-09-25 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-09-25 — start here; read `now`'s 2026-09-24 items 7-11 and `exp/2026/09/24/e_field/status.md` (CORRECTION section)

Working in Python (`equilibrium/substrate/continuous.py`) per Philip; AMR
continuous C++ is a while away (stubs exist, `equilibrium::dst::continuous`).

1. ~~Induction on AE~~ DONE 09-25 (`exp/2026/09/25/induction/status.md`):
   J = `continuous.current_from_magnetisation` balances statically (sampled J
   provably doesn't: CG diverges). Dynamic Phi2/M 1.04-1.05, Phi1 steady,
   matches the discrete static target (1.049) <1%; Neumann to ~5% at h=0.5
   (tube width + resolution). h-convergence needs a C++ kernel (item 5).
2. ~~Dipole convergence~~ DONE 09-25 (`exp/2026/09/25/dipole/status.md`):
   09-24 match RETRACTED (point-dipole reference, unresolved h=0.5). h=0.25:
   slope -1.994, flat flux; source work +8% vs form-factor-corrected theory,
   converging 2nd order. Flux measurement itself +7% at h=0.25. Optional third
   point: longer wavelength / wider source.
3. **E-field ontology**: Philip 09-25: E stays a stored DOF until decided
   otherwise. Sources J/rho: where do they come from in DST? E boundary
   treatment still undecided (none now).
4. **Seeders**: fix W_k in seed_dipole_ic / curl_full_dipole_ic / t2 torus,
   or retire them; re-check any result that depended on them.
5. **C++ continuous model, level 0 STARTED 09-25**
   (`inc/eq/core/data/lattice/continuous/adaptive_lattice.hpp`): `step`
   (= discrete level-0 rule, justified), `step_sponge`, EXPERIMENTAL
   `step_ae` + `curl_curl_sym` (numpy-identical to 1e-15, ~170x faster);
   face BCs throw. Python: `continuous.AnchorCluster` / `anchor_cluster`.
   Tests: `tst/substrate/test_continuous_cpp.py`;
   `test_point_source_poisson_continuous` now passes (plumbing, same rule).
   Philip 09-25: AMR later -- make level 0 bulletproof first
   (prolong/restrict deferred). DONE 09-25: medium at face anchors
   (`set_face_medium`, L = h/<ell> = coarse / mean discrete size, not per-cell
   L), lensing unchanged (GR/T 2.017). Memory 2242 -> ~590 B/cell
   (operators per face pattern, no per-face scratch); scripts' diagnostics
   row-restricted. Runs named by R/h. Level-0 checklist: (1) h=0.125 third
   point RUNNING 09-25 evening: induction R144, then dipole R160 (logs in
   exp/2026/09/25/{induction,dipole}/*R1[46][04]*); (2) step_ae energy DONE:
   bounded (TestStepAEEnergy); FOUND: curl_curl_sym is NOT zero on discrete
   gradients -- grid-scale u leaks O(1) into B, smooth u at 2nd order
   (TestCurlCurlOfGradient; docstring corrected). Philip: continuous model
   needs convergence, not exact discrete identities -- grid-scale u is
   under-resolution (possible refinement signal). Open: u<->phi still uses
   step()'s face difference while E/B use the LS reconstruction; one
   reconstruction for both would make them consistent by construction;
   (3) E boundary (sponge, PML later); (4) sigma in step_ae as div(sigma A)
   at the face (Philip ok'd); (5) face BCs: implement or document excluded.
   Next: E boundary; lam with E (sigma placement, Philip's call -- still open:
   sigma div A vs div(sigma A)). Old mod.cpp notes: n_steps loop in C++,
   uniform model has no outflow BC.
5b. **Lensing** (`exp/2026/09/25/lensing/status.md`): factor 2 confirmed as
   kernel behaviour; open: derive sigma = ell (rate ~ size) and ell(depletion)
   from the substrate rule; optional ray-optics numerics check at a >~ 20.
   Also untested: Coulomb-as-time-rate deficit/glut drift test.
5c. **Charge** (09-25 phase_interaction): mean flux divergence required;
   the "how" is open (physics has none). Candidates: topological (Wheeler
   charge-without-charge, winding) -- ties to the flagged soliton idea.
6. **Spin**: derive the energy current from step() (or use S = c^2 E x B on
   AE) and compute L_z of circulating states; m=3 vs m=4 (or tilted torus) to
   separate lattice-anisotropy scattering from real decay.
7. **SCF / variational steady-state search** for the electron instead of
   time evolution (mem project_scf_steady_state_method_idea).
8. **Bell canal stage 2** reframed: build the pressure field of two bodies
   first, then seed a phi holonomy along the canal it forms.
- Minor: `lattice.py` _KELVIN_VOLUME = 2.177 but the BCC d1=1 cell volume is
  0.770 (edge d1/sqrt6); only used in step_lam_sink cost.
- Uncommitted: 09-25 (continuous C++ level 0, lattice.cpp bindings,
  continuous.py additions, tests, exp/2026/09/25) on top of all of 09-24 evening (C++ namespace move + continuous stubs,
  lattice.cpp bindings, continuous.py, tests, exp/2026/09/24/e_field, now,
  next, doc/mems). Nothing staged.

---
