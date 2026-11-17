# Entries for 2026-09-30 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-09-30 evening — bath seeds: the pressure hint is reference bias; proper time: plateaus = frozen circulation, not slow clocks; held uniform field: dipole unpushable by symmetry, gauss offset not acceleration, symmetric states unstable

1. **Bath seeds** (`exp/2026/09/30/ambient_flux_lam/status.md` "Bath seeds", `analyze_seeds.py`, `seeds_T0.01.png`;
   run.py now records core/total energies of domain and bath + core sigma, `--seed_tag`, `--tag`): alpha 100 T 0.01,
   4 seeds: gauss +0.030 +- 0.002, dipole +0.058 +- 0.005, flat from ~4 to 20 crossings. NULL (gauss amp 0.1, 1%
   energy, no well): +0.026 +- 0.002, same core lam offset (+0.14) -> the gauss excess is the metric: the decorrelated
   domain drifts from the closed bath copy (sponge vs closed walls). Dipole ~0.03 above null; alpha 300 (2 seeds):
   <= 0.01 / negative (own null not run). Bath scoring needs a same-alpha null ensemble. CONDITIONAL.
2. **Proper time** (`exp/2026/09/30/electron_survival/status.md` "Proper-time", `proper_time.py/png`, saved records):
   no collapse of Ecore/E0 vs tau across g; long-run plateaus flat in proper time (dipole g 36: 0.166 over ~85 proper
   crossings) = zero-frequency frozen circulation (1e), not a slowed clock.
3. **Held uniform field** (`exp/2026/09/30/held_field/status.md`; `uniform_field` in ambient_bc.py, shares
   `_harmonic_field` with proton_microfield, +2 tests, 7/7): (u, phi) -> -(u, phi) invariance x pi rotation pins a
   z-dipole's centroid in any uniform E (no push, as classically). Gauss g 100: z_c/E collapses (linear) to ~3-4
   crossings, then E = 0.01 holds a bounded +0.2-0.3 cell offset, no acceleration; dipole polarisation noise-dominated.
   E = 0 controls: symmetry-forbidden components grow from roundoff (dipole e-fold ~1.5 crossings, 1e-17 -> 1e-2 by 60).
4. **Philip**: "real expectations" are not known yet -- descriptive vs mechanical models; we look for means to produce
   descriptive expectations, no rule for it. The L1 "real-expectation track" was renamed: everything is conditional
   (mem feedback_two_kinds_of_survival_test updated).
5. **Philip, far background (next item 5)**: depends on the vicinity -- a basically empty solar-system point: far field
   dominates, lam uniform over the region one electron occupies; active plasma / ion cloud: different gradients on
   different sides of the electron (mem project_far_background_context_dependent). Consequence (conditional,
   self-sourced rule): uniform lam_bg = exact time relabelling, so in empty space the background cannot hold a free
   electron -- persistence must be intrinsic. Scale (EM-scale r_e/r assumed): side-to-side asymmetry ~ (r_e/d)^2 =
   1e-25 at solar-wind spacing, alpha^4 ~ 3e-9 at d = a0 (r_e/a0 = alpha^2): only dense matter is non-uniform.
6. Full suite 469/469 (462 + AmbientBC 5 + uniform_field 2). Nothing committed. Session moved on to spinors.

---

## SESSION 2026-09-30 — electron survival 1a/1b: only g = alpha amp^2 matters, uniform floor = exact relabelling, Euler step has O(dt) energy error with state-dependent sigma (implicit midpoint added), no survivor; AmbientBC (bath + held proton field) built and tested; potential (1/r) lam in a bath: no proper-time survivor

1. **1a/1b** (`exp/2026/09/30/electron_survival/status.md`, `self_lam_long.py`): (alpha, amp) -> (alpha/s^2, s amp)
   exact, only g = alpha amp^2 matters. A uniform additive floor is an exact relabelling, (eps, alpha, dt) ==
   (0, alpha/(1+eps), dt/(1+eps)) (`floor_equivalence.py`, 3e-16); L1 near-Earth lam ~5.5e-7 (galaxy 5.4e-7, Sun
   9.9e-9, Earth+Moon 1.5e-11) = the void run. Symplectic Euler is not symplectic once sigma depends on the state:
   H error 4.8e-2 / 2.3e-2 / 1.2e-2 at dt, dt/2, dt/4; implicit midpoint (sigma at the midpoint) conserves H to
   1e-13 in 5-7 iterations (`--integrator midpoint`). 1a (200 crossings): leakage slows but continues; g = 100 late
   retention 0.2-0.3, not dt-converged even with exact H (trajectories separate after t ~ 500: sensitive dynamics);
   the survivor is 86-98% frozen divergence-free phi = manufactured circulation (1e: not survival). 1b (g 2..200,
   widths 1/1.5/2, Euler dt, biased ~10-20%): no threshold or preferred size; retention tracks core sigma (clock
   slowdown). Verdict CONDITIONAL (cold vacuum, absorbing sponge, void lam).
2. **Philip's physics calls (mems)**: free-electron formation context open -> low lam, NOT void, first; the proton's
   Coulomb field extent may be a prerequisite for electrons (formation vs persistence); H2 default over H1 = p/e
   mismatch, net neutrality ~1e-21 e -> mismatch in field shape/extent; decay may mean the BACKGROUND is wrong
   (inward focusing pressure); modelling is progressive "like a partial derivative" (hold things fixed); two kinds
   of test: survives real expectations (goal) vs survives these conditions (stepping stone); lam = proxy for
   substrate depletion from everything local; far background uniform = an assumption; lam follows the potential
   (1/r, Newtonian), not the field (1/r^2). Near-Earth L1 split: shadow (tail lobes, CMB only) vs sunlit (solar
   wind ~5 cm^-3, 400 km/s, IMF 5 nT; sunlight ~1 kV/m = 6e-18 of the field at r_e, uniform/quasi-static at electron
   scale -> held uniform field). Idealisations (black body, Jeans cube/swindle) are signposts, not goals.
   mems: project_free_electron_context_low_lam, project_h2_charge_mismatch, project_decay_may_be_wrong_background,
   project_lam_background_plus_electron, feedback_two_kinds_of_survival_test (+ pinning, reproduce_dont_reify).
3. **AmbientBC** (`equilibrium/substrate/ambient_bc.py`, `tst/substrate/test_ambient_bc.py` 5/5): sponge relaxes
   toward bath(t) + static_phi instead of zero; bath = shallow copy of the lattice (own u/phi/lam/ell; deepcopy
   recurses through Cell objects), same wave step; `proton_microfield` = point charges outside the domain, exactly
   discrete-harmonic inside; its boundary-cell divergence (external flux through missing faces) is restored on u
   BEFORE the step (after the step leaked 2% via symplectic ordering) -> held to 6e-16. Linear step(): domain and
   bath superpose exactly -> ambient cannot act on a structure; only a nonlinear rule couples them.
4. **Potential (1/r) lam in a bath** (`exp/2026/09/30/ambient_flux_lam/status.md`, `run.py --source potential`,
   CONDITIONAL): 1 + lam = (1+lam_far)(1+lam_amb+alpha psi), psi = lattice Poisson (splu, 1.3 ms) of energy density
   minus the bath mean (Jeans swindle), midpoint, R = 12, 20 crossings. T = ambient pressure knob (lattice units),
   not a temperature (Planck: nothing at fm scale physically). Bath sub-mean dips make depletion < 0 unless a held
   lam_amb >~ alpha T (Poisson can't set it) -> alpha_eff = alpha/(1+lam_amb). Cold: alpha 100 gone; gauss alpha 1000
   keeps 0.88 but core lam 28 (sigma 0.036, < 1 proper crossing) = frozen-clock / black-hole-like limit; dipole
   alpha 1000 leaks (0.91 -> 0.34). Raw near-Earth = cold to 5 digits. T = 0.01 lowers strong-case retention
   (0.88 -> 0.39, 0.34 -> 0.15); alpha 100 keeps 0.05-0.10 vs 0 cold (1-2 noise units); T = 0.1 inside noise.
   Domain - bath departure is invalid for T > 0 (bath decorrelates under the nonlinear rule): scored as core energy
   excess over the bath. A --source flux (1/r^2) variant exists, not run (superseded by the 1/r clarification).
5. Process slip: five runs launched with `&` (against feedback_background_jobs); a harness-tracked PID waiter was
   added instead of restarting. Nothing committed (Philip commits).

---


# Entries for 2026-09-30 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## (superseded 09-30 evening) QUEUED 2026-09-30 — start here; read `now`'s 2026-09-29 session

Philip commits/pushes himself (09-29 evening) to test on CI. Suite: 462/462 locally with
`PYTHONPATH=$PWD:$PWD/tpp/phlop EQ_TEST_N_THREADS=1 python3 -m phlop.run.test_cases -i tst -c 6`
(phlop PR #56 merged, tpp/phlop = 74a1050 / 0.0.42). phlop dropped from requirements.txt (Philip):
mkn clones it into tpp/phlop (mkn.yaml sub) and sh/test.sh now puts $PWD/tpp/phlop on PYTHONPATH, so
CI uses master (whole-file exec-yaml key) without a pip release. Its only runtime dep here is pyyaml
(dill is for --dump only).

1. 09-30 a/b DONE (`exp/2026/09/30/electron_survival/status.md`): only g = alpha amp^2 matters; uniform additive
   floor = exact relabelling (L1 lam ~5.5e-7 = void run); Euler step has O(dt) energy error with state-dependent
   sigma -> implicit midpoint (exact H) added; late retention 0.2-0.3 at g = 100, not dt-converged (sensitive
   dynamics), survivor >= 86% frozen circulation; no threshold / preferred state. NEXT: thermal ambient
   background (Philip: decay may mean the background is wrong -- inward focusing pressure), T scan, midpoint;
   C++ midpoint if swept. Scope: formation may need the proton's Coulomb extent; H2 default -> p/e mismatch in
   field shape/extent (neutrality ~1e-21 e).
   09-30 later: `equilibrium/substrate/ambient_bc.py` (AmbientBC: sponge -> evolving bath + held static field;
   proton_microfield; tst/substrate/test_ambient_bc.py 5/5). Potential (1/r, Newtonian-style) lam in a bath
   (`exp/2026/09/30/ambient_flux_lam/status.md`, conditional): no proper-time survivor cold or pressurised; only the
   frozen-clock limit (alpha 1000). Weak-coupling pressure hint at noise level -> seeds. Real-expectation track:
   cold + held uniform fields (sunlit/shadow L1, solar wind vs tail lobes); uniform B needs AE.
   **Electron survival under 2(a)** (`exp/2026/09/29/electron_survival/self_lam.py`, status.md):
   a. long runs of the strong cases (gauss / dipole alpha=4 amp=3): does the time-dilation-trapped lump
      keep leaking at a rate ~ sigma_core, or stop?
   b. fine amplitude scan (alpha amp^2 from ~4 to ~40): threshold or preferred state (one e/m)?
   c. push test: translates rigidly under a force (selection criterion 2)?
   d. eps > 0 floor (Philip 09-27: a candidate that decays at the floor is out).
   e. frozen-circulation bookkeeping: always split core energy u / grad / sol; manufactured frozen phi
      is not survival.
   Selection criteria (09-27 block below): persists without pinning; translates rigidly; one e/m;
   moment aligns/precesses (g/2); radiates when accelerated. Plus 09-29 targets: dielectric index
   ((n-1)/(n+1))^2 reflection, c/n) as a collective response; sheath via inertia ratio; Debye
   length ~ sqrt(eps T/n); real Saha (spread ~ 1/sqrt(mT), bound/free ~ exp(-chi/kT) T^(3/2)).
2. **Charge needs a carrier**: sigma/lam coupling in step_ae (div(sigma A) at the face, consistent with
   the relational sigma now in step()); then (a) moving lam structure -> J -> div E, (b) div E never
   appears/vanishes without a structure (Gauss). Philip: charge = proton/electron interface ->
   divergence should localise at the deficit/glut interface; free electron = probe (Saha).
3. **BC leftovers**: BC arrays not remapped after AMR, not persisted; fn BCs still inert on refined
   clusters; PML worth keeping (best absorber at R150) -- re-measure with face-symmetric decay;
   directional sponge unproven. Static far field: Dirichlet (now working) for static tests, sponge
   for waves -- they conflict for phi-damping sponges.
4. **Kernels**: step_lam_sink one-sided (breaks antisymmetry by design); step_cross v1 dissipative;
   Python v1 lacks C++'s sigma^2. step_lam: sigma*lam per cell -> sum lam drifts slightly.
5. **Carried over**: casimir_polder integrates half the BCC zone and claims 1/r^3 (real CP 1/r^6-7);
   continuous-model level-0 items (09-28 item 3 below: [111] iso dipole, DEC packet sweep, E boundary);
   nanobind capsule deleter mismatch; R160 'dec' partial snapshots deletable; PHARE parked (flip B_phi).
   Philip's open physics calls: free-electron interface (vacuum floor?); dielectric mechanism
   (coefficient vs emergent, don't assume lam).

---
