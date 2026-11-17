# Entries for 2026-09-24 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-09-24 — electron-only falsification battery: every existing electron IC is neutral (class-level); hop-path windings can't be circular (pi_eff→3.75); step()'s static charge field lives in phi not u; periodic flux can't make charge linearly; flux-sourced lam ("drag") confines energy but makes no charge

Read this before continuing. Detailed per-test record:
`exp/2026/09/24/electron_ic_falsify/status.md` (battery plan T0-T6, every
result table). Winding-erosion runs: `exp/2026/09/24/winding_erosion/`.

### Philip's framing this session (working model — supersedes doc/dst.md where they differ)

- **Electrons only**; protons "a while away". Goal: experiments that RULE
  electron ICs OUT. Spheromak is not a control (it SHOULD decay).
- **Electron = "glut"**: excess charge (like static buildup) — the reciprocal
  that fills the proton's deficit moat; the moat belongs to the proton.
  (mem: project_electron_glut_in_moat)
- **doc/dst.md is preliminary**, don't chase its inconsistencies; IC geometry
  is preliminary too → every verdict must say class-level vs geometry-specific.
  (mem: feedback_doc_preliminary)
- The target is a **steady-state equilibrium** — decay is a failure mode. The
  electron may rotate / carry periodic flux, but the rest of the domain should
  have ~zero rate of change (quiet, Saha-style uniform background). So short
  runs measuring du/dt, dphi/dt outside the structure suffice → large domains
  (R100) are affordable. IC should fill the domain (~R60 with BC ~R70).
- A working geometry should show **clues in known dimensionless quantities**
  (alpha = r_e/lambdabar_C, Zitterbewegung 1/2, g≈2.0023) — only if they
  emerge untuned (battery tier T6).
- **phi (flux) is primary**, moving periodically in a bounded system; u is a
  secondary effect of phi travel ("drag"); the electric field around an
  electron is imposed by the electron.
- **Faces are anchors, not paths**: the coarse mesh samples a continuum; the
  true discrete substrate is not computable and only sets limits. Discrete
  face operations inside the continuous model are a failure mode; "8 tiles"
  might survive as a continuum limit. **Any sharp corners are suspect.**
  (mem: project_faces_as_anchors)
- **Model split**: a uniform discrete model (no AMR — "AMR discrete makes no
  sense") for scale-free raw-vacuum tests, and an AMR continuous model for
  structures. (mem: project_split_discrete_continuous_models)

### Results, in order

1. **09-22's "best candidate" fails** (octagon `init_toroidal_winding`,
   `winding_erosion/`). The 09-22 persistence run was t=1.5 (<1% of a mode
   period) and tracked total E, which step() conserves anyway. Energy within
   3 cells of the ring: 0.77 (t=1.5) → 0.50 (t=20). The seeded cos(theta) u
   rings down at grid frequency and radiates; the phi loop is static (≈div-free).
   Open-arc control identical (0.503 vs 0.502) → closure does nothing.
   Uniform lam=1 = pure time rescaling. lam tube around the filament
   (1/(1+(d/a)^2)) only delays radiation (0.511/0.520 at t=20 for a=3/6).
2. **T0 net charge at t=0: all 8 existing dipole/electron ICs are NEUTRAL**
   (double-ring dipole ± vecpot, dipole_mode_ic, curl-full dipole, Hopf pair,
   bare loop, octagon winding, spheromak ref). Uniform/odd-parity u + div-free
   phi → zero shell flux for ANY geometry. Class-level: the entire existing
   gallery is magnetic-dipole content only.
3. **Vacuum checks (R35)**: `outflow` BC drains a uniform background (u 0.1 →
   0.06) — artefact of outflow only; neumann / no-BC hold it exactly. Use
   neumann for background runs. A held u core (fixed_u) makes E_phi grow as t^2
   BIT-IDENTICALLY under all 3 BCs → intrinsic to step(): a static u gradient
   drives dphi/dt = -grad u forever. **step()'s static exterior charge field is
   radial phi ~ 1/r^2 with uniform u**, not u ~ 1/r (doc's diffusion form).
   Caveat (faces-as-anchors): this is a property of the discrete rule.
4. **Hop-path rings aren't circular** (`pi_eff_rings.py`): path length/2R =
   3.31 (square) and 3.61→3.75 for octagon hops 2→16 (+15% → +19.5%,
   converging AWAY from pi), radius ripple ±7%. Staircase paradox → no
   hop-path winding (square, octagon, hop-built helical knot) is pi-exact at
   any size. Class ruled out.
5. **T2 travelling torus** (`t2_travelling_torus.py`, smooth field sampled at
   face midpoints, R0=35, A=20, m=4, R70, neumann):
   - linear: E conserved (1.0000), wave travels, radiates (Enear 0.59-0.71,
     partly reflected back off the wall). **No charge**: shell Q mean ≪ rms,
     sign flips between shells → linear no-go confirmed.
   - drag (EXPERIMENTAL: lam relaxes to kappa*|phi|^2, tau=1): **Enear 0.91-0.94
     — the flux builds its own waveguide**, first construction to hold most of
     its energy. But the m=4 circulation scatters (amplitude 0.30 → 0.03-0.06),
     lam runs away in hotspots (max ~54, no cap), and still **no charge**. A
     |phi|^2 drag can't rectify by symmetry (harmonics 0, 2m × m-harmonic
     divergence never give m=0) — a charge-making drag must depend on flux
     direction / phase / chirality.

6. **Saturated drag + reflection-free linear** (R70 OUTFLOW, status.md table):
   linear Enear 0.44-0.55 (the old 0.59-0.71 was neumann-wall reflection);
   uncapped drag 0.90 but the m=4 circulation decays as fast as linear (it traps
   disorganised energy in lam hotspots); lam_sat=4 keeps the mode 2.5x better
   (amp 0.14 vs 0.06) at Enear 0.82, no runaway; lam_sat=1 weaker (0.64).
   Monotonic decay in every arm, no plateau -> FAIL steady state. No charge in
   any arm (Q mean << rms). |phi|^2-magnitude drag of any saturation form is
   ruled out as a charge mechanism (class-level, symmetry).

7. **Model split** (Philip): AMR C++ moved unchanged into
   `equilibrium::dst::discrete` (kept as reference until Philip decides per
   function; "discrete AMR" makes no sense -- discrete = uniform). New
   `inc/eq/core/data/lattice/continuous/adaptive_lattice.hpp` stubs mirror it
   (NotImplementedError naming the discrete analogue; deviations need written
   justification). Python: `equilibrium.lattice.discrete` / `.continuous`,
   top-level names alias discrete. `test_point_source_poisson.py` split into
   `_uniform` (passes) and `_continuous` (fails at step, intended). C++
   lattice tests pass. (I used `git mv` for the test rename; the index is
   clean again as of session end, nothing staged.)
8. **Work in Python for now** (Philip; AMR continuous "a while away"):
   `equilibrium/substrate/continuous.py` + `tst/substrate/test_continuous.py`
   (9 tests). phi≈A mapping (Philip: "we have been modelling A generally"):
   phi_k = W_k A(face mid).d_k, lattice dt = dt/h, phi seeded dt/2 ahead ->
   step() converges 2nd order (err 7.6e-3 / 1.9e-3 / 4.8e-4 at h=.5/.25/.125)
   to u_t = -c^2 div A, A_t = -grad u, c^2 = 40/3. **Seeders without W_k
   (seed_dipole_ic, curl_full_dipole_ic, today's t2 torus) don't sample a
   single A** (37% error, no convergence; traveling_wave.py has w_k, OK).
   Antisymmetry leaves 7 face DOF/cell: 3 = A, 4 non-vector.
9. **step() = E=0 sector under phi≈A**: u_t = -c^2 div A is the Lorenz
   condition, A_t = -grad u is E=0. Explains frozen curl, E_full≡0, and V2's
   phi ramp in one statement. With AB (A real): holonomy conserved exactly,
   nothing changes a holonomy (no Faraday). E field: "we may need an E field,
   it's an open question" (Philip).
10. **EXPERIMENTAL E field, `continuous.AE`** (numpy): E at face anchors,
   E_t = c^2 curl curl A - J, A_t = -grad u - E. First operator was
   unstable (non-symmetric; ~2.3 e-folds/time at any dt) -- replaced by
   `curl_curl_sym` = C0SQ*antisym(K^T K) with exact adjoint (symmetric,
   exact zero on gradients, div-free output); edge mask removed.
   On the stable operator: transverse waves propagate 2nd order (step()
   freezes them); Poynting E×B balance 10% / 4.5% (h=.5/.25); **Hertzian
   dipole: P/P_theory 0.87-0.93 (r=5-8), sin^2 pattern, pole/eq 0.02-0.04,
   slope -2.4 (theory -2)** -- first classical radiation match in the Poynting
   thread. Induction not settled (loop self-flux drifts; J null-space
   diagnosis). Full detail: `exp/2026/09/24/e_field/status.md` (read the
   CORRECTION section; the first table there is superseded).
11. **Ontology notes (Philip)**: methods that work (DFT, SCF, Hartree-Fock)
   are indicators of ontology, not coincidences -- ask what the substrate
   must be for them to work (mem feedback_methods_indicate_ontology). Geometry
   is primary; waves and wavefunctions are emergent; entanglement = a canal
   formed by pressures ("tension" is a placeholder) from the presence and
   distant interaction of particles. My candidate: pressure shapes the canal,
   a phi holonomy (AB sector) is what it carries (mem
   project_waves_emergent_from_geometry). Spin: likely the angular momentum
   of internal circulation; BCC has no continuous rotation symmetry, and m=4
   about z couples to the lattice's own 4-fold anisotropy (today's T2 m=4 is
   the worst case). Non-uniform density => gradients, unavoidable; under
   step() they always drive phi, with E a static E = -grad u is allowed.

### Bugs / corrections found (mine)

- `_FACE_HOP` is the FULL neighbour offset (verified for all 14 faces), despite
  adaptive_flux_tube.py's "hop to face midpoint" comment. Core seeders already
  use pos + 0.5*_FACE_HOP; my first t2 run used pos + _FACE_HOP → phi not
  antisymmetric → E grew 3x. Fixed. u^2 + 0.5*sum phi^2 IS step()'s conserved
  norm (derived from _step_pass2_uniform, requires phi antisymmetry).
- First t2 used quadrature u/phi → standing wave; in phase travels.
- Long-wave speed from the update: c^2 = 1/2 sum a_k^2 d_kx^2 → c ≈ 3.65
  (outflow's default c_lattice is sqrt(10) ≈ 3.16).
- I retracted "linear ⇒ no geometry can confine": only true for uniform lam.

---


# Entries for 2026-09-24 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-09-24 — electron falsification battery; read `now`'s 2026-09-24 entry and `exp/2026/09/24/electron_ic_falsify/status.md` first

Supersedes the 2026-09-22 queue below (its items 1-3 — helical/bundle/solid
hop-path windings — are ruled out: hop paths can't be circular, pi_eff→3.75;
item 4 answered NEGATIVE; item 5 partly answered: every existing IC is
neutral). Electrons only; protons deferred.

Decision offered to Philip at session end (not yet answered):

1. ~~Saturating flux-sourced lam~~ DONE 09-24 (status.md, now item 6):
   saturation trades confinement for circulation survival (lam_sat=4 best),
   but every arm decays monotonically and none makes charge. Not pursued
   further unless paired with item 2.
2. **Charge mechanism** — |phi|^2 drag can't make charge (symmetry). Candidates:
   drag that depends on flux direction / phase / chirality. Needs Philip's
   input on the physics before building.
3. **Which field carries Coulomb** — step()'s static exterior is radial
   phi ~ 1/r^2 with uniform u; doc's is u ~ 1/r. Asked Philip; he reframed
   (phi primary, u secondary via drag) but didn't pick. Relevant once a
   continuous-model update rule is chosen.

Also queued:
- **Rate test at R100** (Philip's steady-state criterion): per radial bin,
  |du/dt|, |dphi/dt| over a few tens of steps, exterior must be ~0. Neumann
  BC, quiet uniform background (noise=0). Cheap; apply to every candidate.
- **pi_eff for smooth fields**: thin-tube torus (A ≪ R0), in-phase travelling
  wave, measure rotation rate vs c ≈ 3.65 → effective pi; repeat at 2-3
  resolutions (should converge to pi if the continuum sampling is sound).
- **Refinement convergence (T5)** for anything that survives — required, per
  faces-as-anchors; also re-check V2 (phi ramp) and pi_eff across resolutions
  to separate discrete-rule artefacts from continuum behaviour.
- Continuous model: what continuum equation the AMR continuous model should
  discretize — open, Philip's call.
- (Uncommitted status: see the 2026-09-25 section above.)

---
