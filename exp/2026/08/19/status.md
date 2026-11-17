# Entries for 2026-08-19 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## PREVIOUS (2026-08-19 EOD, compressed): DST has NO validated recipe for a 3D Poynting/energy-flux vector — this is the real open problem, not a "pick the right formula" question

Full blow-by-blow in [[project_poynting_dipole_calibration]] and
[[project_curl_full_dipole_spheromak]] memories + the exp/ reports
they cite; this is the reconciled summary.

### SOLID — keep, cite with confidence

- `m0` root cause (Poynting-ring far-zone diagnosis) is real: `m0=1e-4`
  in `probe_poynting_dipole_compare.py` was picked, not derived, and
  self-depletes the measured region — no `m0` value fixes it.
- `epsilon0=1` is definitional (from `doc/dst.md`'s own `Q ≡
  Σ A_ij·∂u/∂r` charge definition, no separate scaling constant);
  `mu0=1/(epsilon0*c^2)=0.1` in DST's native cell units, derived from
  the already-validated `c^2=10` dispersion relation
  (`test_maxwell.py`). Both are Gaussian/Heaviside-Lorentz-style (no
  `4pi`) — compare against Gaussian-unit classical formulas only, not
  SI-rationalized textbook ones.
- Axial lattice spacing is `2/sqrt(3)≈1.1547`, NOT 1 — confirmed
  directly from `lat.pos`. Any "wavelength = N cells" construction
  must multiply by this factor.
- **Traveling-wave reproducer WORKS, cleanly.** Derived the exact
  companion `phi` for a one-directional plane wave `u=U·cos(K·x-ωt)`
  from continuous-time linearization of the update rule:
  `Φ_k = (2σw_k/ω)·sin(0.5·K·offset_k)·U`. Confirmed by direct
  shift-match: correlation **0.97-0.99** for both `u` and `phi`,
  robust `HALF_LEN=20` to `60`, wavelength sweep 2-20 cells/wavelength
  with no breakdown at any tested scale. Answers "is DST missing a
  basic photon-propagation capability" — no, it reproduces the
  classical one-directional shift with high fidelity once measured
  correctly. Visual:
  `exp/2026/08/19/18_30_traveling_wave_reproducer/traveling_wave_ic_vs_evolved.png`.
- `E_full := -∇u - ∂φ/∂t` is IDENTICALLY zero under plain `step()`,
  verified directly at machine precision (`rms~1e-18`) for a general
  (non-plane-wave) field; the `-∇u` and `-∂φ/∂t` terms are each real
  and nonzero (`rms~0.085`) and exactly cancel. A genuine algebraic
  identity of the update rule — means the "u=scalar potential,
  phi=vector potential" interpretation makes E trivially zero always
  under free dynamics. Useful diagnostic fact, not itself a flux
  recipe.

### Known-broken conventions and failed attempts — see failure files

- doc/dst/failures/incompatible-em-field-conventions.txt — three
  different E/B conventions used across sessions contradict each
  other; open question whether the "drop step_cross" Poynting-ring
  fix was real or an artifact.
- doc/dst/failures/poynting-local-theorem-check-fails.txt — local
  Poynting-theorem check failed (correlation -0.01); likely a
  naive-finite-difference-divergence bug, not a physics falsification.
- doc/dst/failures/hertzian-dipole-wrong-angular-ratio.txt — Hertzian
  dipole equator/pole ratio pinned at ~0.3, wrong direction, under
  both continuous and lattice-native sources; several explanations
  ruled out.
- doc/dst/failures/grad-sigma-cross-grad-u-not-magnetism-mechanism.txt
  — the `∇σ×∇u` "leak" formula is deprecated as a magnetism mechanism
  (category error), though the narrow algebra (can a lam-gradient
  generate net circulation from none) still stands.

### Fast regression tests added (per Philip: "we might need more super short pass/ok tests too")

- `tst/substrate/test_maxwell.py::TestFullPotentialFieldIsZero` — the
  `E_full≡0` identity, machine precision, ~0.1s.
- `tst/substrate/test_cylinder_lattice.py::TestTravelingWaveReproducer`
  — traveling-wave shift-match at minimal validated config
  (`R=12,HALF_LEN=20,wavelength=8`), correlation >0.95 threshold, ~5s.
Both pass, no regressions (10 + 9 tests total). Add more of these as
findings get nailed down.

### Concrete plan / open next steps

1. Derive the actual local energy-flux identity from first principles,
   the same rigorous symbolic way as the traveling-wave companion-phi
   and `E_full≡0` were derived. Start from `H_i = 0.5u_i² +
   0.5Σφ_ik²` (already validated conserved quantity) and find the flux
   `J` satisfying `dH_i/dt = (discrete divergence of J)_i` EXACTLY —
   likely needs proper staggered/Yee-grid-style bookkeeping
   (energy-at-cells, flux-at-faces), not naive 3-axis finite
   difference.
2. Once that flux is found and validated (exact identity), rebuild the
   Hertzian dipole and Poynting-ring far-field checks on top of it —
   not on any of the three known-broken conventions.
3. Snell's law test (`σ₂sinθ₁=σ₁sinθ₂`, from `σ` multiplying both
   `du/dt` and `dφ/dt` symmetrically → `c∝σ`) is independent of the
   E/B-convention crisis — uses only `u` and `σ`. Not yet built, can
   proceed any time.
4. `equilibrium/substrate/loop_antenna.py` (finite constant-current
   loop far-field, `ka≈1.04`) remains a valid reference SHAPE
   function, useful once a validated `S` exists — pure angular
   correction, was never meant to fix a radial-slope/ratio anomaly by
   itself.
