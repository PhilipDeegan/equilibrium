# Candidate in a held uniform field — 2026-09-30 (CONDITIONAL; next 10-01 item 3)

**INVALID as structure evidence (Philip 10-01): lump width 1.5 cells. Still valid: the exact symmetry argument ((u, phi) -> -(u, phi) x pi rotation pins a symmetric dipole's centroid in a uniform field); uniform_field helper and tests. An electron below R20 (radius 20 cells) is totally invalid, R20 maybe too small (mem feedback_structure_resolution). Survival / retention / response numbers below are records of the setup only.**

`run.py`: self-sourced lam rule of `../electron_survival` (lam = alpha (u^2 + 1/2 sum phi^2) of the TOTAL field,
candidate + held), g = alpha = 100, width 1.5, implicit midpoint, cold vacuum, R = 12, 60 crossings,
AmbientBC(T = 0, static_phi = uniform_field(E)) (new helper in equilibrium/substrate/ambient_bc.py, tests 7/7).
Gauss with E along z (push), z-dipole with E along x (torque / polarisation), E = 0, 0.001, 0.003, 0.01 (lattice
units). Plot `held_field.png` (`plot.py`). End states in /bk/dst/2026/09/30/held_field/.
Real fields are ~1e-17 of the candidate's field: at that level any response is linear and invisible in float64, so only
the response per unit E is readable. (Also, as Philip notes 09-30, "real expectations" are not known yet: this is a
chosen condition, like the rest.)

## Symmetry (exact, all orders)
The rule is invariant under (u, phi) -> -(u, phi) (lam quadratic). A z-dipole (u ~ z g) in any uniform E in the x-z
plane is invariant under (pi rotation about y) x (sign flip): its energy centroid cannot move. Seen: c = 0 to 1e-4.
So a neutral dipole can only be torqued / polarised by a uniform field, never pushed (as classically). Only the gauss
lump (monopole-like) can be pushed; the push is odd in E (z-reflection maps E -> -E).
With a linear step() the held field would superpose and exert nothing (AmbientBC docstring); any push here comes
through the nonlinear lam cross term 2 alpha phi_c . phi_E (a refraction / lensing push, not a linear Coulomb force).

## Results
- Gauss push: z_c / E collapses for the three E up to ~3-4 crossings (linear regime), +E direction, z_c ~ 10-15 E by
  t ~ 4. After that no acceleration: E = 0.01 settles at a steady offset z_c ~ +0.2-0.3 cells (~25-30 E) to t = 60;
  E = 0.001 / 0.003 wander by ~0.05-0.15 cells independent of E (sensitive-dynamics floor), so their z_c / E is noise.
  A bounded displaced equilibrium, not a free particle accelerating under a force (selection criterion 2 not met).
- Dipole polarisation (sum x u): no linear response; |sum x u| ~ 10-40 absolute for all E (noise-dominated).
- E = 0 controls: the symmetric states are UNSTABLE. Symmetry-forbidden components grow from roundoff: dipole
  x-symmetry 1e-17 -> 1e-2 (x_c core 0.02 cells) by t = 60, e-fold ~1.5 crossings; gauss p_z 1e-16 -> 1e-8 then
  saturates. This is the "sensitive dynamics" of 1a made quantitative: after ~10-20 crossings the late state is set by
  amplified perturbations, not by E.
Retention is unchanged by E (Ecore 0.44-0.53 gauss, 0.27-0.34 dipole at t = 60).

## Verdict (conditional)
In this rule a held uniform field produces a short linear transient and then a bounded offset of the lump (gauss) or
nothing measurable (dipole), under a background of exponential symmetry breaking. No free translation under force, no
response coefficient to extrapolate to real field strengths.
