# Electron IC falsification battery (2026-09-24)

Goal: rule electron IC candidates OUT, cheapest decisive test first. Electrons
only (protons deferred). Working model (Philip, 09-24): electron = "glut",
excess charge like static buildup; target is a steady-state equilibrium, so
decay is a failure mode. IC should fill the domain (~R60, BC ~R70).

Each test is pass/fail; an IC that fails any test is out, later tests are only
run on survivors.

| # | Test | Cost | Fail criterion |
|---|------|------|----------------|
| T0 | Net charge at t=0 (`t0_charge.py`) | seconds | Q through enclosing shells ~0 (neutral) |
| T1 | Charge persistence: Q(R_s) vs t under plain step(), outflow BC at R70 | ~hours at R70; screen at R35 | Q(t) decays / leaks out of domain |
| T2 | Energy retention inside r<R_IC, outflow BC | same run as T1 | E_inside(t)/E0 keeps falling (no plateau near 1) |
| T3 | Shape steady state: radial u(r), energy profile drift | same run | profile not converging |
| T4 | Perturbation response: small amplitude/displacement kick on a T1-T3 survivor | same cost as T1 | perturbation grows or disperses the IC instead of relaxing |
| T5 | Resolution: repeat verdicts at 2x scale | 8x cost | verdict changes with resolution |

**T6 -- known-quantity clues (survivors only, never tuned for).** Philip: a
working geometry will likely show clues in known quantities. Only
dimensionless ratios (lattice units/scale are not fixed):
- exterior-field size vs internal oscillation wavelength ~ r_e/lambdabar_C = alpha
- circulation radius vs own measured frequency (omega*R/c) ~ Zitterbewegung 1/2
- moment / (charge x angular momentum) ~ g = 2.0023
- (a0/lambdabar_C = 1/alpha, m_p/m_e = 1836 need a proton -- later)
Record whether each emerged from unaided dynamics or needed a tuned parameter;
only the former counts as a clue.

**IC geometry is also preliminary (Philip, 09-24).** Every verdict must say
whether it rules out the construction CLASS (field content/topology: e.g.
uniform u, div-free phi => Q=0 for ANY shape) or only THIS GEOMETRY (ring
radius, tube width, scale, pitch). A geometry-specific failure is not a
ruling on the class; only class-level failures remove a family outright.

Out of scope for now: step_cross-based tests (translation mechanism, kept out
by Philip), e/p inertia ratio (needs a proton).

Known going in: under the reversible step(), a static state needs du=0 across
every face, so a free u excess with no source cannot be static -- T1 on any
pure-u glut is expected to fail unless phi structure holds it. That is the
test, not an assumption to design around.

## T0 result (2026-09-24, `t0_charge.log`) -- all 8 existing candidates NEUTRAL

| IC | u | phi | Q on enclosing shells |
|---|---|---|---|
| init_electron_dipole_ic | uniform | div-cleaned | 0 (rel 8e-16) |
| init_electron_dipole_ic_vecpot | uniform | A.dl, div-cleaned | 0 (rel 2e-16) |
| init_spheromak_ic (reference only) | internal structure | Taylor | 0 exactly outside r=50 |
| seed_dipole_ic | l=1 (odd) | grad U | 0 at every shell |
| seed_curl_full_dipole_ic | l=1 | curl-full | 0 |
| seed_curl_full_pair | 0 | 1-cell filaments (don't cross most shells) | 0 |
| bare current loop | 0 | A.dl | 0 (rel 2e-17) |
| init_toroidal_winding octagon | cos(theta) | cos(theta_mid) | 0 |

**Class-level, not geometry-specific**: every construction is either uniform/
odd-parity u (no l=0 excess) plus div-free phi -- magnetic-dipole content only.
Net shell flux Q = sum W du (or sum W phi) is then zero for ANY shape, scale or
pitch. So the whole existing gallery is ruled out as an electron at the
content level, not just these geometries. What's missing is an l=0 u excess
(the "glut") or a phi with net divergence -- no IC in the repo has either.
Next: build that class (u glut, optionally + phi winding) and run T1.

## Vacuum checks + bare glut, R35, outflow BC (`t1_vacuum_and_glut.py`, `t1_*.log`)

- **V0 uniform u=0.1: FAILS.** Outflow BC drains a uniform background (u
  0.1 -> ~0.06 by t=23, with spurious phi/Q structure). `outflow` is not
  transparent to a nonzero constant state, so every run with a nonzero
  background or a 1/r tail under outflow is contaminated (V3, E1 below).
  Short residual tests are nearly immune (c~3.16 cells/t, 20 steps = 0.3 cells).
- V1 free pulse: ~94% of energy leaves by t~15; ~6% stays trapped (reflection).
- **V2/V3 held core (fixed_u) + exterior: u(r) does form a ~1/r profile, but
  E_phi grows without bound (~t^2: 3.7e4 -> 3.0e5 by t=23).** In step()'s
  first-order form (du ~ -sum W phi, dphi ~ face du), a static u gradient
  drives phi linearly forever: dphi/dt = -grad u != 0 at every exterior face.
  A u-gradient Coulomb field is therefore NOT a steady state of step() --
  class-level, any geometry. The static-exterior charge field of step() is
  instead radial phi ~ 1/r^2 with uniform u (div phi = 0 outside the core ->
  du/dt = 0; du = 0 across faces -> dphi/dt = 0). Contradicts doc/dst.md's
  diffusion-form Coulomb (u ~ 1/r) -- see
  project_electron_u_phi_convention_conflict; Philip's call which is right.
- E1 bare glut (a=25): drains/sloshes, not interpretable given V0's BC failure.

### Separation check (Philip: "are we sure 1 and 2 aren't related?") -- they aren't
`sep_{V0,V2}_{outflow,neumann,none}.log`, 2400 steps:
- V0 uniform u: exact fixed point under neumann and no-BC (u=0.1 forever);
  only outflow drains it (from t~5, when the edge signal reaches r<15).
  => (1) is purely an outflow-BC artefact. Use neumann for background runs.
- V2 held core: E_phi(r<15) BIT-IDENTICAL across all three BCs through t=11
  (1.0e3, 6.7e3, 1.6e4, 2.9e4, 6.8e4 at t=1,3,5,7,11 -- ~t^2), u(r) static
  ~1/r immediately. => (2) is intrinsic to step(), not BC-driven: an imposed
  static u gradient makes phi ramp without bound.

## Hop-path rings can't be circular (`pi_eff_rings.py`) -- class-level fail
Philip: square/octagon may not be circular enough for the pi-precision an
electron's toroidal flux needs; "any sharp corners are suspect".
pi_eff = hop-path length / (2 * mean radius):
square 8-tile 3.314 (+5.5%); octagon hops=2/4/8/12/16: 3.612/3.694/3.734/
3.747/3.754 (+15.0% -> +19.5%, CONVERGING AWAY from pi), radius ripple +-7%.
Staircase paradox: a hop path measures circumference in the lattice metric,
so no hop-path winding (square, octagon, hop-built helical torus knot) is
pi-exact at any size/resolution. Class ruled out. Smooth sampled fields
(t2 torus, cos^2 profile, C1 at the tube surface, no corners) are the
remaining class; their error is dispersion anisotropy, which should vanish
as wavelength/cell -> inf -- measure pi_eff from the t2 pattern rotation rate.

## T2 travelling torus, m=4, R70 neumann (`t2_{linear,drag}_m4.log`)
- **E/E0 invalid** (swings 1.5-3.2 under the LINEAR step): u^2+0.5*sum phi^2
  is not step()'s conserved norm when all 14 faces carry flux (W=3/1 weighting
  presumably needed). Find the real conserved form before using E again.
- **Not travelling**: u_mode cos component 0.000 throughout -> standing wave.
  Quadrature seeding was wrong; dphi/dt ~ du/dtheta needs u, phi IN PHASE.
- **No charge either arm**: Q_u on 3 exterior shells <= 1e-12, <u> tube/ext
  ~1e-18. Linear no-go confirmed numerically. Drag lam=kappa*|phi|^2 can't
  rectify by symmetry (|phi|^2 has harmonics 0,2m; times the m-harmonic
  divergence never gives m=0) -- any rectifying drag must depend on flux
  direction/phase/chirality, not magnitude.
- Drag runaway: lam_max -> ~35 (sigma ~0.03) in hotspots, tube mean ~1.5.

### T2 correction: my seeding bug, not physics
`_FACE_HOP` is the FULL neighbour offset (verified: nb offset == _FACE_HOP for
all 14 faces), despite adaptive_flux_tube.py's "hop to face midpoint" comment.
Core seeders already use pos + 0.5*_FACE_HOP (correct); my t2 script used
pos + _FACE_HOP -> phi not antisymmetric (residual 0.31 of max) -> E grew 3x.
u^2 + 0.5*sum phi^2 IS the conserved norm of step() (derived from
_step_pass2_uniform; needs phi_jk' = -phi_ik). Fixed + in-phase seeding: R30
check E/E0 = 1.0000 +- 2e-4 over 1200 steps, mode (c,s) rotates (travels),
period ~5.1 t; still radiates (Enear 0.62 by t=6). The earlier t2 E/E0 and
"standing wave" results are void; re-running R70 (`t2_*_m4_fixed.log`).
Note for pi_eff: long-wave speed from the update is c^2 = 1/2 sum_k a_k^2 d_kx^2
= 13.33 -> c ~ 3.65 (not outflow's default c_lattice=sqrt(10)); a clean pi_eff
needs a thin tube (A << R0), the A=8/R0=15 check mixes radii.

### T2 fixed-seeding R70 results (`t2_{linear,drag}_m4_fixed.log/.npz`)
| | linear | drag (lam = |phi|^2, tau=1) |
|---|---|---|
| E/E0 | 1.0000 (conserved) | 1.0006 (sigma varies) |
| Enear/E0 t=10-20 / 20-31 | 0.59 / 0.71 (radiates; returns off neumann wall) | 0.91 / 0.94 |
| m=4 mode amplitude | travels, 0.30 -> 0.14-0.28 | breaks up, -> ~0.03-0.06 |
| Q mean on shells 58/62.5/67 (t=20-31) | +0.4/+2.3/+0.7, rms 7-10 | +0.8/-6.3/+3.6, rms 33-65 |
| <u>tube | ~1e-5, zero-mean | drifts to -7e-4 |
| lam_max | 0 | runaway to ~54 |
- **No charge in either arm**: shell Q mean << its rms, sign inconsistent across
  shells -> wave noise, not a DC Coulomb flux. Linear no-go confirmed with a
  genuinely travelling wave.
- **Drag confines**: flux-sourced lam (~0.5 in tube) keeps 91-94% of energy
  near the torus vs 59-71% linear -- a waveguide made by the flux itself. But
  the organised m=4 travelling mode doesn't survive (scatters), and lam runs
  away in hotspots (no cap). Small DC u drift is consistent with the
  sigma-weighted update no longer being a pure divergence, not a charge.

### T2 saturated drag + reflection-free linear, R70 OUTFLOW (`t2_*_m4_out*.log/.npz`, `*_final.npz` = full u/phi/lam)
Outflow is valid here (u background 0; V0's drain needs a nonzero constant).
lam target = kappa*P/(1+P/lam_sat), P = sum phi^2/amp^2 (EXPERIMENTAL, lam_sat picked).

| arm | E_total t=31 | Enear t=10-20 / 20-31 | m=4 amp (0.30 at t=0) t=10-20 / 20-31 | lam_max | <lam>tube |
|---|---|---|---|---|---|
| linear | 0.48 | 0.55 / 0.44 | 0.093 / 0.054 | 0 | 0 |
| drag uncapped | 0.92 | 0.92 / 0.90 | 0.099 / 0.058 | 54 (runaway) | 0.52 |
| drag lam_sat=1 | 0.70 | 0.79 / 0.64 | 0.189 / 0.102 | 0.8 | 0.21 |
| drag lam_sat=4 | 0.86 | 0.89 / 0.82 | 0.194 / 0.141 | 2.7 | 0.33 |

- Neumann-wall reflection inflated the old linear Enear (0.59-0.71 -> 0.44-0.55
  here; half the energy leaves the domain by t=31). Linear torus radiates, confirmed.
- Uncapped drag's confinement is real (not reflection): 0.90 with outflow. But
  it holds energy WITHOUT holding the circulation -- m=4 amp decays as fast as
  linear (0.058 vs 0.054). It traps disorganised energy in lam hotspots.
- Saturation trades confinement for organisation: lam_sat=4 keeps the m=4 mode
  2.5x better than uncapped/linear (0.141) at Enear 0.82, no runaway. Still a
  monotonic decay in every arm -- no plateau, not a steady state. FAIL (T2/T3).
- Charge: shell Q mean << rms, signs flip across shells and windows in all four
  arms -> no DC Coulomb flux. FAIL (T1), as the symmetry argument predicts
  for any |phi|^2-magnitude drag, saturated or not.
- Class scope: rules out |phi|^2-magnitude drag (any saturation form) as a
  charge mechanism -- class-level by symmetry. Confinement verdict is
  geometry-specific (one torus R0=35/A=20/m=4, one kappa/tau).
