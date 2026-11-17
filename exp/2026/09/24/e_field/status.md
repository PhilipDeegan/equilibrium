# E-field test (2026-09-24) -- is an E field worth adding?

Convention phi ~ A, u ~ V (Philip). Under it step() is the E=0 sector:
u_t = -c^2 div A (Lorenz condition), A_t = -grad u. EXPERIMENTAL extension
`equilibrium.substrate.continuous.AE`: E stored at the face anchors (same
sampling as phi), E_t = c^2 curl curl A (grad div - BCC Laplacian of the
reconstructed A), A_t = -grad u - E. Boundary: E held 0 within 2 layers of
the edge (undecided). Numpy only.

## Waves (`ae_waves.py`, `ae_waves_dt1.log`), R=8, gaussian w=1.2, T=0.6
| pulse | model | errA h=0.5 | errA h=0.25 |
|---|---|---|---|
| transverse (A ⊥ k, u=0) | step() | 1.037 | 1.040 (frozen) |
| transverse | AE | 2.02e-2 | 5.24e-3 (3.9x) |
| longitudinal | step() | 2.0e-2 | 5.1e-3 |
| longitudinal | AE | 5.5e-2 | 1.4e-2 (3.9x, ~3x worse than step()) |
Longitudinal penalty: discrete curl curl of a gradient vanishes only to O(h^2).

## Poynting balance (`poynting_balance.py`, `poynting_balance_h*.log`)
IC u=0, A=0, E = curl(z G), gaussian s0=1; sphere Rs=4, R=10, T=1.2.
U = 1/2 (|E|^2 + c^2 |B|^2), S = c^2 E x B (both from reconstructions).
| h | max |U0 - U_in - int F dt| / U0 | total EM energy drift | max |u| (spurious) |
|---|---|---|---|
| 0.5 | 12% | 9% | 0.11 |
| 0.25 | 2.4% | 1.4% | 0.03 |
Poynting's theorem holds and converges with refinement (under step() alone the
local Poynting check correlated at -0.01). Spurious u generation shrinks ~3.5x.

Tests: `tst/substrate/test_continuous.py::TestTransverseWaveNeedsE`.
Open: J/rho source (none yet), E boundary treatment, energy of the full
(u, A, E) scheme (only the EM part measured), whether E is a stored DOF in
DST or emerges.

## CORRECTION (same evening): the first AE operator was unstable -- results above superseded
`curl_curl_faces` (grad div - BCC Laplacian of the reconstruction) is not
symmetric: noise grows ~2.3 e-folds per unit time at dt factors 1.0/0.5/0.35
(`stability*.log`), not a CFL issue. Short tests (T<=1.2) hid it; it caused
the late induction drift and a dipole blow-up (S ~1e4x theory).
Also found: the 2-layer E=0 edge mask breaks the energy invariant and
gradient-orthogonality (off now); random-noise ICs must be antisymmetrised.

Fix: `curl_curl_sym` = C0SQ * antisym(K^T K phi), K = `curl_K` (W-weighted
least-squares reconstruction = Euclidean adjoint of sample_A, then LS
gradient, curl), K^T its exact adjoint (`curl_K_adjoint`, checked to machine
precision). Symmetric PSD, exactly zero on anchor gradients, output exactly
discretely div-free. Scale C0SQ = 40/3 is the measured limit of a fitted
constant (13.380/13.345/13.336 at h=0.5/0.25/0.125). Consistency error vs
sampled curl curl: 4.8% / 0.59% at h=0.5/0.25.
Invariants (`invariants_nomask.log`, antisymmetric noise, 3600 steps): H_T
bounded 0.967-0.993 (half-step measurement wobble, no drift); longitudinal
sector exactly step() (|u| 141.7 vs step() control 141.9).

### Results on the stable operator (`*_sym*.log`)
- Transverse wave errA: 7.24e-2 / 1.82e-2 (h=0.5/0.25, 2nd order; 3.6x the
  unstable version's error). Test reference updated.
- Poynting balance worst |err|: 10% (h=0.5), 4.5% (h=0.25); total EM energy
  drift <=3% / <=0.6%.
- **Hertzian dipole, R=20, h=0.5, lam=4 (8 cells/wavelength), 2 periods**:
  P/P_theory = 0.87 / 0.93 / 0.93 / 0.80 at r = 5 / 6.5 / 8 / 9.5; sin^2 fit
  pattern residual 2-13%; pole/equator 0.02-0.04 (theory 0); equatorial slope
  -2.40 (theory -2; outer shell low -- check resolution / window / wall).
  First classical dipole-radiation match in the Poynting thread.
- **Induction NOT settled**: Phi2/M 1.13-1.31 (hold mean 1.20) and loop 1's
  self-flux Phi1 drifts through zero (0.22 -> -0.19), |u| grows slowly
  (4e-2). Diagnosis (untested): the sampled div-free J has a component in
  null(K) orthogonal to gradients (the 4 non-vector face DOF/cell) that no A
  can balance -> e grows linearly there. Fix to try: J = C0SQ*antisym(K^T m)
  with m a smoothed-disc magnetisation (J = curl m).

## RETRACTED 2026-09-25 (see exp/2026/09/25/dipole/status.md)
The dipole "P/P_theory 0.87-0.93" match: P_theory was the point-dipole value
(gaussian source form factor F^2 = 0.411 missing), and at h=0.5 the shell flux
exceeds the source work by ~50%. At h=0.25 the field is clean (slope -1.994)
and the source work is within 8% of F^2 P_th, converging 2nd order.
Induction: resolved, exp/2026/09/25/induction/status.md.
