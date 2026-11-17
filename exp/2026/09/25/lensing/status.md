# Lensing / Shapiro delay, depletion = cell size + tick rate (2026-09-25)

Philip: lam = substrate depletion -> gradient in cell size -> both spatial and
time-rate contraction; smaller cells toward the centre have less throughput
(tick rate ~ cell size). Far view: cell size ell = 1 - eps, tick rate
sigma = ell, light one cell per tick -> far-view speed ell^2 ~ 1 - 2 eps =
GR isotropic weak field (full 4GM/bc^2). Time-only (today's sigma(lam)) gives
half; smaller cells with FASTER ticks gives none (Nordstrom 1913).

`lensing.py`: continuous::step at level 0 with per-cell L (new: symmetric
pass-2 face coefficient, `set_leaf_L`); sigma = tick rate, L = mesh hop in
local cells, speed ~ sigma/L. Plane u/phi pulse (step()'s longitudinal wave)
past eps = mu/sqrt(r^2+a^2); arrival-time centroid per detector cell minus
vacuum baseline; reference = straight-line eikonal delay.

| run | T dT(0) (ref) | GR dT(0) (ref) | GR/T | N |
|---|---|---|---|---|
| mu=0.3 a=3, bare gaussian w=3 | 0.316 (0.549) | 0.637 (1.126) | 2.02 | 0 |
| mu=0.8 a=8, carrier lam=5, w=5 | 1.241 (0.975) | 2.552 (2.016) | 2.06 | 0 |

- GR/T = 2 (+2-6%, second order at eps0 = 0.1) and N = 0 exactly: the kernel
  propagates at sigma/L as intended. At first order in eps this ratio is
  algebraic (the GR arm's operator perturbation is exactly twice T's), so it
  confirms the implementation, not new physics.
- Absolute delays are NOT eikonal (0.57x bare, 1.27x carrier; lattice group
  index at lam=5 is only 1.09): lens scale is comparable to the Fresnel scale
  (sqrt(lam D) ~ 10 vs a = 8), i.e. wave-optics regime. Deflection alpha(b):
  bare pulse wrong sign (diffraction-dominated), carrier right sign, noisy.
  Reaching ray optics needs a >~ 20 (~15M cells) -- a numerics check of the
  measurement, not a physics test.
- The physics content is the input relation sigma = ell (tick rate ~ cell
  size) and eps(depletion); it has to come from the substrate rule, which
  the simulation cannot supply.
- Known limits (Philip): substrate dx likely quantized (smallest cell);
  unequal adjacent cells distort the junction. Per-cell L = coarse average;
  junctions not modelled; ignore for now.
- `run/substrate/eddington.py` (2026-05) is stale: calls
  build_bcc_sphere(..., ell0=...), removed in the alpha/ell sweep.

## Face medium (2026-09-25 evening, Philip: L = coarse size / mean discrete size)

continuous::step no longer reads per-cell L. The medium is sampled at the face
anchors from the continuum field, as deviations from vacuum
(`set_face_medium(dsig, dn)`, n = L/h = 1/<ell>); coarse cells stay uniform.
Replaces the junction average 2/(L_i+L_nb). With no medium: sigma_f = midpoint
interpolation of 1/(1+lam), = discrete at level 0.
Rerun mu=0.3 a=3 (`lensing_mu0.3_face.log`): T 0.3157, GR 0.6369, GR/T 2.017,
N 0 -- unchanged from the cell-L run (O(h^2) difference for smooth eps).
Tests: `tst/substrate/test_continuous_cpp.py` TestFaceMedium.
Precision: deviations are stored and entered directly, so the coefficient
keeps eps to ~1e-16 relative, but u/phi still accumulate in float64; eps
below ~1e-12 needs a perturbative (base + deviation) evolution, not storage.
