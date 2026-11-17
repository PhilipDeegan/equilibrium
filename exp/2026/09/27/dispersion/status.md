# Plane-wave dispersion on step_ae (level 0 continuous, C++)

`plane_wave.py` (packet, lam=4, envelope s=1.2, travels 6; phase of the k-projection in a cylinder
rho<4 vs exact), `sweep.sh`, raw rows `results.jsonl`, log `sweep.log`. Test:
`TestPlaneWaveDispersion` in tst/substrate/test_continuous_cpp.py.

dc/c (phase speed error; all slow), coefficient dc/c / (kh)^2 in brackets:

| kh | T x (E) | T [111] (E) | L x (u) | L [111] (u) |
|---|---|---|---|---|
| 0.785 | -0.101 (0.164) | -0.071 (0.114) | -0.0111 (0.0179) | -0.0190 (0.0307) |
| 0.393 | -0.0269 (0.174) | -0.0184 (0.120) | -0.00278 (0.0180) | -0.00477 (0.0309) |
| 0.196 | -0.00678 (0.176) | -0.00462 (0.120) | -0.00070 (0.0182) | -0.00120 (0.0312) |

- Method validated: measured u dc/c = exact leapfrog step() symbol prediction to <1% at every h/dir.
- Both sectors converge 2nd order. T sector (A, E, B agree within ~5%): coefficient ~0.17 along x,
  ~0.12 along [111]. L sector (u, and A_L with it): 0.018 along x, 0.031 along [111].
- T/L ratio: 9.5x along a cubic axis, 3.9x along [111]; anisotropies go opposite ways.
- Longitudinal wave leaks essentially nothing into E for smooth fields: E/(k u) 4e-6, 7e-8, 7e-10
  (much faster than h^2; consistent with the curl-grad leak being grid-scale only).
- Matches the dipole R160 lags at kh=0.196 (E/B 0.0057 = between x 0.0068 and [111] 0.0046;
  u 0.0012 = [111]-like, the dipole's u lies off-axis).

Meaning: in the Lorenz-gauge picture the scalar/gauge sector (u) and the transverse sector (E, B)
should travel at the same c. At finite h they don't: the transverse sector, whose curl curl goes
through the per-cell LS reconstruction, is 4-10x more dispersive. Input for the reconstruction
decision (Philip's call): unify the reconstructions (one dispersion by construction), or accept
two O((kh)^2) lags, or aim the T sector at step()'s accuracy.

## Pinned down: exact Bloch symbol (`symbol.py`)

Symbol of C0SQ * curl_curl_sym on the infinite lattice (7 owned faces per cell per k) reproduces the
measured T-mode dc/c: kh=0.196 x -0.00678 vs -0.00678, [111] -0.00461 vs -0.00462; kh=0.393 to <0.5%;
kh=0.785 to 2-3% (packet bandwidth). u sector = step()'s symbol, exact everywhere.

Small-kh coefficients dc/c / (kh)^2:

| n | pol | T total | LS gradient g | reconstruction rest | u (step) |
|---|---|---|---|---|---|
| [100] | any | -0.176 | -0.139 (= -5/36) | -0.038 | -0.018 |
| [111] | any | -0.120 | -0.102 | -0.018 | -0.031 |
| [110] | z | -0.149 | -0.111 | -0.038 | -0.028 |
| [110] | [1-10] | -0.119 | -0.111 | -0.008 | -0.028 |

1. 80-85% of the transverse error is the gradient in curl_K: LS fit of neighbour differences of
   cell-centred A. Bloch: g = P_g (e^{i theta} - 1)/h = i sum_m P_g[:,m] sin(theta_m)/h -- a centred
   difference across two cells (span 2h), error theta^2/6, vs step()'s compact face difference
   (span h, theta^2/24). Along x analytically k_eff/k = 1 - (5/36)(kh)^2.
2. The rest is the face -> cell reconstruction (cos(theta/2) averaging, applied in K and again in
   K^T); it depends on polarization: **birefringence** along [110] (-0.149 vs -0.119 (kh)^2; at
   kh=0.785 the two polarizations differ by 1.6% in speed). Degenerate along [100], [111].
3. 5 of the 7 face modes per k have omega = 0 exactly: the longitudinal one plus 4 that K cannot see
   at all (K reads phi only through the 3-vector A_rec; 7 - 3 = 4). These are the anchor-residual
   DOFs: no restoring force, they don't propagate (cf. current_from_magnetisation's null(K) note).
   A sampled smooth plane wave puts 0.05% (kh=0.196) - 0.8% (kh=0.785) of its amplitude there,
   2nd order, up to 3x more along [110].

Options (Philip's call; nothing changed):
- compact curl: circulation of phi around the triangles of the BCC Delaunay tets (DEC d1, Hodge
  stars from the Kelvin faces). Compact like step() (expect ~u-sector dispersion), curl(grad) = 0
  exactly (removes the curl-grad leak by construction), but the 4 extra face modes then get
  nonzero (grid-scale) frequencies instead of zero -- need its symbol before deciding.
- keep LS but take the gradient from face values directly (compact) instead of cell-centred A.
- accept: both sectors 2nd order; the mismatch is a 4-10x difference in the (kh)^2 coefficient.

## Compact alternative: DEC on the BCC Delaunay / Kelvin complex (`dec_symbol.py`, symbol only)

1-forms on the 14 hops (7 per cell), 2-forms on the Delaunay triangles (12 per cell, all hex-hex-square),
Hodge stars from the Kelvin cell: *1 = |face| / |hop| = 3/8 (hex) : 1/8 (square), *2 = |Kelvin edge| /
|triangle| = 1, *0 = cell volume. curl curl = *1^-1 d1^H *2 d1.

- d1 d0 = 0 to 1e-16: curl(grad) exactly zero (the curl-grad leak is gone by construction).
- Continuum limit omega^2/k^2 = 1 for both transverse modes and for the 0-form Laplacian, with
  purely geometric stars: no fitted constant (LS needed the fitted C0SQ scale).
- Transverse dc/c / (kh)^2: [100] -0.0087, [111] -0.0110, [110] -0.0122 / -0.0087 (birefringent),
  [123] -0.0115 / -0.0094. 15-20x smaller than LS (-0.176 / -0.120), ~2x smaller than step()'s u.
- Exactly 1 zero mode per k (the gradient). The other 4 face DOFs become gapped optical branches at
  omega h / c = 4.9-5.7, nearly flat (non-propagating grid-scale oscillation) -- instead of LS's 4
  frozen omega = 0 modes. Leapfrog bound omega_max dt < 2 -> dt < 0.35 h / c, far above stable_dt.
- A smooth sampled transverse wave puts 6e-4 (kh=0.196, [111]) of its amplitude in the gapped modes,
  2nd order; (kh)^4 along [100]. Same size as LS's null-mode fraction.

Side finding: step()'s u Laplacian weights are W^2 = 9 : 1 (W on both legs), while the Kelvin Hodge star
ratio is 3 : 1 = W. So W itself is the geometric DEC star, but step() applies it twice. With the
geometric 3:1 the u sector would be -0.0243 [100] / -0.0289 [111] (anisotropy +-9% vs step's +-26%).
Both are consistent (any ratio is isotropic at O(k^2)); they differ at O((kh)^2). Philip's call.
