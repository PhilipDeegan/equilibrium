# Hertzian dipole on AE, h=0.125 (2026-09-26)

Same setup as 09-25 (`dipole_radiation.py`, R=20, lam=4, src_w=0.6, C++ `step_ae`),
N=22.3M, 12604 steps, 641 measurements over [4.79, 8.08]. Plots: 101 frames in
`dipole_R160_h0.125_plots/`; 11 full-state snapshots in `/bk/dst/2026/09/26/dipole_R160_h0.125/`.

Reference: gaussian-source dipole, F^2 P_th = exp(-(k s)^2) P_th = 0.4114 P_th.

| h | source work / P_th | shell flux r=5 / 6.5 / 8 / 9.5 | eq. slope |
|---|---|---|---|
| 0.5 | 0.598 | 0.866 / 0.932 / 0.934 / 0.801 | -2.40 |
| 0.25 | 0.444 | 0.474 / 0.477 / 0.476 / 0.475 | -1.994 |
| 0.125 | 0.419 | 0.427 / 0.428 / 0.427 / 0.427 | -2.000 |

- Source work error vs 0.4114: 0.187 / 0.033 / 0.0076 (ratios 5.7, 4.3): 2nd order.
  Richardson (p=2) from 0.25/0.125: 0.4107, reference to 0.2%.
- Shell flux error: 0.064 / 0.016 (ratio 4.0). Richardson: 0.411.
- Flux - source work: 7% at h=0.25 -> 2% at h=0.125 (the flux measurement's own error).
- Flux flat across shells to 0.2%: no sign of wall reflection in the window.
- pole/eq = 0.059 and pattern resid = 0.017 are exactly the 20-degree-bin average of
  sin^2 vs the centre-value fit (0.0597, 0.0174): the pattern is sin^2 to within binning.

Scope: prescribed J, EXPERIMENTAL A-E extension of step(); dipole along z (cubic axis),
off-axis orientation untested.

## R160 full-field comparison with the exact retarded field (`r160_vs_exact.py`, `r160_phase_amp.py`)

Exact: outside a spherically symmetric source the field is the point dipole times F = exp(-(ks)^2/2);
after the one-period ramp p(t) = -I0 cos(wt)/w exactly (no static charge). A, u analytic; E = -A_t - grad u,
B = curl A; phi taken half a step ahead of u and E. Shells r = 3..9.5 (|r - R| < h), window snapshots
7562..12604. Errors normalised by the period-rms amplitude (exact at t and t + T/4).

- **Pointwise convergence**: e(h=0.25)/e(h=0.125) on the 342k shared sites (2x BCC is a sublattice), same
  t_end = 8.079: A 4.1-4.3, B 4.2-4.4, E 4.1-4.4, u 3.7-3.9 over 2.5 < r < 8.5 (8.5-10: 3.9-4.7, reflection
  front nearby). Whole field, near + far terms, 2nd order.
- **Error is almost all phase**: fit X_num = amp X_exact(psi - delta). E, B: delta = 0.0057 kr (numerical
  phase speed 0.57% slow at kh = 0.196, i.e. ~0.15 (kh)^2), amp 1.015-1.02 (-> flux +3-4%, matching the
  in-run 0.427/0.411). Residual after removing phase + amp: 0.4-1.2%. So the pointwise error grows linearly
  in r (3.5% at r=3 -> 9% at r=9.5) while the flux stays at 2-4%.
- **Fields lag differently**: E, B 0.0057 kr, A 0.0042 kr (A residual 1-5%: reconstruction shape error),
  u 0.0012 kr, u amp 1.0035. u (step()'s face difference) and E/B (LS reconstruction) have different
  dispersion; Lorenz / the scalar potential holds to O(h^2) but not in phase. Same root as the curl-grad leak
  (one reconstruction for both would give one dispersion relation) -- a concrete, measurable symptom for
  `next` item 5.
- **Scalar potential (Lorenz check, item 1c)**: u vs the analytic retarded potential of the oscillating
  charge: error/amp 0.4-2.5%, converging ~3.8x per halving.
- **No late drift (item 1a)**: E error/amp at r=9.5 in same-phase snapshots 7562/10083 0.049/0.034;
  alternating snapshots differ because phase error is largest at field nodes. The final snapshot's r=9.5
  fit shifts (amp E 1.030 vs B 1.005): start of the wall reflection (front at ~10.5 at t_end, by design).
  Instantaneous shell flux error +-0.1-0.18 F^2 P_th alternates in sign with phase (reactive + phase error),
  no trend.
- **Lattice anisotropy at fixed (r, th=90 deg) (item 1d)**: error/amp vs azimuth has a 4-fold modulation,
  +-1% of the error at r=3-5 and +-4.5% at r=6.5-9.5 (absolute +-0.004 of amplitude at r=9.5): the phase
  speed is direction dependent at the ~5% level of its own O(h^2) error. Grows with r, so it is dispersion,
  not the source.
- **Grid-scale content at the shells (item 1e, partial)**: the residual after the 2-parameter phase/amp
  fit bounds everything not sinusoidal-in-psi, grid scale included: <= 1.2% of amplitude (~1e-4 of the
  energy) for E and B at r = 3..9.5. Not measured: the source region, and u's grid-scale part (curl-grad
  leak) there.

## Off-axis dipole, [111] (`--axis 1,1,1`, `dipole_R80_h0.25_ax111_cpp.log`, plots `dipole_R80_h0.25_ax111_plots/`)

Script now takes `--axis` (source direction; theta from it; slice plane contains it; in-run flux also binned
in 8 azimuths about the axis). Snapshots in `/bk/dst/2026/09/26/dipole_R80_h0.25_ax111/`.

| h | axis | source work | shell flux r=5 / 6.5 / 8 / 9.5 | eq. slope | pattern resid | eq. azimuth spread |
|---|---|---|---|---|---|---|
| 0.5 | z | 0.598 | 0.866 / 0.932 / 0.934 / 0.801 | -2.40 | | |
| 0.5 | [111] | 0.598 | 0.866 / 0.932 / 0.931 / 0.802 | -1.92 | 0.04-0.11 | 0.008-0.042 |
| 0.25 | z | 0.444 | 0.474 / 0.477 / 0.476 / 0.475 | -1.994 | 0.016-0.019 | |
| 0.25 | [111] | 0.444 | 0.474 / 0.476 / 0.476 / 0.475 | -2.000 | 0.019-0.021 | 0.001-0.002 |

- Orientation independent: source work identical to 3 digits, flux within 0.2%, at both h.
- Time-averaged flux vs azimuth about [111]: spread 0.1-0.2% at h=0.25 (1-4% at h=0.5): converging.
  Consistent with the on-axis R160 finding that anisotropy lives in the phase speed at ~5% of its own
  O(h^2) error, which the time-averaged flux barely sees.
- Visual: final frame clean, lobes along n, no visible reflection in the window.
