# Hertzian dipole convergence on AE (2026-09-25), `next` 09-25 item 2

`dipole_radiation.py` (copy of 09-24's, `--cpp` uses C++ `continuous::step_ae`,
numpy-identical at h=0.5 to every printed digit; also measures the source work
-<J.E> = (3/80) V sum_faces J_k e_k over the window). R=20, lam=4, src_w=0.6.

## The 09-24 reference was wrong for this source
P_th = I0^2 w^2/(12 pi c^3) is the POINT dipole. The source is a gaussian of
width s=0.6 at k = w/c = 1.571: retarded form factor F = exp(-(ks)^2/2) = 0.641,
so the right reference is F^2 P_th = 0.411 P_th. Checked directly: far-field
A_z (equator, r=4-12, final state, h=0.25) fits I(t-r/c)/(4 pi c^2 r) with
amplitude 0.642 of the point value (fit resid 10%).

## Results (all / P_th point; corrected reference 0.411)
| h | N | source work | shell flux r=5 / 6.5 / 8 / 9.5 | eq. slope | pattern resid |
|---|---|---|---|---|---|
| 0.5 | 348k | 0.598 | 0.866 / 0.932 / 0.934 / 0.801 | -2.40 | 0.02-0.13 |
| 0.25 | 2.78M | 0.444 | 0.474 / 0.477 / 0.476 / 0.475 | -1.994 | 0.016-0.019 |
- h=0.5 is not resolved: flux exceeds the source work by 45-56% and varies
  with r -- the 09-24 "P/P_th 0.87-0.93, first classical match" was a
  coincidence of an unresolved flux measurement and a wrong reference.
  RETRACTED.
- h=0.25: flux flat in r to 0.6%, slope -1.994, sin^2 pattern: a conserved
  radiating field. Source work 0.444 vs 0.411 (+8%), converging from +46%
  (error ratio 5.7 per halving, 2nd order; Richardson ~0.433, +5%).
  Shell flux 7% above the source work: the reconstructed-E x curl_A flux
  measurement's own error at h=0.25 (09-24 Poynting balance: 4.5% there).
- Window: 3 periods at h=0.25 vs 2 at h=0.5 (t_end has a -3h/c margin).
- h=0.125 (22M cells) needs ~25GB+; not attempted. A third point via a
  longer wavelength / wider source at h=0.5/0.25 is the cheaper next check.
